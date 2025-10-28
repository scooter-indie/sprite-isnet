# train_sprite_isnet.py - Main training script for sprite IS-Net
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time

# Import IS-Net model (from the DIS repo)
from models import ISNetDIS

# Import custom dataset and config
from sprite_dataset import create_sprite_dataloaders
from config_sprite import SpriteConfig


class HybridLoss(nn.Module):
    """
    Hybrid loss combining BCE and IoU loss.
    Works well for multi-object segmentation like sprites.
    """
    def __init__(self, bce_weight=0.7, iou_weight=0.3):
        super(HybridLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def iou_loss(self, pred, target):
        """IoU (Intersection over Union) loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Intersection and union
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        return 1 - iou.mean()
    
    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        return self.bce_weight * bce + self.iou_weight * iou


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class SpriteTrainer:
    """Trainer class for sprite IS-Net model"""
    
    def __init__(self, config):
        self.config = config
        config.ensure_dirs()
        
        # Verify configuration
        issues = config.verify_paths()
        if issues:
            print("\n✗ CONFIGURATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        
        # Setup device
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() 
                                  else 'cpu')
        print(f"\nUsing device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create model
        print("\nInitializing IS-Net model...")
        self.model = ISNetDIS().to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Load pretrained weights if specified
        if config.USE_PRETRAINED:
            if config.PRETRAINED_MODEL and os.path.exists(config.PRETRAINED_MODEL):
                print(f"\nLoading pretrained model: {config.PRETRAINED_MODEL}")
                pretrained_dict = torch.load(config.PRETRAINED_MODEL, 
                                            map_location=self.device)
                
                # Handle different state dict formats
                if isinstance(pretrained_dict, dict) and 'model_state_dict' in pretrained_dict:
                    pretrained_dict = pretrained_dict['model_state_dict']
                
                self.model.load_state_dict(pretrained_dict, strict=False)
                print("✓ Pretrained weights loaded successfully")
            else:
                print(f"\n⚠ Pretrained model not found or not specified")
                print("Training from scratch with random initialization...")
        else:
            print("\n✓ Training from scratch - no pretrained weights")
            print("  Model initialized with random weights")

        
        # Freeze encoder if specified
        if config.FREEZE_ENCODER:
            print("\n🔒 Freezing encoder weights...")
            frozen_params = 0
            for name, param in self.model.named_parameters():
                if 'encoder' in name.lower() or 'stage' in name.lower():
                    param.requires_grad = False
                    frozen_params += param.numel()
            print(f"Frozen parameters: {frozen_params:,}")
            
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Remaining trainable parameters: {trainable_params:,}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        self.train_loader, self.valid_loader = create_sprite_dataloaders(
            config.TRAIN_IMG_DIR,
            config.TRAIN_MASK_DIR,
            config.VALID_IMG_DIR,
            config.VALID_MASK_DIR,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            target_size=config.INPUT_SIZE
        )
        
        print(f"Training batches per epoch: {len(self.train_loader)}")
        print(f"Validation batches per epoch: {len(self.valid_loader)}")
        
        # Setup loss function
        self.criterion = HybridLoss(
            bce_weight=config.BCE_WEIGHT,
            iou_weight=config.IOU_WEIGHT
        )
        print(f"\nUsing Hybrid Loss (BCE: {config.BCE_WEIGHT}, IoU: {config.IOU_WEIGHT})")
        
        # Setup optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if config.OPTIMIZER == 'AdamW':
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            self.optimizer = optim.Adam(
                trainable_params,
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        print(f"Optimizer: {config.OPTIMIZER}")
        
        # Setup learning rate scheduler
        if config.SCHEDULER == 'CosineAnnealing':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.EPOCHS
            )
            print(f"LR Scheduler: CosineAnnealing")
        elif config.SCHEDULER == 'StepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.LR_DECAY_EPOCHS,
                gamma=config.LR_DECAY_RATE
            )
            print(f"LR Scheduler: StepLR (milestones: {config.LR_DECAY_EPOCHS})")
        else:
            self.scheduler = None
            print("LR Scheduler: None")
        
        # Setup tensorboard
        if config.USE_TENSORBOARD:
            log_dir = os.path.join(config.LOG_DIR, 
                                  f"sprite_isnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None
        
        # Setup early stopping
        if config.USE_EARLY_STOPPING:
            self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
            print(f"Early stopping enabled (patience: {config.EARLY_STOPPING_PATIENCE})")
        else:
            self.early_stopping = None
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Train]',
                   ncols=100)
        
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # IS-Net outputs multiple side outputs for deep supervision
            # Calculate loss for all outputs
            if isinstance(outputs, (list, tuple)):
                # Main output (last in list)
                main_loss = self.criterion(outputs[-1], masks)
                
                # Side outputs (intermediate supervision)
                side_loss = 0
                for side_output in outputs[:-1]:
                    side_loss += self.criterion(side_output, masks)
                
                # Combined loss
                loss = main_loss + 0.3 * side_loss
            else:
                loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{epoch_loss/batch_count:.4f}'
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config.LOG_FREQ == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        avg_loss = epoch_loss / len(self.train_loader)
        
        # Log epoch metrics
        if self.writer:
            self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
            self.writer.add_scalar('train/learning_rate', 
                                  self.optimizer.param_groups[0]['lr'], epoch)
        
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['learning_rate'].append(
            self.optimizer.param_groups[0]['lr']
        )
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        pbar = tqdm(self.valid_loader, 
                   desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Valid]',
                   ncols=100)
        
        with torch.no_grad():
            for images, masks, _ in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                # Use final output for validation
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[-1]
                
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(self.valid_loader)
        
        # Log validation metrics
        if self.writer:
            self.writer.add_scalar('valid/loss', avg_val_loss, epoch)
        
        self.training_history['valid_loss'].append(avg_val_loss)
        
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'sprite_isnet_epoch_{epoch+1:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f'  Checkpoint saved: {os.path.basename(checkpoint_path)}')
        
        # Save best model (just state dict for easier loading)
        if is_best:
            best_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                'sprite_isnet_best.pth'
            )
            torch.save(self.model.state_dict(), best_path)
            print(f'  ✓ Best model saved: {os.path.basename(best_path)}')
        
        # Save latest model (for easy resumption)
        latest_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            'sprite_isnet_latest.pth'
        )
        torch.save(checkpoint, latest_path)
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        self.config.print_config()
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.VALIDATE_FREQ == 0:
                val_loss = self.validate(epoch)
                
                # Check if best model
                is_best = val_loss < self.best_loss
                if is_best:
                    improvement = self.best_loss - val_loss
                    self.best_loss = val_loss
                    print(f'  🎯 New best model! (improved by {improvement:.4f})')
                
                # Early stopping check
                if self.early_stopping:
                    self.early_stopping(val_loss)
                    if self.early_stopping.should_stop:
                        print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                        print(f"   No improvement for {self.early_stopping.patience} epochs")
                        break
                
                # Save checkpoint
                if epoch % self.config.SAVE_FREQ == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            else:
                # Just save periodically even without validation
                if epoch % self.config.SAVE_FREQ == 0:
                    self.save_checkpoint(epoch, False)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f'  Epoch time: {epoch_time/60:.1f} min | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            print()
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"Best model saved to: {os.path.join(self.config.CHECKPOINT_DIR, 'sprite_isnet_best.pth')}")
        
        if self.writer:
            self.writer.close()
        
        return self.training_history


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("SPRITE IS-NET TRAINING")
    print("="*70)
    
    # Load configuration
    config = SpriteConfig()
    
    # Create trainer
    trainer = SpriteTrainer(config)
    
    # Start training
    try:
        history = trainer.train()
        
        # Save training history
        import json
        history_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved to: {history_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving current state...")
        trainer.save_checkpoint(trainer.start_epoch, False)
        print("✓ State saved. You can resume training later.")
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
