# config_sprite.py - Configuration for sprite IS-Net training
import os
from pathlib import Path

class SpriteConfig:
    """Configuration for sprite sheet background removal training"""
    
    # ===== Paths (Windows-style) =====
    DATA_ROOT = r'E:\Projects\sprite-data'
    PROJECT_ROOT = r'E:\Projects\sprite-isnet'
    
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train', 'images')
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, 'train', 'masks')
    
    VALID_IMG_DIR = os.path.join(DATA_ROOT, 'valid', 'images')
    VALID_MASK_DIR = os.path.join(DATA_ROOT, 'valid', 'masks')
    
    TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test', 'images')
    TEST_MASK_DIR = os.path.join(DATA_ROOT, 'test', 'masks')
    
    # Model checkpoints
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'saved_models', 'sprite-isnet')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    # Pretrained model path
    PRETRAINED_MODEL = None
    
    # ===== Training Hyperparameters =====
    BATCH_SIZE = 4  # Reduce to 2 if GPU memory issues
    NUM_WORKERS = 4  # Number of data loading workers
    EPOCHS = 300
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    
    # Image settings
    INPUT_SIZE = 1024  # Match your typical sprite sheet size
    # Common sprite sheet sizes: 512, 1024, 2048
    
    # ===== Model Settings =====
    USE_PRETRAINED = False  # Transfer learning from general model
    FREEZE_ENCODER = False  # Set True to only train decoder (faster, less flexible)
    
    # ===== Optimization =====
    OPTIMIZER = 'AdamW'  # 'Adam' or 'AdamW'
    SCHEDULER = 'CosineAnnealing'  # 'StepLR', 'CosineAnnealing', or None
    
    # Learning rate scheduling
    LR_DECAY_EPOCHS = [100, 150]  # For StepLR
    LR_DECAY_RATE = 0.1
    
    # ===== Loss Function =====
    # IS-Net uses combination of BCE and IoU loss
    LOSS_TYPE = 'hybrid'  # 'bce', 'iou', or 'hybrid'
    BCE_WEIGHT = 0.7  # Weight for BCE loss in hybrid mode
    IOU_WEIGHT = 0.3  # Weight for IoU loss in hybrid mode
    
    # ===== Checkpointing =====
    SAVE_FREQ = 10  # Save checkpoint every N epochs
    VALIDATE_FREQ = 5  # Validate every N epochs
    
    # ===== Logging =====
    LOG_FREQ = 10  # Log every N batches
    USE_TENSORBOARD = True
    
    # ===== Device =====
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    
    # ===== Resume Training =====
    RESUME = False
    RESUME_CHECKPOINT = None  # Path to checkpoint to resume from
    
    # ===== Early Stopping =====
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 50  # Stop if no improvement for N epochs
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        Path(cls.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def verify_paths(cls):
        """Verify all required paths exist"""
        issues = []
        
        # Check data directories
        if not os.path.exists(cls.TRAIN_IMG_DIR):
            issues.append(f"Training images not found: {cls.TRAIN_IMG_DIR}")
        if not os.path.exists(cls.TRAIN_MASK_DIR):
            issues.append(f"Training masks not found: {cls.TRAIN_MASK_DIR}")
        if not os.path.exists(cls.VALID_IMG_DIR):
            issues.append(f"Validation images not found: {cls.VALID_IMG_DIR}")
        if not os.path.exists(cls.VALID_MASK_DIR):
            issues.append(f"Validation masks not found: {cls.VALID_MASK_DIR}")
        
        # Check pretrained model
        if cls.USE_PRETRAINED and not os.path.exists(cls.PRETRAINED_MODEL):
            issues.append(f"Pretrained model not found: {cls.PRETRAINED_MODEL}")
        
        return issues
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 70)
        print("SPRITE IS-NET TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Training images: {cls.TRAIN_IMG_DIR}")
        print(f"Training masks: {cls.TRAIN_MASK_DIR}")
        print(f"Validation images: {cls.VALID_IMG_DIR}")
        print(f"Validation masks: {cls.VALID_MASK_DIR}")
        print(f"\nBatch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Input size: {cls.INPUT_SIZE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Use pretrained: {cls.USE_PRETRAINED}")
        if cls.USE_PRETRAINED:
            print(f"Pretrained model: {cls.PRETRAINED_MODEL}")
        print(f"Freeze encoder: {cls.FREEZE_ENCODER}")
        print(f"\nCheckpoint directory: {cls.CHECKPOINT_DIR}")
        print(f"Log directory: {cls.LOG_DIR}")
        print("=" * 70)


if __name__ == '__main__':
    SpriteConfig.ensure_dirs()
    
    # Verify paths
    issues = SpriteConfig.verify_paths()
    if issues:
        print("\n⚠ CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Configuration is valid!")
    
    SpriteConfig.print_config()
