# 03_prepare_data.py - Prepare and validate training data

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

class DataPreparation:
    """Helper class for preparing sprite training data"""
    
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.train_img = self.data_root / 'train' / 'images'
        self.train_mask = self.data_root / 'train' / 'masks'
        self.valid_img = self.data_root / 'valid' / 'images'
        self.valid_mask = self.data_root / 'valid' / 'masks'
    
    def validate_dataset(self):
        """Validate that images have corresponding masks"""
        print("\n" + "="*60)
        print("DATASET VALIDATION")
        print("="*60)
        
        issues = []
        
        # Check training data
        print("\n[Training Set]")
        train_images = sorted(list(self.train_img.glob('*.png')) + list(self.train_img.glob('*.jpg')))
        print(f"  Images found: {len(train_images)}")
        
        missing_masks = []
        for img_path in train_images:
            mask_path = self.train_mask / (img_path.stem + '.png')
            if not mask_path.exists():
                missing_masks.append(img_path.name)
        
        if missing_masks:
            print(f"  ✗ Missing masks for {len(missing_masks)} images:")
            for name in missing_masks[:5]:  # Show first 5
                print(f"    - {name}")
            if len(missing_masks) > 5:
                print(f"    ... and {len(missing_masks)-5} more")
            issues.append(f"{len(missing_masks)} training images missing masks")
        else:
            print(f"  ✓ All images have corresponding masks")
        
        # Check validation data
        print("\n[Validation Set]")
        valid_images = sorted(list(self.valid_img.glob('*.png')) + list(self.valid_img.glob('*.jpg')))
        print(f"  Images found: {len(valid_images)}")
        
        missing_masks = []
        for img_path in valid_images:
            mask_path = self.valid_mask / (img_path.stem + '.png')
            if not mask_path.exists():
                missing_masks.append(img_path.name)
        
        if missing_masks:
            print(f"  ✗ Missing masks for {len(missing_masks)} images:")
            for name in missing_masks[:5]:
                print(f"    - {name}")
            if len(missing_masks) > 5:
                print(f"    ... and {len(missing_masks)-5} more")
            issues.append(f"{len(missing_masks)} validation images missing masks")
        else:
            print(f"  ✓ All images have corresponding masks")
        
        # Recommendations
        print("\n[Recommendations]")
        total_samples = len(train_images) + len(valid_images)
        
        if total_samples < 50:
            print(f"  ⚠ Warning: Only {total_samples} total samples")
            print(f"    Recommended minimum: 100 samples")
            print(f"    Better results with: 500+ samples")
        elif total_samples < 200:
            print(f"  ⚠ Note: {total_samples} total samples")
            print(f"    This is the minimum for reasonable results")
            print(f"    Consider adding more for better performance")
        else:
            print(f"  ✓ Good: {total_samples} total samples")
        
        # Train/validation split
        if len(valid_images) > 0:
            val_ratio = len(valid_images) / (len(train_images) + len(valid_images))
            print(f"\n  Validation split: {val_ratio*100:.1f}%")
            if val_ratio < 0.1 or val_ratio > 0.3:
                print(f"    ⚠ Recommended: 10-20% validation split")
        
        print("\n" + "="*60)
        
        if issues:
            print("\n✗ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("\n✓ Dataset validation passed!")
            return True
    
    def visualize_samples(self, num_samples=5, output_dir=None):
        """Create visualization of image/mask pairs"""
        print("\n" + "="*60)
        print("CREATING SAMPLE VISUALIZATIONS")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.data_root / 'visualizations'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get training samples
        train_images = sorted(list(self.train_img.glob('*.png')) + list(self.train_img.glob('*.jpg')))
        
        if len(train_images) == 0:
            print("  ✗ No images found to visualize")
            return
        
        samples = train_images[:min(num_samples, len(train_images))]
        
        for idx, img_path in enumerate(samples):
            mask_path = self.train_mask / (img_path.stem + '.png')
            
            if not mask_path.exists():
                print(f"  Skipping {img_path.name} (no mask)")
                continue
            
            # Load image and mask
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                print(f"  ✗ Failed to load {img_path.name}")
                continue
            
            # Resize if too large
            max_size = 800
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
            
            # Create colored mask overlay
            mask_colored = np.zeros_like(img)
            mask_colored[:, :, 1] = mask  # Green channel for mask
            
            # Create overlay
            overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            
            # Create side-by-side comparison
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            comparison = np.hstack([img, mask_3ch, overlay])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'Mask', (img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'Overlay', (img.shape[1]*2 + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Save
            output_path = output_dir / f'sample_{idx+1:02d}_{img_path.stem}.png'
            cv2.imwrite(str(output_path), comparison)
            print(f"  ✓ Created: {output_path.name}")
        
        print(f"\n✓ Visualizations saved to: {output_dir}")
        print("="*60)
    
    def check_mask_quality(self):
        """Check mask quality (coverage, noise, etc.)"""
        print("\n" + "="*60)
        print("MASK QUALITY CHECK")
        print("="*60)
        
        train_masks = sorted(list(self.train_mask.glob('*.png')))
        
        if len(train_masks) == 0:
            print("  ✗ No masks found")
            return
        
        issues = []
        stats = {
            'too_sparse': [],  # Less than 5% coverage
            'too_dense': [],   # More than 95% coverage
            'good': []
        }
        
        print(f"\nChecking {len(train_masks)} masks...")
        
        for mask_path in tqdm(train_masks):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                issues.append(f"Failed to load: {mask_path.name}")
                continue
            
            # Calculate coverage (percentage of white pixels)
            coverage = (mask > 127).sum() / mask.size * 100
            
            if coverage < 5:
                stats['too_sparse'].append((mask_path.name, coverage))
            elif coverage > 95:
                stats['too_dense'].append((mask_path.name, coverage))
            else:
                stats['good'].append((mask_path.name, coverage))
        
        # Print results
        print(f"\n[Results]")
        print(f"  Good masks: {len(stats['good'])} ({len(stats['good'])/len(train_masks)*100:.1f}%)")
        print(f"  Too sparse (<5% coverage): {len(stats['too_sparse'])}")
        print(f"  Too dense (>95% coverage): {len(stats['too_dense'])}")
        
        if stats['too_sparse']:
            print(f"\n  ⚠ Masks with low coverage (might be mostly background):")
            for name, cov in stats['too_sparse'][:5]:
                print(f"    - {name}: {cov:.1f}% coverage")
        
        if stats['too_dense']:
            print(f"\n  ⚠ Masks with high coverage (might be inverted):")
            for name, cov in stats['too_dense'][:5]:
                print(f"    - {name}: {cov:.1f}% coverage")
        
        print("\n" + "="*60)
    
    def auto_split_train_valid(self, train_ratio=0.85):
        """Automatically split data into train/valid if validation is empty"""
        if len(list(self.valid_img.glob('*'))) > 0:
            print("\nValidation set already exists, skipping auto-split")
            return
        
        print("\n" + "="*60)
        print("AUTO-SPLITTING TRAIN/VALIDATION")
        print("="*60)
        
        all_images = sorted(list(self.train_img.glob('*.png')) + list(self.train_img.glob('*.jpg')))
        
        if len(all_images) < 10:
            print(f"  ⚠ Only {len(all_images)} images, skipping auto-split")
            return
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        valid_images = all_images[split_idx:]
        
        print(f"\n  Moving {len(valid_images)} images to validation set...")
        
        # Move validation images and masks
        for img_path in tqdm(valid_images):
            # Move image
            new_img_path = self.valid_img / img_path.name
            shutil.move(str(img_path), str(new_img_path))
            
            # Move corresponding mask
            mask_path = self.train_mask / (img_path.stem + '.png')
            if mask_path.exists():
                new_mask_path = self.valid_mask / mask_path.name
                shutil.move(str(mask_path), str(new_mask_path))
        
        print(f"\n✓ Split complete!")
        print(f"  Training: {len(train_images)} images")
        print(f"  Validation: {len(valid_images)} images")
        print("="*60)


def main():
    """Main data preparation workflow"""
    data_root = r'E:\Projects\sprite-data'
    
    prep = DataPreparation(data_root)
    
    print("\n" + "="*60)
    print("SPRITE DATA PREPARATION")
    print("="*60)
    
    # Auto-split if needed
    prep.auto_split_train_valid(train_ratio=0.85)
    
    # Validate dataset
    valid = prep.validate_dataset()
    
    if valid:
        # Check mask quality
        prep.check_mask_quality()
        
        # Create visualizations
        prep.visualize_samples(num_samples=10)
        
        print("\n" + "="*60)
        print("✓ DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nYour data is ready for training!")
    else:
        print("\n" + "="*60)
        print("✗ PLEASE FIX ISSUES BEFORE TRAINING")
        print("="*60)


if __name__ == '__main__':
    main()
