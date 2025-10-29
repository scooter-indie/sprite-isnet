# sprite_dataset.py - Custom dataset for sprite sheets
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms

class SpriteDataset(Dataset):
    """
    Dataset class for sprite sheet background removal.
    Handles sprite sheets with 4-128 sprites on non-transparent backgrounds.
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, target_size=1024):
        """
        Args:
            image_dir (str): Directory with sprite sheet images
            mask_dir (str): Directory with corresponding binary masks
            transform: Optional transforms
            target_size (int): Size to resize images (default 1024)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'))])
        
        # Verify corresponding masks exist
        valid_images = []
        for img_name in self.images:
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_name)
            if os.path.exists(mask_path):
                valid_images.append(img_name)
        
        self.images = valid_images
        print(f"Loaded {len(self.images)} images with masks from {image_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Try mask with same name but .png extension
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Read image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Pad to target size
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, 
                                  cv2.BORDER_CONSTANT, value=0)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize mask to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return image, mask, img_name


def create_sprite_dataloaders(train_img_dir, train_mask_dir,
                              valid_img_dir, valid_mask_dir,
                              batch_size=4, num_workers=4, 
                              target_size=1024):
    """
    Create dataloaders for training and validation.
    
    Returns:
        train_loader, valid_loader
    """
    train_dataset = SpriteDataset(
        train_img_dir, train_mask_dir, 
        target_size=target_size
    )
    
    valid_dataset = SpriteDataset(
        valid_img_dir, valid_mask_dir,
        target_size=target_size
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    return train_loader, valid_loader


# Test function
def test_dataset():
    """Test dataset loading"""
    dataset = SpriteDataset(
        r'E:\Projects\sprite-data\train\images',
        r'E:\Projects\sprite-data\train\masks',
        target_size=1024
    )

    
    if len(dataset) > 0:
        img, mask, name = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
        print(f"File name: {name}")
        return True
    return False


if __name__ == '__main__':
    test_dataset()
