# inference_sprite.py - Inference script for sprite background removal
import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

# Import IS-Net model
from models import ISNetDIS


class SpriteInference:
    """Inference class for sprite background removal"""
    
    def __init__(self, model_path, device='cuda', input_size=1024):
        """
        Args:
            model_path: Path to trained .pth model
            device: 'cuda' or 'cpu'
            input_size: Input image size (should match training)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = ISNetDIS().to(self.device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
        
        # Warm-up run
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, input_size, input_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        print("✓ Ready for inference")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Store original size
        orig_h, orig_w = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        scale = self.input_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), 
                            interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Normalize to [0, 1]
        normalized = padded.astype(np.float32) / 255.0
        
        # Convert to tensor (HWC -> CHW)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        
        return tensor, (orig_h, orig_w), (new_h, new_w), (top, left)
    
    def postprocess(self, mask, orig_size, resized_size, pad_offset):
        """Postprocess mask to original image size"""
        # Remove batch dimension
        mask = mask.squeeze().cpu().numpy()
        
        # Convert to uint8
        mask = (mask * 255).astype(np.uint8)
        
        # Remove padding
        orig_h, orig_w = orig_size
        new_h, new_w = resized_size
        top, left = pad_offset
        
        mask = mask[top:top+new_h, left:left+new_w]
        
        # Resize back to original size
        mask = cv2.resize(mask, (orig_w, orig_h), 
                         interpolation=cv2.INTER_LINEAR)
        
        return mask
    
    @torch.no_grad()
    def predict(self, image):
        """
        Predict mask for input image
        
        Args:
            image: numpy array (H, W, 3) in RGB format
        
        Returns:
            mask: numpy array (H, W) with values 0-255
        """
        # Preprocess
        tensor, orig_size, resized_size, pad_offset = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Inference
        outputs = self.model(tensor)
        
        # Get final output
        if isinstance(outputs, (list, tuple)):
            pred_mask = outputs[-1]
        else:
            pred_mask = outputs
        
        # Apply sigmoid
        pred_mask = torch.sigmoid(pred_mask)
        
        # Postprocess
        mask = self.postprocess(pred_mask, orig_size, resized_size, pad_offset)
        
        return mask
    
    def remove_background(self, image_path, output_path=None, 
                         return_rgba=True, save_mask=False):
        """
        Remove background from image and save result
        
        Args:
            image_path: Path to input image
            output_path: Path to save output (optional)
            return_rgba: If True, returns RGBA image with transparent bg
            save_mask: If True, also saves the mask separately
        
        Returns:
            output_image: Processed image (RGBA if return_rgba=True)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get mask
        mask = self.predict(image_rgb)
        
        if return_rgba:
            # Create RGBA image
            rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            rgba[:, :, 3] = mask
            output = rgba
        else:
            # Apply mask to image
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            output = (image * mask_3channel).astype(np.uint8)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if return_rgba:
                # Use PIL for RGBA PNG
                Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)).save(output_path)
            else:
                cv2.imwrite(str(output_path), output)
            
            # Save mask separately if requested
            if save_mask:
                mask_path = output_path.parent / f"{output_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)
        
        return output
    
    def batch_process(self, input_dir, output_dir, save_masks=False, 
                     file_suffix='_nobg'):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            save_masks: If True, also save mask images
            file_suffix: Suffix to add to output filenames
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_dir.glob(ext)))
        
        image_files = sorted(image_files)
        
        print(f"\nFound {len(image_files)} images to process")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        
        if len(image_files) == 0:
            print("No images found!")
            return
        
        success_count = 0
        error_count = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            output_path = output_dir / f"{img_path.stem}{file_suffix}.png"
            
            try:
                self.remove_background(
                    img_path, 
                    output_path, 
                    return_rgba=True,
                    save_mask=save_masks
                )
                success_count += 1
            except Exception as e:
                print(f"\n✗ Error processing {img_path.name}: {e}")
                error_count += 1
        
        print(f"\n" + "="*60)
        print(f"Batch processing complete!")
        print(f"  Successful: {success_count}/{len(image_files)}")
        print(f"  Failed: {error_count}/{len(image_files)}")
        print(f"  Output directory: {output_dir}")
        print("="*60)


def main():
    """Command-line interface for sprite inference"""
    parser = argparse.ArgumentParser(description='Sprite Background Removal Inference')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image or directory')
    parser.add_argument('--batch', action='store_true',
                       help='Batch process directory')
    parser.add_argument('--save-masks', action='store_true',
                       help='Save mask images separately')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--input-size', type=int, default=1024,
                       help='Model input size (default: 1024)')
    
    args = parser.parse_args()
    
    # Create inference object
    inferencer = SpriteInference(
        model_path=args.model,
        device=args.device,
        input_size=args.input_size
    )
    
    # Process
    if args.batch:
        inferencer.batch_process(
            input_dir=args.input,
            output_dir=args.output,
            save_masks=args.save_masks
        )
    else:
        result = inferencer.remove_background(
            image_path=args.input,
            output_path=args.output,
            save_mask=args.save_masks
        )
        print(f"\n✓ Output saved to: {args.output}")


if __name__ == '__main__':
    # If run without arguments, show example usage
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("SPRITE BACKGROUND REMOVAL INFERENCE")
        print("="*60)
        
        print("\nExample usage:")
        print("\n1. Single image:")
        print("   python inference_sprite.py \\")
        print("     --model E:\\Projects\\sprite-isnet\\saved_models\\sprite-isnet\\sprite_isnet_best.pth \\")
        print("     --input E:\\sprite-data\\test\\images\\test1.png \\")
        print("     --output E:\\sprite-data\\test\\output\\test1_nobg.png")
        
        print("\n2. Batch processing:")
        print("   python inference_sprite.py \\")
        print("     --model E:\\Projects\\sprite-isnet\\saved_models\\sprite-isnet\\sprite_isnet_best.pth \\")
        print("     --input E:\\sprite-data\\test\\images \\")
        print("     --output E:\\sprite-data\\test\\output \\")
        print("     --batch --save-masks")
        
        print("\n" + "="*60)
        sys.exit(0)
    
    main()
