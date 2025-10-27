# test_rembg_integration.py - Test custom ONNX model with rembg
import os
import sys
import time
from pathlib import Path
import shutil

def check_rembg_installed():
    """Check if rembg is installed"""
    try:
        import rembg
        print(f"✓ rembg version: {rembg.__version__}")
        return True
    except ImportError:
        print("✗ rembg not installed")
        print("\nInstall with:")
        print("  pip install rembg[gpu,cli]  # For GPU support")
        print("  pip install rembg[cpu,cli]  # For CPU only")
        return False


def setup_custom_model(onnx_path, model_name='sprite_isnet'):
    """Copy ONNX model to rembg directory"""
    print("\n" + "="*60)
    print("SETTING UP CUSTOM MODEL")
    print("="*60)
    
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        print(f"✗ ONNX model not found: {onnx_path}")
        return None
    
    # rembg model directory
    rembg_dir = Path.home() / '.u2net'
    rembg_dir.mkdir(exist_ok=True)
    
    # Copy model
    dest_path = rembg_dir / f"{model_name}.onnx"
    print(f"\nCopying model...")
    print(f"  From: {onnx_path}")
    print(f"  To: {dest_path}")
    
    shutil.copy2(onnx_path, dest_path)
    
    # Verify
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model copied successfully ({size_mb:.2f} MB)")
        return dest_path
    else:
        print(f"  ✗ Failed to copy model")
        return None


def test_single_image(model_path, test_image, output_dir):
    """Test background removal on single image"""
    print("\n" + "="*60)
    print("TESTING SINGLE IMAGE")
    print("="*60)
    
    from rembg import remove, new_session
    from PIL import Image
    import numpy as np
    
    test_image = Path(test_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    print(f"\nInput: {test_image.name}")
    
    # Load image
    print("Loading image...")
    input_img = Image.open(test_image)
    print(f"  Size: {input_img.size}")
    print(f"  Mode: {input_img.mode}")
    
    # Create session with custom model
    print("\nCreating rembg session with custom model...")
    try:
        # Note: rembg doesn't directly support custom model paths in session
        # We need to use the u2net_custom model type
        session = new_session('u2net_custom')
        
        # Manually set model path (this is a workaround)
        session.inner_session.model_path = str(model_path)
        
        print("  ✓ Session created")
    except Exception as e:
        print(f"  ✗ Failed to create session: {e}")
        print("\n  Alternative: Use CLI instead")
        return False
    
    # Remove background
    print("\nRemoving background...")
    start_time = time.time()
    
    try:
        output_img = remove(input_img, session=session)
        elapsed = time.time() - start_time
        
        print(f"  ✓ Background removed in {elapsed:.2f} seconds")
        
        # Save output
        output_path = output_dir / f"{test_image.stem}_nobg.png"
        output_img.save(output_path)
        print(f"  ✓ Saved to: {output_path}")
        
        # Save comparison
        comparison_path = output_dir / f"{test_image.stem}_comparison.png"
        
        # Create side-by-side comparison
        width = input_img.width * 2
        height = input_img.height
        comparison = Image.new('RGB', (width, height), (128, 128, 128))
        comparison.paste(input_img.convert('RGB'), (0, 0))
        comparison.paste(output_img.convert('RGB'), (input_img.width, 0))
        comparison.save(comparison_path)
        
        print(f"  ✓ Comparison saved to: {comparison_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_mode(model_path, test_image, output_dir):
    """Test using rembg CLI (more reliable)"""
    print("\n" + "="*60)
    print("TESTING CLI MODE")
    print("="*60)
    
    import subprocess
    
    test_image = Path(test_image)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not test_image.exists():
        print(f"✗ Test image not found: {test_image}")
        return False
    
    output_path = output_dir / f"{test_image.stem}_cli_nobg.png"
    
    # Build rembg command
    model_name = model_path.stem
    cmd = [
        'rembg', 'i',
        '-m', 'u2net_custom',
        '-x', f'{{"model_path": "~/.u2net/{model_path.name}"}}',
        str(test_image),
        str(output_path)
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Input: {test_image}")
    print(f"Output: {output_path}")
    
    print("\nRunning rembg CLI...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✓ Background removed in {elapsed:.2f} seconds")
            print(f"  ✓ Saved to: {output_path}")
            
            if result.stdout:
                print(f"\nOutput: {result.stdout}")
            
            return True
        else:
            print(f"  ✗ Command failed with return code {result.returncode}")
            if result.stderr:
                print(f"\nError: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error running command: {e}")
        return False


def batch_test_cli(model_path, input_dir, output_dir, num_images=5):
    """Test batch processing with CLI"""
    print("\n" + "="*60)
    print("TESTING BATCH PROCESSING")
    print("="*60)
    
    import subprocess
    from tqdm import tqdm
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get test images
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(ext)))
    
    image_files = sorted(image_files)[:num_images]
    
    if len(image_files) == 0:
        print(f"✗ No images found in {input_dir}")
        return False
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    success_count = 0
    total_time = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        output_path = output_dir / f"{img_path.stem}_nobg.png"
        
        cmd = [
            'rembg', 'i',
            '-m', 'u2net_custom',
            '-x', f'{{"model_path": "~/.u2net/{model_path.name}"}}',
            str(img_path),
            str(output_path)
        ]
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            success_count += 1
            total_time += elapsed
    
    avg_time = total_time / len(image_files)
    
    print(f"\n" + "="*60)
    print(f"Batch processing complete:")
    print(f"  Successful: {success_count}/{len(image_files)}")
    print(f"  Average time: {avg_time:.2f} seconds/image")
    print(f"  Total time: {total_time:.2f} seconds")
    print("="*60)
    
    return success_count == len(image_files)


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("REMBG INTEGRATION TEST")
    print("="*70)
    
    # Configuration
    onnx_model = r'E:\Projects\sprite-isnet\onnx_models\sprite_isnet.onnx'
    test_image = r'E:\sprite-data\test\images\test1.png'
    test_dir = r'E:\sprite-data\test\images'
    output_dir = r'E:\sprite-data\test\output\rembg_test'
    
    # Check rembg installation
    if not check_rembg_installed():
        return
    
    # Setup custom model
    model_path = setup_custom_model(onnx_model)
    if not model_path:
        return
    
    # Test CLI mode (most reliable)
    print("\n" + "="*70)
    print("Starting tests...")
    print("="*70)
    
    # Test 1: Single image with CLI
    if os.path.exists(test_image):
        test_cli_mode(model_path, test_image, output_dir)
    else:
        print(f"\n⚠ Test image not found: {test_image}")
        print("  Skipping single image test")
    
    # Test 2: Batch processing
    if os.path.exists(test_dir):
        batch_test_cli(model_path, test_dir, output_dir, num_images=5)
    else:
        print(f"\n⚠ Test directory not found: {test_dir}")
        print("  Skipping batch test")
    
    # Final summary
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print(f"\nModel location: {model_path}")
    print(f"Output directory: {output_dir}")
    print("\nTo use your model:")
    print(f'  rembg i -m u2net_custom -x \'{{"model_path": "~/.u2net/{model_path.name}"}}\' input.png output.png')


if __name__ == '__main__':
    main()
