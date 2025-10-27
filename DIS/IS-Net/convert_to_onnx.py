# convert_to_onnx.py - Convert trained IS-Net to ONNX format
import torch
import torch.onnx
import os
import sys
import argparse
from pathlib import Path

# Import IS-Net model
from models import ISNetDIS


def verify_onnx_model(onnx_path, test_input_size=1024):
    """Verify ONNX model is valid and working"""
    import onnx
    import onnxruntime as ort
    
    print("\n" + "="*60)
    print("VERIFYING ONNX MODEL")
    print("="*60)
    
    # Check model
    print("\n[1/3] Checking ONNX model structure...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model structure is valid")
    except Exception as e:
        print(f"  ✗ ONNX model check failed: {e}")
        return False
    
    # Get model info
    print("\n[2/3] Model information:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Producer: {onnx_model.producer_name} {onnx_model.producer_version}")
    print(f"  Opset version: {onnx_model.opset_import[0].version}")
    
    # Test inference
    print("\n[3/3] Testing ONNX inference...")
    try:
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # Get input/output info
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"  Input name: {input_name}")
        print(f"  Output name: {output_name}")
        
        # Create test input
        import numpy as np
        test_input = np.random.randn(1, 3, test_input_size, test_input_size).astype(np.float32)
        
        # Run inference
        import time
        start = time.time()
        ort_outputs = ort_session.run([output_name], {input_name: test_input})
        inference_time = (time.time() - start) * 1000
        
        print(f"  Output shape: {ort_outputs[0].shape}")
        print(f"  Inference time: {inference_time:.2f} ms")
        print("  ✓ ONNX inference successful!")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ONNX inference failed: {e}")
        return False


def convert_isnet_to_onnx(pytorch_model_path, onnx_output_path, 
                          input_size=1024, opset_version=14,
                          simplify=True):
    """
    Convert PyTorch IS-Net model to ONNX format for rembg
    
    Args:
        pytorch_model_path: Path to .pth model
        onnx_output_path: Path to save .onnx model
        input_size: Model input size
        opset_version: ONNX opset version (14 recommended for rembg)
        simplify: Whether to simplify ONNX model (requires onnx-simplifier)
    """
    print("\n" + "="*60)
    print("IS-NET TO ONNX CONVERSION")
    print("="*60)
    
    # Verify input file exists
    if not os.path.exists(pytorch_model_path):
        print(f"\n✗ Error: Model file not found: {pytorch_model_path}")
        return False
    
    print(f"\nInput model: {pytorch_model_path}")
    print(f"Output ONNX: {onnx_output_path}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Opset version: {opset_version}")
    
    # Load model
    print(f"\n[1/5] Loading PyTorch model...")
    device = torch.device('cpu')  # Convert on CPU for compatibility
    
    model = ISNetDIS()
    state_dict = torch.load(pytorch_model_path, map_location=device)
    
    # Handle different state dict formats
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            print("  Loading from checkpoint format")
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            print("  Loading from state_dict format")
            state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict)
    model.eval()
    print("  ✓ Model loaded successfully")
    
    # Create dummy input
    print(f"\n[2/5] Creating dummy input...")
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Test model forward pass
    print(f"\n[3/5] Testing model forward pass...")
    with torch.no_grad():
        test_output = model(dummy_input)
        if isinstance(test_output, (list, tuple)):
            print(f"  Model outputs {len(test_output)} tensors (using final output)")
            test_output = test_output[-1]
        print(f"  Output shape: {test_output.shape}")
    
    # Wrap model to return only final output
    class ISNetWrapper(torch.nn.Module):
        """Wrapper to ensure model returns single tensor"""
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            outputs = self.model(x)
            if isinstance(outputs, (list, tuple)):
                return outputs[-1]  # Return only final output
            return outputs
    
    wrapped_model = ISNetWrapper(model)
    wrapped_model.eval()
    
    # Export to ONNX
    print(f"\n[4/5] Exporting to ONNX...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(onnx_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            },
            verbose=False
        )
        print(f"  ✓ ONNX export successful")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        return False
    
    # Get file size
    size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.2f} MB")
    
    # Simplify ONNX model (optional but recommended)
    if simplify:
        print(f"\n[5/5] Simplifying ONNX model...")
        try:
            import onnxsim
            import onnx
            
            model_onnx = onnx.load(onnx_output_path)
            model_simplified, check = onnxsim.simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, onnx_output_path)
                new_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
                print(f"  ✓ Model simplified")
                print(f"  New size: {new_size_mb:.2f} MB (saved {size_mb - new_size_mb:.2f} MB)")
            else:
                print(f"  ⚠ Simplification check failed, keeping original")
        except ImportError:
            print(f"  ℹ onnx-simplifier not installed, skipping")
            print(f"    Install with: pip install onnx-simplifier")
        except Exception as e:
            print(f"  ⚠ Simplification failed: {e}")
            print(f"    Keeping original model")
    else:
        print(f"\n[5/5] Skipping simplification")
    
    # Verify ONNX model
    if verify_onnx_model(onnx_output_path, input_size):
        print("\n" + "="*60)
        print("✓ CONVERSION SUCCESSFUL!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("✗ CONVERSION COMPLETED BUT VERIFICATION FAILED")
        print("="*60)
        return False


def print_usage_instructions(onnx_path):
    """Print instructions for using ONNX model with rembg"""
    print("\n" + "="*60)
    print("HOW TO USE WITH REMBG")
    print("="*60)
    
    onnx_path = Path(onnx_path)
    rembg_path = Path.home() / '.u2net' / onnx_path.name
    
    print(f"\n1. Copy ONNX model to rembg directory:")
    print(f'   Copy-Item "{onnx_path}" "{rembg_path}"')
    
    print(f"\n2. Use with rembg CLI:")
    print(f'   rembg i -m u2net_custom -x \'{{"model_path": "~/.u2net/{onnx_path.name}"}}\' input.png output.png')
    
    print(f"\n3. Use with rembg Python API:")
    print(f"   from rembg import remove, new_session")
    print(f"   session = new_session('u2net_custom', providers=['CUDAExecutionProvider'])")
    print(f"   session.model_path = r'{rembg_path}'")
    print(f"   output = remove(input_image, session=session)")
    
    print(f"\n4. Batch processing with PowerShell:")
    print(f"   Get-ChildItem input\\*.png | ForEach-Object {{")
    print(f"       rembg i -m u2net_custom -x \'{{"model_path": "~/.u2net/{onnx_path.name}"}}\' $_.FullName output\\$($_.Name)")
    print(f"   }}")
    
    print("="*60)


def main():
    """Main conversion function with CLI"""
    parser = argparse.ArgumentParser(
        description='Convert IS-Net PyTorch model to ONNX format'
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to PyTorch model (.pth file)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path for output ONNX model (.onnx file)')
    parser.add_argument('--input-size', type=int, default=1024,
                       help='Model input size (default: 1024)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version (default: 14)')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Skip ONNX model simplification')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip ONNX model verification')
    
    args = parser.parse_args()
    
    # Convert
    success = convert_isnet_to_onnx(
        pytorch_model_path=args.model,
        onnx_output_path=args.output,
        input_size=args.input_size,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )
    
    if success:
        # Print usage instructions
        print_usage_instructions(args.output)
        sys.exit(0)
    else:
        print("\n✗ Conversion failed")
        sys.exit(1)


if __name__ == '__main__':
    # If run without arguments, use default paths
    import sys
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("CONVERT IS-NET TO ONNX")
        print("="*60)
        
        # Use default paths
        default_model = r'E:\Projects\sprite-isnet\saved_models\sprite-isnet\sprite_isnet_best.pth'
        default_output = r'E:\Projects\sprite-isnet\onnx_models\sprite_isnet.onnx'
        
        if not os.path.exists(default_model):
            print("\n✗ Default model not found!")
            print(f"   Expected: {default_model}")
            print("\nUsage:")
            print("  python convert_to_onnx.py --model <model.pth> --output <output.onnx>")
            print("\nOptions:")
            print("  --input-size SIZE    Model input size (default: 1024)")
            print("  --opset VERSION      ONNX opset version (default: 14)")
            print("  --no-simplify        Skip model simplification")
            print("  --no-verify          Skip model verification")
            sys.exit(1)
        
        print(f"\nUsing default paths:")
        print(f"  Model: {default_model}")
        print(f"  Output: {default_output}")
        print()
        
        success = convert_isnet_to_onnx(
            pytorch_model_path=default_model,
            onnx_output_path=default_output,
            input_size=1024,
            opset_version=14,
            simplify=True
        )
        
        if success:
            print_usage_instructions(default_output)
            sys.exit(0)
        else:
            sys.exit(1)
    
    main()
