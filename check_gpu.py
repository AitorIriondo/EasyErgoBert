"""Check GPU/CUDA availability."""
import sys

print("Python:", sys.version)
print("=" * 50)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")

        # Test tensor on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"\nGPU tensor test: PASSED")
        except Exception as e:
            print(f"\nGPU tensor test: FAILED - {e}")
    else:
        print("\nCUDA not available. Checking why...")
        try:
            print(f"  torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")
        except:
            pass

except ImportError as e:
    print(f"PyTorch not installed: {e}")
except Exception as e:
    print(f"Error: {e}")
