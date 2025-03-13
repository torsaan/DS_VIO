import torch
import gc

def clear_cuda_memory(verbose=True):
    """
    Clears CUDA memory and garbage collector with detailed diagnostics.
    
    Args:
        verbose: Whether to print detailed information
    """
    # Run garbage collector first
    gc.collect()
    
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            if verbose:
                print(f"\nGPU: {torch.cuda.get_device_name(device)}")
                print(f"Memory allocated before clearing: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                print(f"Memory reserved before clearing: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            
            # Clear the cache
            torch.cuda.empty_cache()
            
            if verbose:
                print(f"Memory allocated after clearing: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                print(f"Memory reserved after clearing: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                print("CUDA memory cleared successfully.")
        except RuntimeError as e:
            print(f"Error while clearing CUDA memory: {e}")
    else:
        print("CUDA is not available. This could be due to:")
        print("  - NVIDIA drivers not properly installed")
        print("  - CUDA Toolkit not properly installed")
        print("  - PyTorch built without CUDA support")
        print("  - Environment variable issues (CUDA_VISIBLE_DEVICES)")
        
        # Try to provide more diagnostic information
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\nNVIDIA drivers are installed. Check PyTorch CUDA compatibility.")
            else:
                print("\nCould not run nvidia-smi. NVIDIA drivers might not be properly installed.")
        except:
            pass

if __name__ == "__main__":
    # Check PyTorch CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    
    # Call the clear function
    clear_cuda_memory()