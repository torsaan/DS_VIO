import torch

def clear_cuda_memory():
    """Clears CUDA memory and prints a confirmation message."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA memory cleared.")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    clear_cuda_memory() #example of how to call the function.