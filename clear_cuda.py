import torch
import gc

def get_gpu_memory_info(device=None):
    """
    Returns GPU memory information (total, allocated, reserved, free).

    Args:
        device (torch.device or int, optional): The GPU device to query. Defaults to the current device.

    Returns:
        tuple: (total_memory, allocated_memory, reserved_memory, free_memory) in MB.
    """
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.cuda.current_device()

    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    free_memory = total_memory - reserved_memory

    return total_memory, allocated_memory, reserved_memory, free_memory

def clear_cuda():
    """
    Clears CUDA memory and resets the CUDA context.
    """
    if torch.cuda.is_available():
        print("Before clearing:")
        memory_info = get_gpu_memory_info()
        if memory_info:
            total, allocated, reserved, free = memory_info
            print(f"  Total: {total:.2f} MB, Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Free: {free:.2f} MB")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        try:
            torch.cuda.ipc_collect()
        except:
            pass
        gc.collect()

        print("After clearing:")
        memory_info = get_gpu_memory_info()
        if memory_info:
            total, allocated, reserved, free = memory_info
            print(f"  Total: {total:.2f} MB, Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Free: {free:.2f} MB")
        print("CUDA memory cleared.")
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(device)}")
            tensor = torch.randn(2000, 2000).to(device)
            print(f"Allocated some memory on {device}")
            memory_info = get_gpu_memory_info()
            if memory_info:
                total, allocated, reserved, free = memory_info
                print(f"  Total: {total:.2f} MB, Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Free: {free:.2f} MB")

            del tensor
            print("Tensor deleted")
            clear_cuda()

        else:
            print("CUDA not available, skipping test")

    except Exception as e:
        print(f"An error occurred: {e}")
        clear_cuda()