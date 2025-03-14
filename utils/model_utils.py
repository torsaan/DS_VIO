# utils/model_utils.py
import torch
import torch.nn as nn
import numpy as np

def format_input_tensor(tensor, model_type):
    """
    Ensure input tensor is in the correct format for different model types
    
    Args:
        tensor: Input tensor of shape [B, T, C, H, W]
        model_type: The type of model ('3d_cnn', '2d_cnn_lstm', etc.)
        
    Returns:
        Correctly formatted tensor(s) for the given model type
    """
    # Check input dimensions (we expect [B, T, C, H, W] for most models from dataloader)
    if tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor, got {tensor.dim()}D")
    
    # Defaults to returning the tensor as is
    formatted_tensor = tensor
    
    # For 3D CNNs, the input should be [B, C, T, H, W]
    if model_type in ['3d_cnn', 'i3d', 'slowfast', 'r2plus1d', 'two_stream']:
        # Permute from [B, T, C, H, W] to [B, C, T, H, W]
        formatted_tensor = tensor.permute(0, 2, 1, 3, 4)
    
    # For two-stream networks, also generate a flow tensor
    if model_type == 'two_stream':
        # For testing, create a fake flow tensor using the first 2 channels
        flow_tensor = formatted_tensor[:, :2, :-1]  # [B, 2, T-1, H, W]
        return (formatted_tensor, flow_tensor)
    
    # Return the tensor
    return formatted_tensor

def create_fake_batch(batch_size=2, num_frames=16, height=224, width=224, model_type='3d_cnn'):
    """
    Create a fake batch of data for testing models
    
    Args:
        batch_size: Number of samples in the batch
        num_frames: Number of frames per video
        height: Frame height
        width: Frame width
        model_type: Type of model for which to format the data
        
    Returns:
        Tuple of (inputs, labels) suitable for the specified model
    """
    # Create random frames tensor: [B, T, C, H, W]
    frames = torch.randn(batch_size, num_frames, 3, height, width)
    
    # Create random labels
    labels = torch.randint(0, 2, (batch_size,))
    
    # Format tensor for the model type
    inputs = format_input_tensor(frames, model_type)
    
    return inputs, labels

def print_tensor_shape(tensor, name="Tensor"):
    """
    Print the shape of a tensor or list/tuple of tensors for debugging
    
    Args:
        tensor: Tensor or collection of tensors
        name: Name to use in the output
    """
    if isinstance(tensor, (list, tuple)):
        print(f"{name} is a {type(tensor).__name__} of length {len(tensor)}")
        for i, t in enumerate(tensor):
            if isinstance(t, torch.Tensor):
                print(f"  {name}[{i}] shape: {t.shape}")
            else:
                print(f"  {name}[{i}] type: {type(t).__name__}")
    elif isinstance(tensor, torch.Tensor):
        print(f"{name} shape: {tensor.shape}")
    else:
        print(f"{name} type: {type(tensor).__name__}")

def get_model_input_shape(model_type, batch_size=1, num_frames=16, height=224, width=224):
    """
    Get the expected input shape for a model type
    
    Args:
        model_type: Type of model
        batch_size: Batch size
        num_frames: Number of frames per video
        height: Frame height
        width: Frame width
        
    Returns:
        String describing the expected input shape
    """
    if model_type in ['3d_cnn', 'i3d', 'slowfast', 'r2plus1d']:
        return f"[{batch_size}, 3, {num_frames}, {height}, {width}] (BCTHW)"
    elif model_type in ['2d_cnn_lstm', 'transformer', 'cnn_lstm']:
        return f"[{batch_size}, {num_frames}, 3, {height}, {width}] (BTCHW)"
    elif model_type == 'two_stream':
        rgb = f"[{batch_size}, 3, {num_frames}, {height}, {width}] (BCTHW)"
        flow = f"[{batch_size}, 2, {num_frames-1}, {height}, {width}] (BCTHW)"
        return f"RGB: {rgb}, Flow: {flow}"
    else:
        return f"[{batch_size}, {num_frames}, 3, {height}, {width}] (BTCHW) (default)"

def get_model_params(model):
    """
    Get the number of parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }

def print_model_summary(model, input_shape=None):
    """
    Print a summary of the model architecture and parameters
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape to print
    """
    print(f"\n{'=' * 40}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'=' * 40}")
    
    if input_shape:
        print(f"Input shape: {input_shape}")
    
    params = get_model_params(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,} ({params['trainable']/params['total']*100:.1f}%)")
    print(f"Frozen parameters: {params['frozen']:,} ({params['frozen']/params['total']*100:.1f}%)")
    
    print(f"{'=' * 40}")