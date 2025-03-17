# hyperparameters.py
import os

# Base configuration
DATA_DIR = './Data'
POSE_DIR = './Data/pose_keypoints'
BATCH_SIZE = 8
NUM_EPOCHS = 30
NUM_FRAMES = 16
FRAME_WIDTH = 224
FRAME_HEIGHT = 224

# Optimizer configurations
OPTIMIZERS = {
    'adam': {
        'lr': 0.0001,
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999)
    },
    'sgd': {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4
    },
    'adamw': {
        'lr': 0.0002,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999)
    }
}

# Model-specific hyperparameters - only include use_pose for models that support it
MODEL_CONFIGS = {
    '3d_cnn': {
        'num_classes': 2,
        'dropout_prob': 0.5,
        'use_pose': False,
        'pretrained': True,
    },
    '2d_cnn_lstm': {
        'num_classes': 2,
        'lstm_hidden_size': 512,
        'lstm_num_layers': 2,
        'dropout_prob': 0.5,
        'use_pose': False,
        'pretrained': True,
    },
    'transformer': {
        'num_classes': 2,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'use_pose': False,
    },
    'i3d': {
        'num_classes': 2,
        'dropout_prob': 0.5,
        'use_pose': False,
        'pretrained': True,
    },
    'slowfast': {
        'num_classes': 2,
        'alpha': 8,
        'beta': 1/8,
        'dropout_prob': 0.5,
        'pretrained': True,
    },
    'r2plus1d': {
        'num_classes': 2,
        'dropout_prob': 0.5,
        'frozen_layers': None,
        'pretrained': True,
    },
    'two_stream': {
        'num_classes': 2,
        'spatial_weight': 1.0,
        'temporal_weight': 1.5,
        'dropout_prob': 0.5,
        'fusion': 'late',
        'spatial_backbone': 'r3d_18',
        'pretrained': True,
    },
    'simple_cnn': {
        'num_classes': 2,
        'dropout_prob': 0.5,
        'use_pose': False
    },
    'temporal_3d_cnn': {
        'num_classes': 2
    },
    'cnn_lstm': {
        'num_classes': 2,
        'lstm_hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.5,
        'activation': 'relu'
    }
}

# And also update TRAINING_CONFIGS to include cnn_lstm
TRAINING_CONFIGS = {
    '3d_cnn': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    '2d_cnn_lstm': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'transformer': {
        'optimizer': 'adamw',
        'lr': 0.0002
    },
    'i3d': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'slowfast': {
        'optimizer': 'sgd',
        'lr': 0.001
    },
    'r2plus1d': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'two_stream': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'simple_cnn': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'temporal_3d_cnn': {
        'optimizer': 'adam',
        'lr': 0.0001
    },
    'cnn_lstm': {
        'optimizer': 'adam',
        'lr': 0.0001
    }
}

# Function to get model config with defaults
def get_model_config(model_type, **overrides):
    """
    Get configuration for a model with optional parameter overrides
    
    Args:
        model_type: Type of model
        **overrides: Keyword arguments to override defaults
        
    Returns:
        Dictionary with model configuration
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Start with default config
    config = MODEL_CONFIGS[model_type].copy()
    
    # Apply overrides
    for key, value in overrides.items():
        config[key] = value
    
    return config

# Function to get training configuration
def get_training_config(model_type, **overrides):
    """
    Get training configuration for a model
    
    Args:
        model_type: Type of model
        **overrides: Additional keyword arguments to override defaults
        
    Returns:
        Dictionary with training configuration
    """
    if model_type not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Start with default config
    config = TRAINING_CONFIGS[model_type].copy()
    
    # Apply overrides
    for key, value in overrides.items():
        config[key] = value
    
    return config

# Function to get optimizer with correct parameters
def get_optimizer(model, model_type=None, lr=None, optimizer_name=None, **kwargs):
    """
    Get optimizer instance for a model
    
    Args:
        model: PyTorch model
        model_type: Type of model (to use default optimizer)
        lr: Learning rate (overrides default)
        optimizer_name: Name of optimizer (overrides default)
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    import torch.optim as optim
    
    # Determine which optimizer to use
    if optimizer_name is None and model_type is not None:
        # Use default from training config
        training_config = get_training_config(model_type)
        optimizer_name = training_config['optimizer']
        default_lr = training_config['lr']
    else:
        # Default to adam if not specified
        optimizer_name = optimizer_name or 'adam'
        default_lr = OPTIMIZERS[optimizer_name]['lr']
    
    # Get base optimizer config
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    config = OPTIMIZERS[optimizer_name].copy()
    
    # Override learning rate if provided
    if lr is not None:
        config['lr'] = lr
    elif model_type is not None and 'lr' in get_training_config(model_type):
        config['lr'] = get_training_config(model_type)['lr']
    
    # Override with any additional parameters
    for key, value in kwargs.items():
        config[key] = value
    
    # Create optimizer
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), **config)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), **config)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), **config)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented in get_optimizer")