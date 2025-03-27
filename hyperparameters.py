# hyperparameters.py
import os

# Base configuration
DATA_DIR = './Data'
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

# Model-specific hyperparameters
MODEL_CONFIGS = {
    '3d_cnn': {
        'num_classes': 2,
        'dropout_prob': 0.5,
        'pretrained': True,
    },
    '2d_cnn_lstm': {
        'num_classes': 2,
        'lstm_hidden_size': 512,
        'lstm_num_layers': 2,
        'dropout_prob': 0.5,
        'pretrained': True,
    },
    'transformer': {
        'num_classes': 2,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
    },
    'slowfast': {
        'num_classes': 2,
        'alpha': 8,
        'beta': 1/8,
        'dropout_prob': 0.5,
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
    }
}

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
    'slowfast': {
        'optimizer': 'sgd',
        'lr': 0.001
    },
    'two_stream': {
        'optimizer': 'adam',
        'lr': 0.0001
    }
}

def get_model_config(model_type, **overrides):
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    config = MODEL_CONFIGS[model_type].copy()
    for key, value in overrides.items():
        config[key] = value
    return config

def get_training_config(model_type, **overrides):
    if model_type not in TRAINING_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    config = TRAINING_CONFIGS[model_type].copy()
    for key, value in overrides.items():
        config[key] = value
    return config

def get_optimizer(model, model_type=None, lr=None, optimizer_name=None, **kwargs):
    import torch.optim as optim
    
    # Remap 'learning_rate' to 'lr' if provided in kwargs.
    if 'learning_rate' in kwargs:
        kwargs['lr'] = kwargs.pop('learning_rate')
    
    # Remove 'optimizer' from kwargs if present
    if 'optimizer' in kwargs:
        kwargs.pop('optimizer')
    
    # Determine which optimizer to use
    if optimizer_name is None and model_type is not None:
        training_config = get_training_config(model_type)
        optimizer_name = training_config['optimizer']
        default_lr = training_config['lr']
    else:
        optimizer_name = optimizer_name or 'adam'
        default_lr = OPTIMIZERS[optimizer_name]['lr']
    
    if optimizer_name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    config = OPTIMIZERS[optimizer_name].copy()
    
    if lr is not None:
        config['lr'] = lr
    elif model_type is not None and 'lr' in get_training_config(model_type):
        config['lr'] = get_training_config(model_type)['lr']
    
    for key, value in kwargs.items():
        config[key] = value
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), **config)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), **config)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), **config)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not implemented in get_optimizer")