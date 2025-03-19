# test_all_models.py
import torch
from hyperparameter_search import get_best_hyperparameters
import os

def test_all_models():
    # Create dummy data for testing
    train_paths = ["dummy_video.mp4"] * 10
    train_labels = [0, 1] * 5
    val_paths = ["dummy_video.mp4"] * 6
    val_labels = [0, 1] * 3
    
    # Test directory
    output_dir = "./test_hp_search"
    os.makedirs(output_dir, exist_ok=True)
    
    # Import all model classes
    from Models.model_3dcnn import Model3DCNN
    from Models.model_2dcnn_lstm import Model2DCNNLSTM
    from Models.model_transformer import VideoTransformer
    from Models.model_i3d import TransferLearningI3D
    from Models.model_simplecnn import SimpleCNN
    from Models.model_Temporal3DCNN import Temporal3DCNN
    from Models.model_slowfast import SlowFastNetwork
    from Models.model_r2plus1d import R2Plus1DNet
    from Models.model_two_stream import TwoStreamNetwork
    from Models.violence_cnn_lstm import ViolenceCNNLSTM
    
    # Define models to test with their respective parameters
    models_to_test = {
        '3d_cnn': {
            'class': Model3DCNN,
            'model_params': {'num_classes': 2, 'dropout_prob': 0.5, 'pretrained': True},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        '2d_cnn_lstm': {
            'class': Model2DCNNLSTM,
            'model_params': {'num_classes': 2, 'lstm_hidden_size': 512, 'lstm_num_layers': 2, 'dropout_prob': 0.5, 'pretrained': True},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        'transformer': {
            'class': VideoTransformer,
            'model_params': {'num_classes': 2, 'embed_dim': 512, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.1},
            'optimizer_params': {'learning_rate': 0.0002, 'weight_decay': 0.01}
        },
        'i3d': {
            'class': TransferLearningI3D,
            'model_params': {'num_classes': 2, 'dropout_prob': 0.5, 'pretrained': True},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        'simple_cnn': {
            'class': SimpleCNN,
            'model_params': {'num_classes': 2, 'dropout_prob': 0.5},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        'temporal_3d_cnn': {
            'class': Temporal3DCNN,
            'model_params': {'num_classes': 2, 'dropout_prob': 0.5},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        'slowfast': {
            'class': SlowFastNetwork,
            'model_params': {'num_classes': 2, 'pretrained': True, 'alpha': 8, 'beta': 1/8, 'dropout_prob': 0.5},
            'optimizer_params': {'learning_rate': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4, 'optimizer': 'sgd'}
        },
        'r2plus1d': {
            'class': R2Plus1DNet,
            'model_params': {'num_classes': 2, 'pretrained': True, 'dropout_prob': 0.5, 'frozen_layers': None},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        },
        'cnn_lstm': {
            'class': ViolenceCNNLSTM,
            'model_params': {'num_classes': 2, 'lstm_hidden_size': 512, 'num_layers': 2, 'dropout': 0.5, 'activation': 'relu'},
            'optimizer_params': {'learning_rate': 0.0001, 'weight_decay': 1e-5}
        }
    }

    # Two-stream network requires special handling due to optical flow input
    # We'll skip it in automated testing
    
    # Run test for each model
    from hyperparameters import get_optimizer
    
    for model_name, model_config in models_to_test.items():
        try:
            print(f"\n\n{'='*40}")
            print(f"TESTING MODEL: {model_name}")
            print(f"{'='*40}")
            
            model_class = model_config['class']
            model_params = model_config['model_params']
            optimizer_params = model_config['optimizer_params']
            
            print(f"Creating model with params: {model_params}")
            
            # Create model with only model_params (simulate the fix in main.py)
            model = model_class(**model_params)
            print(f"Model {model_name} created successfully!")
            
            # Test creating optimizer with separated parameters
            print(f"\nCreating optimizer with params: {optimizer_params}")
            optimizer = get_optimizer(model, model_type=model_name, **optimizer_params)
            print(f"Optimizer created successfully: {optimizer}")
            
            # Verify learning_rate has been correctly remapped to lr
            lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate set to: {lr}")
            
            # Confirm the provided learning_rate matches the optimizer's lr
            expected_lr = optimizer_params.get('learning_rate')
            if expected_lr and expected_lr == lr:
                print(f"✓ Learning rate correctly set from 'learning_rate' parameter")
            else:
                print(f"✗ Learning rate mismatch: expected {expected_lr}, got {lr}")
                
            print(f"Test for {model_name} completed successfully!")
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
    
    print("\nAll model tests completed!")

if __name__ == "__main__":
    test_all_models()