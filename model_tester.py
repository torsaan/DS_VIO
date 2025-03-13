# model_tester.py
import torch
import torch.nn as nn
import argparse
import os
import sys
import traceback
from utils.model_utils import create_fake_batch, print_tensor_shape, print_model_summary, get_model_params
from utils.logger import Logger

# Define which models support pose data
POSE_SUPPORTED_MODELS = ['3d_cnn', '2d_cnn_lstm', 'transformer', 'i3d']

def test_model(model_class, model_type, batch_size=2, num_frames=8, 
              height=224, width=224, use_pose=False, device="cuda"):
    """
    Test a model by creating fake input data and running a forward pass
    
    Args:
        model_class: Model class to instantiate
        model_type: Type of model ('3d_cnn', '2d_cnn_lstm', etc.)
        batch_size: Batch size for test data
        num_frames: Number of frames per video
        height: Frame height
        width: Frame width
        use_pose: Whether to include pose data
        device: Device to use for testing
        
    Returns:
        True if test passed, False otherwise
    """
    logger = Logger(f"test_{model_type}", log_dir="./logs", level=20)
    logger.info(f"Testing {model_type} model with use_pose={use_pose}")
    
    try:
        # Create model with appropriate parameters based on model type
        if model_type in POSE_SUPPORTED_MODELS and use_pose:
            model = model_class(num_classes=2, use_pose=use_pose)
        else:
            # For models that don't support pose, don't pass the use_pose parameter
            model = model_class(num_classes=2)
        
        model.to(device)
        
        # Print model summary
        print_model_summary(model)
        logger.info(f"Model has {get_model_params(model)['total']:,} parameters")
        
        # Create fake batch
        inputs, labels = create_fake_batch(
            batch_size=batch_size, 
            num_frames=num_frames,
            height=height,
            width=width,
            model_type=model_type,
            use_pose=use_pose if model_type in POSE_SUPPORTED_MODELS else False
        )
        
        # Move inputs and labels to device
        if isinstance(inputs, tuple):
            inputs = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs)
        else:
            inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Print input shapes
        logger.info("Input shapes:")
        print_tensor_shape(inputs, "inputs")
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        
        # Print output shape
        logger.info(f"Output shape: {outputs.shape}")
        
        # Check output dimensions
        if outputs.size(0) != batch_size:
            logger.error(f"Expected batch size {batch_size}, got {outputs.size(0)}")
            return False
        
        if outputs.size(1) != 2:  # Binary classification
            logger.error(f"Expected 2 output classes, got {outputs.size(1)}")
            return False
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        logger.info(f"Loss: {loss.item():.4f}")
        
        # Test backward pass
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        logger.info("Backward pass succeeded")
        
        logger.info(f"Model test PASSED: {model_type} with use_pose={use_pose}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        logger.error(traceback.format_exc())
        return False

def test_all_models(model_types=None, use_pose=False, device="cuda"):
    """
    Test multiple model types
    
    Args:
        model_types: List of model types to test (None for all)
        use_pose: Whether to include pose data
        device: Device to use for testing
        
    Returns:
        Dictionary mapping model types to test results
    """
    # Import model classes
    from Models.model_3dcnn import Model3DCNN
    from Models.model_2dcnn_lstm import Model2DCNNLSTM
    from Models.model_transformer import VideoTransformer
    from Models.model_i3d import TransferLearningI3D
    from Models.model_slowfast import SlowFastNetwork
    from Models.model_r2plus1d import R2Plus1DNet
    from Models.model_two_stream import TwoStreamNetwork
    from Models.model_simplecnn import SimpleCNN
    from Models.model_Temporal3DCNN import Temporal3DCNN
    
    # Define all available models
    all_models = {
        '3d_cnn': Model3DCNN,
        '2d_cnn_lstm': Model2DCNNLSTM,
        'transformer': VideoTransformer,
        'i3d': TransferLearningI3D,
        'slowfast': SlowFastNetwork,
        'r2plus1d': R2Plus1DNet,
        'two_stream': TwoStreamNetwork,
        'simple_cnn': SimpleCNN,
        'temporal_3d_cnn': Temporal3DCNN,
    }
    
    # Filter models if model_types is provided
    if model_types is not None:
        models_to_test = {k: v for k, v in all_models.items() if k in model_types}
    else:
        models_to_test = all_models
    
    results = {}
    
    # Create device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test each model
    for model_type, model_class in models_to_test.items():
        print(f"\n{'='*40}")
        print(f"Testing {model_type}")
        print(f"{'='*40}")
        
        # Test without pose
        without_pose = test_model(model_class, model_type, use_pose=False, device=device)
        
        # Test with pose if requested and model supports it
        with_pose = None
        if use_pose and model_type in POSE_SUPPORTED_MODELS:
            with_pose = test_model(model_class, model_type, use_pose=True, device=device)
        
        results[model_type] = {
            'without_pose': without_pose,
            'with_pose': with_pose
        }
    
    # Print summary
    print("\n" + "="*40)
    print("MODEL TEST SUMMARY")
    print("="*40)
    print(f"{'Model Type':<20} {'Without Pose':<15} {'With Pose':<15}")
    print("-"*50)
    
    for model_type, result in results.items():
        without_pose = "PASS" if result['without_pose'] else "FAIL"
        with_pose = "PASS" if result['with_pose'] else "N/A" if result['with_pose'] is None else "FAIL"
        print(f"{model_type:<20} {without_pose:<15} {with_pose:<15}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models for violence detection")
    parser.add_argument("--model_types", nargs="+", 
                      default=None,
                      help="Model types to test (omit for all)")
    parser.add_argument("--use_pose", action="store_true", 
                      help="Test models with pose data")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Run tests
    results = test_all_models(
        model_types=args.model_types,
        use_pose=args.use_pose,
        device=args.device
    )
    
    # Exit with error code if any tests failed
    any_failed = any(not result['without_pose'] for result in results.values())
    if args.use_pose:
        any_failed |= any(not result['with_pose'] for result in results.values() 
                         if result['with_pose'] is not None)
    
    sys.exit(1 if any_failed else 0)