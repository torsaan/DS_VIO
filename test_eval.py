import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Define a simple dummy model for evaluation.
# For example, this model flattens the input and applies a linear layer.
class Dummy3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Dummy3DCNN, self).__init__()
        # For our dummy test, assume each input is [3, 16, 224, 224]
        self.flatten = nn.Flatten()
        in_features = 3 * 16 * 224 * 224
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # x is expected to be [B, 3, 16, 224, 224]
        x = self.flatten(x)
        return self.fc(x)

# Create a dummy dataset for evaluation.
def create_dummy_eval_loader(batch_size=2, num_samples=10):
    # Create random inputs of shape [num_samples, 3, 16, 224, 224]
    inputs = torch.randn(num_samples, 3, 16, 224, 224)
    # Random binary labels (0 or 1)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

# Import evaluation functions from your evaluations.py file.
from evaluations import evaluate_and_compare_models, ensemble_predictions

def test_evaluation():
    device = torch.device("cpu")
    # Create one dummy model.
    model = Dummy3DCNN(num_classes=2)
    models_dict = {"dummy_3dcnn": model}
    
    # Create a dummy test loader.
    test_loader = create_dummy_eval_loader()
    test_loaders = {"dummy_3dcnn": test_loader}
    
    # Call the evaluation function.
    results = evaluate_and_compare_models(models_dict, test_loaders, device, output_dir="./dummy_eval")
    print("Evaluation results:", results)

def test_ensemble():
    device = torch.device("cpu")
    # Create two dummy models.
    model1 = Dummy3DCNN(num_classes=2)
    model2 = Dummy3DCNN(num_classes=2)
    models_dict = {"dummy1": model1, "dummy2": model2}
    
    # Use the same dummy test loader for both models.
    test_loader = create_dummy_eval_loader()
    test_loaders = {"dummy1": test_loader, "dummy2": test_loader}
    
    ensemble_results = ensemble_predictions(models_dict, test_loaders, device, output_dir="./dummy_eval")
    print("Ensemble evaluation results:", ensemble_results)

if __name__ == "__main__":
    print("Testing evaluation function:")
    test_evaluation()
    print("\nTesting ensemble evaluation:")
    test_ensemble()
