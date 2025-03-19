import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from train import train_epoch, validate
from tqdm import tqdm

# Define a dummy dataset for testing
def dummy_dataset(num_samples=10, batch_size=2):
    # For a 3d_cnn, assume each sample is [3, T, H, W]
    # Use small dimensions for testing (e.g., T=4, H=32, W=32)
    C, T, H, W = 3, 4, 32, 32
    data = torch.randn(num_samples, C, T, H, W)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create a simple dummy model (e.g., flatten and linear layer)
class DummyModel(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(DummyModel, self).__init__()
        # Compute number of features after flattening
        in_features = np.prod(input_shape)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)

def test_train_epoch():
    batch_size = 2
    dataloader = dummy_dataset(num_samples=10, batch_size=batch_size)
    # For each sample, input shape is [3, 4, 32, 32]
    model = DummyModel((3, 4, 32, 32))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)
    
    epoch_loss, epoch_acc = train_epoch(model, dataloader, optimizer, criterion, device)
    print(f"Train Epoch Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

def test_validate():
    batch_size = 2
    dataloader = dummy_dataset(num_samples=10, batch_size=batch_size)
    model = DummyModel((3, 4, 32, 32))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    model.to(device)
    
    metrics, all_preds, all_targets, all_probs = validate(model, dataloader, criterion, device)
    print("Validation Metrics:", metrics)
    print("Predictions:", all_preds)
    print("Targets:", all_targets)

if __name__ == '__main__':
    print("Testing train_epoch function:")
    test_train_epoch()
    print("\nTesting validate function:")
    test_validate()
