# Models/model_simplecnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # First, we need to permute the input to get [batch, channels, frames, height, width]
        
        self.features = nn.Sequential(
            # First convolution block
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Second convolution block
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Third convolution block
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Classifier - will determine size during forward pass
        self.classifier = None
        self.num_classes = num_classes
    
    def forward(self, x):
        # Input shape is [batch_size, frames, channels, height, width]
        # Need to permute to [batch_size, channels, frames, height, width]
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
            print(f"Permuted input shape: {x.shape}")
        
        # Pass through feature extractor
        x = self.features(x)
        print(f"After features shape: {x.shape}")
        
        # Flatten
        x = x.view(x.size(0), -1)
        print(f"Flattened shape: {x.shape}")
        
        # Initialize classifier on first forward pass if not done yet
        if self.classifier is None:
            feature_size = x.shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(feature_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes)
            )
            print(f"Created classifier with input size: {feature_size}")
        
        # Classification
        x = self.classifier(x)
        return x