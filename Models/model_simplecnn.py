# Models/model_simplecnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5, use_pose=False):
        super(SimpleCNN, self).__init__()
        
        # Ignore use_pose parameter (included for compatibility)
        self.use_pose = False
        
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
        
        # Global average pooling to reduce dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fixed classifier with standard dimensions
        self.classifier = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    
    def forward(self, x):
        # Input shape is [batch_size, frames, channels, height, width]
        # Need to permute to [batch_size, channels, frames, height, width]
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
            print(f"Permuted input shape: {x.shape}")
        
        # Pass through feature extractor
        x = self.features(x)
        print(f"After features shape: {x.shape}")
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        print(f"Flattened shape: {x.shape}")
        
        # Classification
        x = self.classifier(x)
        return x