# Models/model_Temporal3DCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Temporal3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Temporal3DCNN, self).__init__()
        
        # Temporal feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )
        
        # Spatial feature extraction with more pooling to reduce dimensions
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        
        # Combined features with more aggressive pooling
        self.combined_conv = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # Pool in all dimensions
            # Add another block with pooling to further reduce dimensions
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),  # Pool again
        )
        
        # Add global average pooling to drastically reduce the feature dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Create classifier during initialization with a fixed size
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),  # Now we only have 256 input features thanks to global avg pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Print original input shape for debugging
        print(f"Original input shape: {x.shape}")
        
        # Input shape is [batch_size, frames, channels, height, width]
        # Need to permute to [batch_size, channels, frames, height, width]
        if x.dim() == 5 and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
            print(f"Permuted input shape: {x.shape}")
        
        # Pass through feature extractors
        x = self.temporal_conv(x)
        print(f"After temporal conv: {x.shape}")
        
        x = self.spatial_conv(x)
        print(f"After spatial conv: {x.shape}")
        
        x = self.combined_conv(x)
        print(f"After combined conv: {x.shape}")
        
        # Global average pooling to reduce dimensions
        x = self.global_avg_pool(x)
        print(f"After global avg pool: {x.shape}")
        
        # Flatten - will now be [batch_size, 256]
        x = x.view(x.size(0), -1)
        print(f"Flattened shape: {x.shape}")
        
        # Classification with the pre-defined classifier
        x = self.classifier(x)
        return x