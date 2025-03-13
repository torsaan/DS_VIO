# models/model_3dcnn.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class Model3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5, pretrained=True, use_pose=False):
        super(Model3DCNN, self).__init__()
        # use_pose parameter is kept for backward compatibility but ignored
        
        # Load pre-trained 3D ResNet model
        self.backbone = r3d_18(pretrained=True)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with identity for feature extraction
        self.backbone.fc = nn.Identity()
        
        # Video-only classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the 3D CNN.
        
        Args:
            x: Input video frames tensor of shape [B, C, T, H, W] or [B, T, C, H, W]
            
        Returns:
            Classification output
        """
        # Ensure input is in format [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Input is [B, T, C, H, W], permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        # Process video frames through 3D CNN
        video_features = self.backbone(x)  # [B, feature_dim]
        
        # Classification
        outputs = self.classifier(video_features)
        
        return outputs