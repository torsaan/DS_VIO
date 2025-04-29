# models/model_3dcnn.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class Model3DCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5, pretrained=True):
        super(Model3DCNN, self).__init__()
        # Load pre-trained 3D ResNet model
        self.backbone = r3d_18(pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with identity for feature extraction
        self.backbone.fc = nn.Identity()
        
        # Video-only classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),  # Use parameter instead of hardcoded value
            nn.Linear(256, num_classes)
        )
    
    def forward(self, video_frames):
        # Process video frames through 3D CNN
        video_features = self.backbone(video_frames)  # [B, feature_dim]
        
        # Classification
        outputs = self.classifier(video_features)
        
        return outputs