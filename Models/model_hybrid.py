# models/model_hybrid.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.video import r3d_18

class ModelHybrid(nn.Module):
    """
    Hybrid model that combines both 2D and 3D CNN features.
    Designed specifically for violence detection with multiple visual modalities.
    """
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super(ModelHybrid, self).__init__()
        
        # 3D CNN branch (for motion features)
        self.cnn3d = r3d_18(pretrained=True)
        self.feature_dim_3d = self.cnn3d.fc.in_features
        self.cnn3d.fc = nn.Identity()
        
        # 2D CNN branch (for appearance features)
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.cnn2d = nn.Sequential(*modules)
        self.feature_dim_2d = 2048
        
        # Feature fusion layers
        fusion_input_dim = self.feature_dim_3d + self.feature_dim_2d
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob * 0.6)  # Reduced dropout in deeper layers
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, frames):
        # Process with 3D CNN for motion features
        motion_features = self.cnn3d(frames)  # [B, feature_dim_3d]
        
        # Process with 2D CNN for appearance features
        batch_size, seq_length = frames.size(0), frames.size(1)
        
        # Take the middle frame for appearance features
        middle_idx = seq_length // 2
        middle_frame = frames[:, :, middle_idx, :, :]  # [B, C, H, W]
        
        appearance_features = self.cnn2d(middle_frame)  # [B, feature_dim_2d, 1, 1]
        appearance_features = appearance_features.squeeze(-1).squeeze(-1)  # [B, feature_dim_2d]
        
        # Combine motion and appearance features
        combined_features = torch.cat([motion_features, appearance_features], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(combined_features)
        
        # Classification
        outputs = self.classifier(fused_features)
        
        return outputs