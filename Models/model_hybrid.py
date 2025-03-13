# models/model_hybrid.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.video import r3d_18

class ModelHybrid(nn.Module):
    """
    Hybrid model that combines both 2D and 3D CNN features.
    Modified to use only video features (no pose data).
    """
    def __init__(self, num_classes=2, use_pose=False, pose_input_size=66):
        super(ModelHybrid, self).__init__()
        # use_pose parameter is kept for backward compatibility but ignored
        
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
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, inputs):
        """
        Forward pass through the hybrid model.
        
        Args:
            inputs: Input video frames tensor of shape [B, T, C, H, W] or [B, C, T, H, W]
            
        Returns:
            Classification output
        """
        # Process with video frames only
        frames = inputs
        
        # Process with 3D CNN for motion features
        # Ensure input is in format [B, C, T, H, W]
        if frames.dim() == 5 and frames.shape[1] != 3:
            # Input is [B, T, C, H, W], permute to [B, C, T, H, W]
            motion_input = frames.permute(0, 2, 1, 3, 4)
        else:
            motion_input = frames
            
        motion_features = self.cnn3d(motion_input)  # [B, feature_dim_3d]
        
        # Process with 2D CNN for appearance features
        batch_size, seq_length = frames.size(0), frames.size(1)
        
        # Take the middle frame for appearance features
        middle_idx = seq_length // 2
        middle_frame = frames[:, middle_idx, :, :, :]  # [B, C, H, W]
        
        appearance_features = self.cnn2d(middle_frame)  # [B, feature_dim_2d, 1, 1]
        appearance_features = appearance_features.squeeze(-1).squeeze(-1)  # [B, feature_dim_2d]
        
        # Combine motion and appearance features
        combined_features = torch.cat([motion_features, appearance_features], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(combined_features)
        
        # Classification
        outputs = self.classifier(fused_features)
        
        return outputs