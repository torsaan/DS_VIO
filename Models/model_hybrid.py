# models/model_hybrid.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.video import r3d_18

class ModelHybrid(nn.Module):
    """
    Hybrid model that combines both 2D and 3D CNN features with pose data.
    Designed specifically for violence detection with multiple input modalities.
    """
    def __init__(self, num_classes=2, use_pose=True, pose_input_size=66):
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
        
        # Pose processing is always enabled for this model
        self.use_pose = use_pose
        
        # Pose branch
        if self.use_pose:
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            
            # Temporal modeling for pose
            self.pose_temporal = nn.GRU(
                input_size=64,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
        
        # Feature fusion layers
        if self.use_pose:
            fusion_input_dim = self.feature_dim_3d + self.feature_dim_2d + 128  # 128 = 64*2 (bidirectional)
        else:
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
        if self.use_pose:
            # Unpack inputs: [video_frames, pose_keypoints]
            frames, pose = inputs
            
            # Process with 3D CNN for motion features
            motion_features = self.cnn3d(frames)  # [B, feature_dim_3d]
            
            # Process with 2D CNN for appearance features
            batch_size, seq_length = frames.size(0), frames.size(1)
            
            # Take the middle frame for appearance features
            middle_idx = seq_length // 2
            middle_frame = frames[:, :, middle_idx, :, :]  # [B, C, H, W]
            
            appearance_features = self.cnn2d(middle_frame)  # [B, feature_dim_2d, 1, 1]
            appearance_features = appearance_features.squeeze(-1).squeeze(-1)  # [B, feature_dim_2d]
            
            # Process pose data
            batch_size, seq_length, pose_dim = pose.shape
            
            # Reshape for processing
            pose_features = self.pose_encoder(pose.reshape(-1, pose_dim))
            pose_features = pose_features.reshape(batch_size, seq_length, -1)
            
            # Apply GRU for temporal modeling
            pose_features, _ = self.pose_temporal(pose_features)
            
            # Take the final time step
            pose_features = pose_features[:, -1, :]  # [B, 128]
            
            # Combine all features
            combined_features = torch.cat([motion_features, appearance_features, pose_features], dim=1)
        else:
            # Process with just video (without pose)
            frames = inputs
            
            # Process with 3D CNN for motion
            motion_features = self.cnn3d(frames)
            
            # Process with 2D CNN for appearance
            batch_size, seq_length = frames.size(0), frames.size(1)
            middle_idx = seq_length // 2
            middle_frame = frames[:, :, middle_idx, :, :]
            
            appearance_features = self.cnn2d(middle_frame)
            appearance_features = appearance_features.squeeze(-1).squeeze(-1)
            
            # Combine motion and appearance features
            combined_features = torch.cat([motion_features, appearance_features], dim=1)
        
        # Apply fusion layers
        fused_features = self.fusion(combined_features)
        
        # Classification
        outputs = self.classifier(fused_features)
        
        return outputs