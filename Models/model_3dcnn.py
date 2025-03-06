# models/model_3dcnn.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class Model3DCNN(nn.Module):
    def __init__(self, num_classes=2, use_pose=False, pose_input_size=66):
        super(Model3DCNN, self).__init__()
        # Load pre-trained 3D ResNet model
        self.backbone = r3d_18(pretrained=True)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with identity for feature extraction
        self.backbone.fc = nn.Identity()
        
        # Flag for using pose data
        self.use_pose = use_pose
        
        if use_pose:
            # Pose processing branch
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            
            # Temporal modeling for pose data
            self.pose_temporal = nn.GRU(
                input_size=64,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # Combined classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim + 128, 256),  # 128 = 64*2 (bidirectional)
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            # Video-only classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, inputs):
        if self.use_pose:
            # Unpack inputs
            video_frames, pose_keypoints = inputs
            
            # Process video frames through 3D CNN
            video_features = self.backbone(video_frames)  # [B, feature_dim]
            
            # Process pose data
            batch_size, seq_length, pose_dim = pose_keypoints.shape
            
            # Reshape for temporal processing
            pose_features = self.pose_encoder(pose_keypoints.view(-1, pose_dim))
            pose_features = pose_features.view(batch_size, seq_length, -1)
            
            # Apply GRU
            pose_features, _ = self.pose_temporal(pose_features)
            
            # Take the final time step
            pose_features = pose_features[:, -1, :]  # [B, 128]
            
            # Combine features and classify
            combined_features = torch.cat([video_features, pose_features], dim=1)
            outputs = self.classifier(combined_features)
        else:
            # Process only video frames
            video_frames = inputs
            video_features = self.backbone(video_frames)
            outputs = self.classifier(video_features)
        
        return outputs