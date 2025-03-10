# models/model_i3d.py
import torch
import torch.nn as nn

class TransferLearningI3D(nn.Module):
    def __init__(self, num_classes=2, use_pose=False, pose_input_size=66, dropout_prob=0.5,pretrained=True):
        super(TransferLearningI3D, self).__init__()
        
        # Load pre-trained I3D model (using PyTorchVideo's Slow-Fast R50)
        try:
            self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            # Replace the final classification layer with identity for feature extraction
            self.feature_dim = 2048
            self.backbone.blocks[-1].proj = nn.Identity()
        except Exception as e:
            print(f"Warning: Error loading I3D model from PyTorchVideo: {e}")
            # Fallback to a simple 3D convnet
            from torchvision.models.video import r3d_18
            self.backbone = r3d_18(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
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
            
            # LSTM for temporal modeling of pose data
            self.pose_lstm = nn.LSTM(
                input_size=64,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            
            # Combined classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim + 128, 512),  # 128 = 64*2 (bidirectional)
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            # Video-only classifier
            self.classifier = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
    def forward(self, inputs):
        if self.use_pose:
            # Unpack inputs
            video_frames, pose_keypoints = inputs
            
            # Ensure video frames are in the right format [B, C, T, H, W]
            if video_frames.dim() == 5 and video_frames.shape[1] != 3:
                # Permute from [B, T, C, H, W] to [B, C, T, H, W]
                video_frames = video_frames.permute(0, 2, 1, 3, 4)
            
            # Process video frames
            video_features = self.backbone(video_frames)
            
            # Process pose data
            batch_size, seq_length, pose_dim = pose_keypoints.shape
            
            # Reshape for processing
            pose_features = self.pose_encoder(pose_keypoints.reshape(-1, pose_dim))
            pose_features = pose_features.reshape(batch_size, seq_length, -1)
            
            # Apply LSTM
            pose_features, _ = self.pose_lstm(pose_features)
            
            # Take the final time step
            pose_features = pose_features[:, -1, :]
            
            # Combine features
            combined_features = torch.cat([video_features, pose_features], dim=1)
            
            # Classification
            outputs = self.classifier(combined_features)
            
        else:
            # Process only video frames
            video_frames = inputs
            
            # Ensure video frames are in the right format [B, C, T, H, W]
            if video_frames.dim() == 5 and video_frames.shape[1] != 3:
                # Permute from [B, T, C, H, W] to [B, C, T, H, W]
                video_frames = video_frames.permute(0, 2, 1, 3, 4)
                
            video_features = self.backbone(video_frames)
            outputs = self.classifier(video_features)
        
        return outputs