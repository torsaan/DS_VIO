# models/model_i3d.py
import torch
import torch.nn as nn

class TransferLearningI3D(nn.Module):
    def __init__(self, num_classes=2, use_pose=False, pose_input_size=66, dropout_prob=0.5, pretrained=True):
        super(TransferLearningI3D, self).__init__()
        # use_pose parameter is kept for backward compatibility but ignored
        
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
        
        # Video-only classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, inputs):
        """
        Forward pass through the I3D model.
        
        Args:
            inputs: Input video frames tensor of shape [B, T, C, H, W] or [B, C, T, H, W]
            
        Returns:
            Classification output
        """
        # Process only video frames
        video_frames = inputs
        
        # Ensure video frames are in the right format [B, C, T, H, W]
        if video_frames.dim() == 5 and video_frames.shape[1] != 3:
            # Permute from [B, T, C, H, W] to [B, C, T, H, W]
            video_frames = video_frames.permute(0, 2, 1, 3, 4)
            
        # Process video frames
        video_features = self.backbone(video_frames)
        
        # Classification
        outputs = self.classifier(video_features)
        
        return outputs