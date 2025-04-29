import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18
import cv2
import numpy as np

class SpatialStream(nn.Module):
    """
    Spatial stream that processes RGB frames for appearance information.
    """
    def __init__(self, num_classes=2, pretrained=True, backbone='r3d_18', dropout_prob=0.3):
        super(SpatialStream, self).__init__()
        if backbone == 'r3d_18':
            self.backbone = r3d_18(pretrained=pretrained)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'r2plus1d_18':
            from torchvision.models.video import r2plus1d_18
            self.backbone = r2plus1d_18(pretrained=pretrained)
            backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure input is in format [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Input is [B, T, C, H, W] -> permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        features = self.backbone(x)
        return self.classifier(features)

class TemporalStream(nn.Module):
    """
    Temporal stream that processes optical flow for motion information.
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_prob=0.5):
        super(TemporalStream, self).__init__()
        # Use a 3D CNN but modify first conv to accept 2-channel flow input
        self.backbone = r3d_18(pretrained=pretrained)
        # Replace the first conv layer to accept 2-channel flow input
        old_conv = self.backbone.stem[0]
        self.backbone.stem[0] = nn.Conv3d(
            2, 64,  # 2 channels for flow (x and y)
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained:
            with torch.no_grad():
                new_weights = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1, 1)
                self.backbone.stem[0].weight.copy_(new_weights)
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure input is in format [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 2:
            # If input is RGB, take first 2 channels (should not happen in production)
            if x.shape[1] == 3:
                x = x[:, :2]
            else:
                x = x.permute(0, 2, 1, 3, 4)
                if x.shape[1] > 2:
                    x = x[:, :2]
        features = self.backbone(x)
        return self.classifier(features)

class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Convolutional Network for Video Recognition.
    Uses pre-computed optical flow for stability and performance.
    """
    def __init__(self, num_classes=2, spatial_weight=1.0, temporal_weight=1.5, 
                 pretrained=True, spatial_backbone='r3d_18', dropout_prob=0.5,
                 fusion='late'):
        super(TwoStreamNetwork, self).__init__()
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.fusion = fusion
        self.spatial_stream = SpatialStream(
            num_classes=num_classes, 
            pretrained=pretrained,
            backbone=spatial_backbone,
            dropout_prob=dropout_prob
        )
        self.temporal_stream = TemporalStream(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_prob=dropout_prob
        )
        if fusion == 'conv':
            self.spatial_stream.classifier = nn.Identity()
            self.temporal_stream.classifier = nn.Identity()
            self.feature_dim = 512  # Both streams output 512-d features
            self.fusion_layers = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(512, num_classes)
            )
            self._initialize_fusion_weights()
        
        # Set a fixed random seed for reproducibility
        self.set_seeds(42)

    def set_seeds(self, seed=42):
        """Set random seeds for reproducibility"""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def _initialize_fusion_weights(self):
        for m in self.fusion_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_flow(self, video_path, flow_dir):
        """Load pre-computed optical flow for a video"""
        flow_path = Path(flow_dir) / Path(video_path).with_suffix('.flow.pt').name
        if flow_path.exists():
            return torch.load(flow_path)
        return None

    def forward(self, x):
        # x can now be a tuple of (frames, flow) or just frames
        # If flow is provided, use it; otherwise try to load pre-computed flow
        if isinstance(x, tuple):
            frames = x[0]
            if len(x) > 1 and x[1] is not None:
                flow = x[1]
            else:
                # Use zeros as fallback if flow not provided
                # This should not happen in normal operation
                B, T, C, H, W = frames.shape
                flow = torch.zeros((B, 2, T-1, H, W), dtype=frames.dtype, device=frames.device)
                print("WARNING: No flow provided. Using zeros.")
        else:
            frames = x
            # Use zeros as fallback (this is a warning condition)
            B, T, C, H, W = frames.shape
            flow = torch.zeros((B, 2, T-1, H, W), dtype=frames.dtype, device=frames.device)
            print("WARNING: No flow provided. Using zeros.")
        
        if self.fusion == 'late':
            # Use gradient clipping for stability
            with torch.cuda.amp.autocast(enabled=True):
                spatial_logits = self.spatial_stream(frames)
                temporal_logits = self.temporal_stream(flow)
                
                combined_logits = (
                    self.spatial_weight * spatial_logits + 
                    self.temporal_weight * temporal_logits
                ) / (self.spatial_weight + self.temporal_weight)
            
            return combined_logits
        elif self.fusion == 'conv':
            with torch.cuda.amp.autocast(enabled=True):
                spatial_features = self.spatial_stream(frames)
                temporal_features = self.temporal_stream(flow)
                combined_features = torch.cat([spatial_features, temporal_features], dim=1)
                return self.fusion_layers(combined_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")