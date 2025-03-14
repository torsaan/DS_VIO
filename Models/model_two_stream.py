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
    def __init__(self, num_classes=2, pretrained=True, backbone='r3d_18', dropout_prob=0.5):
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
    Processes RGB frames through a spatial stream and computes optical flow for the temporal stream.
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

    def _initialize_fusion_weights(self):
        for m in self.fusion_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _compute_optical_flow(self, frames):
        """
        Compute optical flow using Farneback's algorithm as a fallback.
        Expects frames tensor of shape [B, T, C, H, W] in range [0, 1].
        Returns a tensor of shape [B, 2, T-1, H, W] with flow (x, y) for each consecutive frame pair.
        """
        import cv2
        import numpy as np

        B, T, C, H, W = frames.shape
        flow_batch = []
        frames_np = frames.cpu().numpy()  # Convert to numpy array

        for b in range(B):
            sample = frames_np[b]  # shape [T, C, H, W]
            sample_flow = []
            gray_frames = []
            # Convert each frame to grayscale
            for t in range(T):
                frame = sample[t].transpose(1, 2, 0)  # [H, W, C]
                # Convert to uint8
                frame_uint8 = (frame * 255).astype(np.uint8)
                frame_gray = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2GRAY)
                gray_frames.append(frame_gray)
            # Compute optical flow between consecutive frames using Farneback's algorithm
            for t in range(T - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    gray_frames[t], gray_frames[t + 1], None,
                    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                    poly_n=5, poly_sigma=1.2, flags=0
                )
                sample_flow.append(flow)  # flow shape: [H, W, 2]
            sample_flow = np.stack(sample_flow, axis=0)  # [T-1, H, W, 2]
            sample_flow = sample_flow.transpose(3, 0, 1, 2)  # [2, T-1, H, W]
            flow_batch.append(sample_flow)

        flow_batch = np.stack(flow_batch, axis=0)  # [B, 2, T-1, H, W]
        return torch.tensor(flow_batch, dtype=frames.dtype, device=frames.device)

    def forward(self, x):
        # x can be a tensor or tuple. If tuple and optical flow is provided, use it.
        if isinstance(x, tuple):
            frames = x[0]
            if len(x) > 1 and x[1] is not None:
                flow = x[1]
            else:
                flow = self._compute_optical_flow(frames)
        else:
            frames = x
            flow = self._compute_optical_flow(frames)
        if self.fusion == 'late':
            spatial_logits = self.spatial_stream(frames)
            temporal_logits = self.temporal_stream(flow)
            combined_logits = (
                self.spatial_weight * spatial_logits + 
                self.temporal_weight * temporal_logits
            ) / (self.spatial_weight + self.temporal_weight)
            return combined_logits
        elif self.fusion == 'conv':
            spatial_features = self.spatial_stream(frames)
            temporal_features = self.temporal_stream(flow)
            combined_features = torch.cat([spatial_features, temporal_features], dim=1)
            return self.fusion_layers(combined_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")
