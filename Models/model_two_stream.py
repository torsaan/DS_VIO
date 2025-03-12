# Models/model_two_stream.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import numpy as np

class SpatialStream(nn.Module):
    """
    Spatial stream that processes RGB frames for appearance information
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
        
        # Initialize the classifier
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
            # Input is [B, T, C, H, W], permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
            
        features = self.backbone(x)
        return self.classifier(features)

class TemporalStream(nn.Module):
    """
    Temporal stream that processes optical flow for motion information
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
        
        # If using pretrained, we need to adapt the weights of the first conv
        if pretrained:
            with torch.no_grad():
                # Average the weights across the RGB channels and replicate for 2 channels
                new_weights = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1, 1)
                self.backbone.stem[0].weight.copy_(new_weights)
        
        # Replace final classifier
        backbone_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the classifier
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Ensure correct format for flow data [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 2:
            # For test purposes, if we're just getting RGB data, adapt it
            # In production, this should be real optical flow data
            if x.shape[1] == 3:  # RGB data
                # Just use first 2 channels (arbitrary for testing)
                x = x[:, :2]
            else:  # Probably [B, T, C, H, W]
                x = x.permute(0, 2, 1, 3, 4)
                if x.shape[1] > 2:  # More than 2 channels
                    x = x[:, :2]  # Just use first 2 channels for testing
        
        features = self.backbone(x)
        return self.classifier(features)

class TwoStreamNetwork(nn.Module):
    """
    Two-Stream Convolutional Network for Video Recognition
    Based on the paper "Two-Stream Convolutional Networks for Action Recognition in Videos"
    and adapted with modern 3D CNN architectures.
    
    This model processes RGB frames through a spatial stream for appearance information
    and optical flow through a temporal stream for motion information.
    """
    def __init__(self, num_classes=2, spatial_weight=1.0, temporal_weight=1.5, 
                 pretrained=True, spatial_backbone='r3d_18', dropout_prob=0.5,
                 fusion='late'):
        """
        Initialize Two-Stream Network
        
        Args:
            num_classes: Number of output classes
            spatial_weight: Weight for spatial stream predictions
            temporal_weight: Weight for temporal stream predictions
            pretrained: Whether to use pretrained backbones
            spatial_backbone: Backbone for spatial stream ('r3d_18' or 'r2plus1d_18')
            dropout_prob: Dropout probability for classifiers
            fusion: Fusion method ('late' or 'conv')
        """
        super(TwoStreamNetwork, self).__init__()
        
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.fusion = fusion
        
        # Initialize the spatial and temporal streams
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
        
        # Optional convolutional fusion
        if fusion == 'conv':
            # Remove the classifiers
            self.spatial_stream.classifier = nn.Identity()
            self.temporal_stream.classifier = nn.Identity()
            
            # Get feature dimensions
            self.feature_dim = 512  # Both streams have 512 features
            
            # Create fusion layers
            self.fusion_layers = nn.Sequential(
                nn.Linear(self.feature_dim * 2, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob),
                nn.Linear(512, num_classes)
            )
            
            # Initialize fusion layers
            self._initialize_fusion_weights()
    
    def _initialize_fusion_weights(self):
        for m in self.fusion_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _generate_fake_flow(self, frames):
        """
        Generate fake optical flow for testing purposes
        In production, this would be replaced by real optical flow
        
        Args:
            frames: Tensor with shape [B, C, T, H, W] or [B, T, C, H, W]
            
        Returns:
            Fake flow tensor with shape [B, 2, T-1, H, W]
        """
        # Ensure frames are in the format [B, C, T, H, W]
        if frames.dim() == 5 and frames.shape[1] != 3:
            frames = frames.permute(0, 2, 1, 3, 4)
        
        # Extract batch size and dimensions
        batch_size, _, seq_length, height, width = frames.shape
        
        # For testing purposes, just create a 2-channel tensor from the first 2 channels of RGB
        # In production, this would be real optical flow calculated between consecutive frames
        fake_flow = frames[:, :2, :-1]  # Use first 2 channels and remove last time step
        
        return fake_flow
    
    def forward(self, x):
        """
        Forward pass through Two-Stream network
        
        Args:
            x: Input tensor or tuple
               - If tensor: RGB frames with shape [B, C, T, H, W] or [B, T, C, H, W]
               - If tuple: (RGB frames, Optical flow) or (RGB frames, _)
            
        Returns:
            Class logits
        """
        # Handle different input types
        if isinstance(x, tuple):
            frames = x[0]
            
            # Check if we're given precomputed flow
            if len(x) > 1 and x[1] is not None and isinstance(x[1], torch.Tensor) and x[1].dim() == 5:
                flow = x[1]
            else:
                # Generate fake flow for testing
                flow = self._generate_fake_flow(frames)
        else:
            frames = x
            # Generate fake flow for testing
            flow = self._generate_fake_flow(frames)
        
        # Late fusion
        if self.fusion == 'late':
            # Forward through spatial stream
            spatial_logits = self.spatial_stream(frames)
            
            # Forward through temporal stream
            temporal_logits = self.temporal_stream(flow)
            
            # Weighted average of predictions
            combined_logits = (
                self.spatial_weight * spatial_logits + 
                self.temporal_weight * temporal_logits
            ) / (self.spatial_weight + self.temporal_weight)
            
            return combined_logits
        
        # Convolutional fusion
        elif self.fusion == 'conv':
            # Get features from each stream
            spatial_features = self.spatial_stream(frames)
            temporal_features = self.temporal_stream(flow)
            
            # Concatenate features
            combined_features = torch.cat([spatial_features, temporal_features], dim=1)
            
            # Pass through fusion layers
            return self.fusion_layers(combined_features)
        
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion}")