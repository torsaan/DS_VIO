import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on important frames"""
    def __init__(self, feature_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, features):
        # features shape: [batch_size, num_frames, feature_dim]
        attention_weights = self.attention(features)  # [batch_size, num_frames, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize over frames
        
        # Apply attention weights
        weighted_features = features * attention_weights
        
        # Aggregate features
        context_vector = torch.sum(weighted_features, dim=1)  # [batch_size, feature_dim]
        
        return context_vector, attention_weights


class R2Plus1DNet(nn.Module):
    """
    R(2+1)D Model for violence detection.
    Based on the paper: "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_prob=0.5, frozen_layers=None, use_attention=True):
        super(R2Plus1DNet, self).__init__()
        
        # Load pre-trained model
        self.backbone = r2plus1d_18(pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with identity
        self.backbone.fc = nn.Identity()
        
        # Freeze specified layers if needed
        if frozen_layers:
            self._freeze_layers(frozen_layers)
        
        # Whether to use temporal attention
        self.use_attention = use_attention
        
        # Extract per-frame features if using attention
        if self.use_attention:
            # Save the layers before the final pooling
            self.early_backbone = nn.Sequential(
                self.backbone.stem,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            )
            
            # Per-frame feature extraction
            self.frame_extractor = nn.Conv3d(
                in_channels=512,  # Output channels of layer4
                out_channels=self.feature_dim,
                kernel_size=(1, 1, 1)
            )
            
            # Temporal attention layer
            self.temporal_attention = TemporalAttention(self.feature_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of added layers"""
        if self.use_attention:
            # Initialize frame extractor
            nn.init.kaiming_normal_(self.frame_extractor.weight, mode='fan_out', nonlinearity='relu')
            if self.frame_extractor.bias is not None:
                nn.init.constant_(self.frame_extractor.bias, 0)
                
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    # _freeze_layers method remains unchanged
    def _freeze_layers(self, layer_names):
        """Freeze specified layers in the backbone network"""
        if 'stem' in layer_names:
            for param in self.backbone.stem.parameters():
                param.requires_grad = False
        
        if 'layer1' in layer_names:
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False
        
        if 'layer2' in layer_names:
            for param in self.backbone.layer2.parameters():
                param.requires_grad = False
        
        if 'layer3' in layer_names:
            for param in self.backbone.layer3.parameters():
                param.requires_grad = False
        
        if 'layer4' in layer_names:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass through R(2+1)D network with optional temporal attention"""
        # Handle different input types
        if isinstance(x, tuple):
            x = x[0]  # Extract frames if given as a tuple
        
        # Check if input is in the format [B, T, C, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Permute to [B, C, T, H, W] for 3D CNN
            x = x.permute(0, 2, 1, 3, 4)
        
        # Process with or without attention
        if self.use_attention:
            # Get features before pooling
            x = self.early_backbone(x)  # [B, 512, T, H, W]
            
            # Extract per-frame features
            frame_features = self.frame_extractor(x)  # [B, feature_dim, T, H, W]
            
            # Global average pooling over spatial dimensions but preserve temporal
            frame_features = frame_features.mean([3, 4])  # [B, feature_dim, T]
            
            # Rearrange to [B, T, feature_dim] for attention
            frame_features = frame_features.permute(0, 2, 1)  # [B, T, feature_dim]
            
            # Apply temporal attention
            features, self.last_attention_weights = self.temporal_attention(frame_features)
        else:
            # Original pathway - full backbone with pooling
            features = self.backbone(x)
        
        # Classification
        outputs = self.classifier(features)
        
        # For compatibility with existing code, return just outputs
        return outputs
    
    def get_attention_weights(self):
        """Method to access the last computed attention weights for visualization"""
        if self.use_attention and hasattr(self, 'last_attention_weights'):
            return self.last_attention_weights
        return None