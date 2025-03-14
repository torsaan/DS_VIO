# models/model_r2plus1d.py
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

class R2Plus1DNet(nn.Module):
    """
    R(2+1)D Model for violence detection.
    Based on the paper: "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_prob=0.5, frozen_layers=None):
        super(R2Plus1DNet, self).__init__()
        
        # Load pre-trained model
        self.backbone = r2plus1d_18(pretrained=pretrained)
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with identity for feature extraction
        self.backbone.fc = nn.Identity()
        
        # Freeze specified layers if needed
        if frozen_layers:
            self._freeze_layers(frozen_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the classifier
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the classifier layers"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        """
        Forward pass through R(2+1)D network
        
        Args:
            x: Input tensor of shape [B, T, C, H, W] or tuple containing frames
            
        Returns:
            Class logits
        """
        # Handle different input types
        if isinstance(x, tuple):
            x = x[0]  # Extract frames if given as a tuple
        
        # Check if input is in the format [B, T, C, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Permute to [B, C, T, H, W] for 3D CNN
            x = x.permute(0, 2, 1, 3, 4)
        
        # Extract features
        features = self.backbone(x)
        
        # Classification
        outputs = self.classifier(features)
        
        return outputs