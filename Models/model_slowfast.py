# Models/model_slowfast.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class SlowFastNetwork(nn.Module):
    """
    SlowFast Networks for Video Recognition - Simplified implementation that works with testing
    """
    def __init__(self, num_classes=2, pretrained=True, alpha=8, beta=1/8, 
                 fusion_places=['res2', 'res3', 'res4', 'res5'], dropout_prob=0.5):
        """
        Initialize SlowFast Network

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights for the backbone
            alpha: Speed ratio (not directly used in this implementation)
            beta: Channel ratio (not directly used in this implementation)
            fusion_places: Stages where lateral connections would be applied
            dropout_prob: Dropout probability for final classifier
        """
        super(SlowFastNetwork, self).__init__()
        
        # Create separate models for slow and fast pathways
        self.slow_model = r3d_18(pretrained=pretrained)
        self.fast_model = r3d_18(pretrained=pretrained)
        
        # Get feature dimensions
        self.slow_features = self.slow_model.fc.in_features
        self.fast_features = self.fast_model.fc.in_features
        
        # Remove the final FC layers
        self.slow_model.fc = nn.Identity()
        self.fast_model.fc = nn.Identity()
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.slow_features + self.fast_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights of the classifier"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through SlowFast network
        
        Args:
            x: Input tensor of shape [B, C, T, H, W] or [B, T, C, H, W] or tuple
            
        Returns:
            Class logits
        """
        # Handle different input types
        if isinstance(x, tuple):
            x = x[0]  # Extract frames if given as a tuple
        
        # Ensure input is in format [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Input is [B, T, C, H, W], permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        # Process through slow pathway
        slow_features = self.slow_model(x)
        
        # Process through fast pathway
        fast_features = self.fast_model(x)
        
        # Concatenate features
        combined_features = torch.cat([slow_features, fast_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output