import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class SlowFastNetwork(nn.Module):
    """
    Proper SlowFast Network with differentiated temporal resolutions and channel capacities.
    The slow pathway processes every α-th frame while the fast pathway processes the full frame rate.
    Fast pathway features are reduced by a factor β before fusion.
    """
    def __init__(self, num_classes=2, pretrained=True, alpha=8, beta=1/8, dropout_prob=0.5):
        super(SlowFastNetwork, self).__init__()
        self.alpha = alpha
        self.beta = beta

        # Slow pathway: processes subsampled frames.
        self.slow_model = r3d_18(pretrained=pretrained)
        self.slow_features = self.slow_model.fc.in_features  # e.g., 512
        self.slow_model.fc = nn.Identity()

        # Fast pathway: processes full frame rate.
        self.fast_model = r3d_18(pretrained=pretrained)
        self.fast_features = self.fast_model.fc.in_features  # e.g., 512
        self.fast_model.fc = nn.Identity()
        # Reduce fast features by factor beta.
        self.fast_reduction = nn.Linear(self.fast_features, int(self.fast_features * beta))

        # Final classifier fuses slow and reduced fast features.
        fused_dim = self.slow_features + int(self.fast_features * beta)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle different input formats
        if x.dim() == 5 and x.shape[1] != 3:  # If format is [B, T, C, H, W]
            x = x.permute(0, 2, 1, 3, 4)  # Convert to [B, C, T, H, W]
        
        # Ensure there are enough frames for temporal sampling
        if x.size(2) < self.alpha:
            # Upsample temporally if fewer frames than alpha
            x = torch.nn.functional.interpolate(
                x, 
                size=(max(self.alpha, x.size(2)), x.size(3), x.size(4)),
                mode='trilinear', 
                align_corners=False
            )
        
        # Slow pathway: use every α-th frame.
        slow_x = x[:, :, ::self.alpha, :, :]
        
        # Fast pathway: use full frame rate.
        fast_x = x
        
        # Ensure both pathways have valid input
        if slow_x.size(2) < 1:
            raise ValueError(f"Slow pathway has no frames after sampling. Input had {x.size(2)} frames, alpha={self.alpha}")
            
        # Process through pathways
        slow_features = self.slow_model(slow_x)  # [B, slow_features]
        fast_features = self.fast_model(fast_x)  # [B, fast_features]
        
        # Feature fusion
        reduced_fast = self.fast_reduction(fast_features)  # [B, fast_features * beta]
        combined_features = torch.cat([slow_features, reduced_fast], dim=1)
        
        # Classification
        out = self.classifier(combined_features)
        return out