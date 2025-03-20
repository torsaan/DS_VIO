import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class SlowFastNetwork(nn.Module):
    """
    SlowFast Network with three configurable fusion strategies:
    - late: Only fuses at the end (after all residual blocks)
    - middle: Fuses at the 3rd and 4th residual blocks
    - early: Fuses at the 2nd, 3rd, and 4th residual blocks
    """
    def __init__(self, num_classes=2, pretrained=True, alpha=8, beta=1/8, dropout_prob=0.5, fusion_type='late'):
        super(SlowFastNetwork, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.fusion_type = fusion_type.lower()
        
        # Validate fusion type
        valid_fusion_types = ['late', 'middle', 'early']
        if self.fusion_type not in valid_fusion_types:
            raise ValueError(f"fusion_type must be one of {valid_fusion_types}, but got {fusion_type}")

        # Initialize slow pathway
        self.slow_stem = nn.Sequential(
            r3d_18(pretrained=pretrained).stem
        )
        self.slow_layer1 = r3d_18(pretrained=pretrained).layer1
        self.slow_layer2 = r3d_18(pretrained=pretrained).layer2
        self.slow_layer3 = r3d_18(pretrained=pretrained).layer3
        self.slow_layer4 = r3d_18(pretrained=pretrained).layer4
        
        # Initialize fast pathway
        self.fast_stem = nn.Sequential(
            r3d_18(pretrained=pretrained).stem
        )
        self.fast_layer1 = r3d_18(pretrained=pretrained).layer1
        self.fast_layer2 = r3d_18(pretrained=pretrained).layer2
        self.fast_layer3 = r3d_18(pretrained=pretrained).layer3
        self.fast_layer4 = r3d_18(pretrained=pretrained).layer4
        
        # Channel dimensions at each stage
        self.dim_layer1 = 64
        self.dim_layer2 = 128
        self.dim_layer3 = 256
        self.dim_layer4 = 512
        
        # Lateral connections (fast → slow) for each fusion strategy
        if self.fusion_type == 'early':
            # Early fusion - lateral connections after 2nd, 3rd, and 4th layers
            self.lateral_layer2 = nn.Conv3d(
                self.dim_layer2, int(self.dim_layer2 * beta),
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.lateral_layer3 = nn.Conv3d(
                self.dim_layer3, int(self.dim_layer3 * beta),
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.lateral_layer4 = nn.Conv3d(
                self.dim_layer4, int(self.dim_layer4 * beta),
                kernel_size=1, stride=1, padding=0, bias=False
            )
            # Final fused dimension
            final_slow_dim = self.dim_layer4 + int(self.dim_layer4 * beta)
            
        elif self.fusion_type == 'middle':
            # Middle fusion - lateral connections after 3rd and 4th layers
            self.lateral_layer3 = nn.Conv3d(
                self.dim_layer3, int(self.dim_layer3 * beta),
                kernel_size=1, stride=1, padding=0, bias=False
            )
            self.lateral_layer4 = nn.Conv3d(
                self.dim_layer4, int(self.dim_layer4 * beta),
                kernel_size=1, stride=1, padding=0, bias=False
            )
            # Final fused dimension
            final_slow_dim = self.dim_layer4 + int(self.dim_layer4 * beta)
            
        else:  # 'late' fusion
            # No lateral connections, only feature fusion at the end
            final_slow_dim = self.dim_layer4
            
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fast pathway feature reduction
        self.fast_reducer = nn.Linear(self.dim_layer4, int(self.dim_layer4 * beta))
        
        # Final classifier with properly calculated input dimension
        fused_dim = final_slow_dim + int(self.dim_layer4 * beta)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
                mode='nearest', 
                align_corners=False
            )
        
        # Slow pathway: use every α-th frame
        slow_x = x[:, :, ::self.alpha, :, :]
        
        # Fast pathway: use full frame rate
        fast_x = x
        
        # Process through stem
        slow_x = self.slow_stem(slow_x)
        fast_x = self.fast_stem(fast_x)
        
        # Layer 1
        slow_x = self.slow_layer1(slow_x)
        fast_x = self.fast_layer1(fast_x)
        
        # Layer 2
        slow_x = self.slow_layer2(slow_x)
        fast_x = self.fast_layer2(fast_x)
        
        # Early fusion - after layer 2
        if self.fusion_type == 'early':
            # Lateral connection
            fast_to_slow = self.lateral_layer2(fast_x)
            
            # Time dimension matching
            if fast_to_slow.shape[2] != slow_x.shape[2]:
                fast_to_slow = torch.nn.functional.interpolate(
                    fast_to_slow,
                    size=(slow_x.shape[2], slow_x.shape[3], slow_x.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
            
            # Fusion by concatenation
            slow_x = torch.cat([slow_x, fast_to_slow], dim=1)
        
        # Layer 3
        slow_x = self.slow_layer3(slow_x)
        fast_x = self.fast_layer3(fast_x)
        
        # Middle or early fusion - after layer 3
        if self.fusion_type in ['middle', 'early']:
            # Lateral connection
            fast_to_slow = self.lateral_layer3(fast_x)
            
            # Time dimension matching
            if fast_to_slow.shape[2] != slow_x.shape[2]:
                fast_to_slow = torch.nn.functional.interpolate(
                    fast_to_slow,
                    size=(slow_x.shape[2], slow_x.shape[3], slow_x.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
            
            # Fusion by concatenation
            slow_x = torch.cat([slow_x, fast_to_slow], dim=1)
        
        # Layer 4
        slow_x = self.slow_layer4(slow_x)
        fast_x = self.fast_layer4(fast_x)
        
        # Middle or early fusion - after layer 4
        if self.fusion_type in ['middle', 'early']:
            # Lateral connection
            fast_to_slow = self.lateral_layer4(fast_x)
            
            # Time dimension matching
            if fast_to_slow.shape[2] != slow_x.shape[2]:
                fast_to_slow = torch.nn.functional.interpolate(
                    fast_to_slow,
                    size=(slow_x.shape[2], slow_x.shape[3], slow_x.shape[4]),
                    mode='trilinear',
                    align_corners=False
                )
            
            # Fusion by concatenation
            slow_x = torch.cat([slow_x, fast_to_slow], dim=1)
        
        # Global average pooling
        slow_x = self.avg_pool(slow_x).view(slow_x.size(0), -1)
        fast_x = self.avg_pool(fast_x).view(fast_x.size(0), -1)
        
        # Reduce fast features
        reduced_fast = self.fast_reducer(fast_x)
        
        # Late fusion (all models do this final concatenation)
        combined_features = torch.cat([slow_x, reduced_fast], dim=1)
        
        # Classification
        out = self.classifier(combined_features)
        return out