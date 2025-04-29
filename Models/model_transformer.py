# models/model_transformer.py
import torch
import torch.nn as nn
import timm

class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2, max_seq_length=100, freeze_backbone=True, 
                 embed_dim=None, num_heads=8, num_layers=2, dropout=0.1):
        super(VideoTransformer, self).__init__()
        
        # Use timm's ViT but remove the classification head
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.embed_dim = self.backbone.embed_dim  # Usually 768 for base ViT
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Optionally unfreeze the last few blocks for fine-tuning
            for param in self.backbone.blocks[-2:].parameters():
                param.requires_grad = True
        
        # Temporal position encoding
        self.register_buffer('temporal_pos_encoding', 
                           torch.zeros(1, max_seq_length, self.embed_dim))
        
        # Initialize position encoding with truncated normal distribution
        nn.init.trunc_normal_(self.temporal_pos_encoding, std=0.02)
        
        # Create temporal transformer encoder with proper parameter usage
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,  # Use parameter
            dim_feedforward=self.embed_dim*4,
            dropout=dropout,  # Use parameter
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # Use parameter
        
        # Video-only classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),  # Use parameter
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Handle different input formats
        if isinstance(x, tuple):
            x = x[0]  # Extract video frames if given as a tuple
            
        if x.dim() == 5 and x.shape[1] != 3:  # If format is [B, T, C, H, W]
            # Continue with [B, T, C, H, W] format
            pass
        else:
            # Permute if in [B, C, T, H, W] format
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, T, C, H, W]
        
        # Process video frames
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Reshape to process frames in batch
        x_reshaped = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))  # [B*T, C, H, W]
        
        # Process all frames at once through ViT backbone
        with torch.no_grad() if all(not p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            features = self.backbone.forward_features(x_reshaped)  # [B*T, num_patches+1, embed_dim]
        
        # Extract CLS tokens
        cls_tokens = features[:, 0]  # [B*T, embed_dim]
        
        # Reshape back to sequence format
        frame_features = cls_tokens.view(batch_size, seq_length, -1)  # [B, T, embed_dim]
        
        # Add positional encoding
        frame_features = frame_features + self.temporal_pos_encoding[:, :seq_length, :]
        
        # Transpose for transformer: [T, B, embed_dim]
        frame_features = frame_features.transpose(0, 1)
        
        # Apply temporal transformer
        temporal_features = self.temporal_encoder(frame_features)  # [T, B, embed_dim]
        
        # Global representation - mean pooling across time
        global_features = temporal_features.mean(dim=0)  # [B, embed_dim]
        
        # Classification
        outputs = self.classifier(global_features)
        
        return outputs
    
    def get_attention_maps(self, x):
        """Extract attention maps from the temporal transformer for visualization"""
        # This requires modifying the transformer to store attention weights
        # Placeholder for now - would need custom transformer implementation
        return None