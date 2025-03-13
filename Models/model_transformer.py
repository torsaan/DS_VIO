# models/model_transformer.py
import torch
import torch.nn as nn
import timm

class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2, use_pose=False, embed_dim=None, num_heads=8, num_layers=2, dropout=0.1):
        super(VideoTransformer, self).__init__()
        # use_pose parameter is kept for backward compatibility but ignored
        
        # Use timm's ViT but remove the classification head
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.embed_dim = self.backbone.embed_dim  # Usually 768 for base ViT
        
        # Create a new temporal transformer encoder - use actual backbone embed_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,  # Use actual embedding dimension from backbone
            nhead=num_heads, 
            dim_feedforward=self.embed_dim*4,
            dropout=dropout,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Video-only classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 512),  # Use actual embedding dim 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, inputs):
        """
        Forward pass through the transformer model.
        
        Args:
            inputs: Input video frames tensor of shape [B, T, C, H, W]
            
        Returns:
            Classification output
        """
        # Process video frames
        x = inputs
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Process each frame with ViT
        frame_features = []
        for t in range(seq_length):
            features = self.backbone.forward_features(x[:, t])  # [B, num_patches + 1, embed_dim]
            cls_token = features[:, 0]  # [B, embed_dim]
            frame_features.append(cls_token)
        
        # Stack temporal sequence
        x = torch.stack(frame_features, dim=1)  # [B, T, embed_dim]
        
        # Reshape for transformer: [T, B, embed_dim]
        x = x.transpose(0, 1)
        
        # Apply temporal transformer
        x = self.temporal_encoder(x)  # [T, B, embed_dim]
        
        # Mean pooling
        x = x.mean(dim=0)  # [B, embed_dim]
        
        # Classification
        outputs = self.classifier(x)
        
        return outputs