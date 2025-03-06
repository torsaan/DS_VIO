# models/model_transformer.py
import torch
import torch.nn as nn
import timm

class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2, use_pose=False, pose_input_size=66):
        super(VideoTransformer, self).__init__()
        
        # Use timm's ViT but remove the classification head
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        embed_dim = self.backbone.embed_dim
        
        # Create a new temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=8, 
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Flag for using pose data
        self.use_pose = use_pose
        
        if use_pose:
            # Pose processing branch
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64)
            )
            
            # Transformer encoder for pose data
            pose_encoder_layer = nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=256,
                dropout=0.1,
                activation='gelu'
            )
            self.pose_transformer = nn.TransformerEncoder(pose_encoder_layer, num_layers=2)
            
            # Combined classifier
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim + 64, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        else:
            # Video-only classifier
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        
    def forward(self, inputs):
        if self.use_pose:
            # Unpack inputs
            x, pose = inputs
            
            # Process video frames
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
            
            # Use mean of temporal features
            x = x.mean(dim=0)  # [B, embed_dim]
            
            # Process pose data
            pose_features = self.pose_encoder(pose.reshape(-1, pose.size(-1)))
            pose_features = pose_features.view(batch_size, seq_length, -1)  # [B, T, 64]
            
            # Transpose for transformer: [T, B, 64]
            pose_features = pose_features.transpose(0, 1)
            
            # Apply pose transformer
            pose_features = self.pose_transformer(pose_features)  # [T, B, 64]
            
            # Mean pooling over time
            pose_features = pose_features.mean(dim=0)  # [B, 64]
            
            # Combine features
            combined_features = torch.cat([x, pose_features], dim=1)
            
            # Classification
            outputs = self.classifier(combined_features)
        else:
            # Process only video frames
            x = inputs
            batch_size, seq_length = x.size(0), x.size(1)
            
            # Process each frame with ViT
            frame_features = []
            for t in range(seq_length):
                features = self.backbone.forward_features(x[:, t])
                cls_token = features[:, 0]
                frame_features.append(cls_token)
            
            # Stack temporal sequence
            x = torch.stack(frame_features, dim=1)
            
            # Reshape for transformer
            x = x.transpose(0, 1)
            
            # Apply temporal transformer
            x = self.temporal_encoder(x)
            
            # Mean pooling
            x = x.mean(dim=0)
            
            # Classification
            outputs = self.classifier(x)
        
        return outputs