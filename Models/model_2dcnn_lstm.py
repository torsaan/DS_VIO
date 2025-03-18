# models/model_2dcnn_lstm.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class Model2DCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, lstm_num_layers=2, dropout_prob=0.5, pretrained=True):
        super(Model2DCNNLSTM, self).__init__()
        # Load pre-trained ResNet but remove the last fully connected layer
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.backbone = nn.Sequential(*modules)
        
        # Feature dimension from resnet50
        self.feature_dim = 2048
        
        # LSTM for temporal modeling of visual features
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_prob,  # This works correctly with num_layers=2
            bidirectional=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_hidden_size, num_classes)
        )
        
    def forward(self, frames):
        batch_size, seq_length = frames.size(0), frames.size(1)
        
        # Reshape for 2D CNN: [B*T, C, H, W]
        frames = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))
        
        # Extract features
        with torch.no_grad():  # Freeze the CNN backbone
            visual_features = self.backbone(frames)  # [B*T, feature_dim, 1, 1]
            visual_features = visual_features.squeeze(-1).squeeze(-1)  # [B*T, feature_dim]
        
        # Reshape for LSTM: [B, T, feature_dim]
        visual_features = visual_features.view(batch_size, seq_length, -1)
        
        # LSTM forward
        visual_features, _ = self.lstm(visual_features)  # [B, T, hidden_size*2]
        
        # Use last time step output
        visual_features = visual_features[:, -1, :]  # [B, hidden_size*2]
        
        # Classification
        outputs = self.classifier(visual_features)
        
        return outputs