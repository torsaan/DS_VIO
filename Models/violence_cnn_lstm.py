# Models/violence_cnn_lstm.py
import torch
import torch.nn as nn
import torchvision.models as models

class ViolenceCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, num_layers=2, dropout=0.5, 
                 activation='relu', freeze_cnn=True, finetune_last_n_layers=0, backbone='resnet50'):
        super(ViolenceCNNLSTM, self).__init__()
        
        # Load pre-trained CNN backbone based on parameter
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.cnn_feature_dim = 2048
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            self.cnn_feature_dim = 512
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.cnn_feature_dim = 2048
        else:
            # Default to ResNet50
            resnet = models.resnet50(pretrained=True)
            self.cnn_feature_dim = 2048
            
        modules = list(resnet.children())[:-1]  # Remove the final fully connected layer
        self.cnn = nn.Sequential(*modules)
        
        # Control CNN freezing based on parameters
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
            # Optionally unfreeze last N layers for fine-tuning
            if finetune_last_n_layers > 0:
                trainable_layers = list(resnet.children())[-finetune_last_n_layers-1:-1]
                for layer in trainable_layers:
                    for param in layer.parameters():
                        param.requires_grad = True
            
        # LSTM for temporal modeling of CNN features
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout  # Works between LSTM layers (num_layers > 1)
        )
        
        # Add output dropout for LSTM (previously missing)
        self.lstm_output_dropout = nn.Dropout(dropout)
        
        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(lstm_hidden_size)
        
        # Choose activation function based on parameter
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()  # default
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.dropout = nn.Dropout(dropout)  # Additional dropout between FC layers
        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_classes)
        
        # Initialize weights for the fully connected layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize the weights of the fully connected layers"""
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, C, H, W] where:
               B = batch size,
               T = number of frames (sequence length),
               C = number of channels,
               H, W = height and width of frame.
        Returns:
            Output logits of shape [B, num_classes].
        """
        B, T, C, H, W = x.shape
        # Reshape to process each frame independently: [B*T, C, H, W]
        x = x.view(B * T, C, H, W)
        
        # Extract per-frame visual features using the CNN backbone
        with torch.set_grad_enabled(any(p.requires_grad for p in self.cnn.parameters())):
            features = self.cnn(x)  # shape [B*T, feature_dim, 1, 1]
        
        features = features.view(B, T, self.cnn_feature_dim)  # shape [B, T, feature_dim]
        
        # Process the sequence of CNN features with LSTM (Let PyTorch initialize states)
        lstm_out, _ = self.lstm(features)  # lstm_out shape: [B, T, lstm_hidden_size]
        
        # Use the last time-step's output for classification
        last_output = lstm_out[:, -1, :]  # shape: [B, lstm_hidden_size]
        
        # Apply layer normalization and output dropout to LSTM output (new)
        last_output = self.layer_norm(last_output)
        last_output = self.lstm_output_dropout(last_output)
        
        # Pass through fully connected layers with activation and dropout
        out = self.activation(self.fc1(last_output))
        out = self.dropout(out)  # Apply dropout between FC layers
        out = self.fc2(out)
        
        return out
