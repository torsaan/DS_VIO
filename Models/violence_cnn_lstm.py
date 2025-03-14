# Models/violence_cnn_lstm.py
import torch
import torch.nn as nn
import torchvision.models as models

class ViolenceCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=512, num_layers=2, dropout=0.5, activation='relu'):
        super(ViolenceCNNLSTM, self).__init__()
        # Load a pre-trained ResNet50 and remove the classification head
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the final fully connected layer
        self.cnn = nn.Sequential(*modules)
        self.cnn_feature_dim = 2048  # ResNet50 outputs a 2048-dim vector per frame
        
        # Optionally freeze CNN parameters if you want to use it as a fixed feature extractor
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        # LSTM for temporal modeling of CNN features
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Choose activation function based on parameter
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # default
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_classes)
        
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
        features = self.cnn(x)  # shape [B*T, 2048, 1, 1]
        features = features.view(B, T, self.cnn_feature_dim)  # shape [B, T, 2048]
        
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=features.device)
        c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=features.device)
        
        # Process the sequence of CNN features with LSTM
        lstm_out, _ = self.lstm(features, (h0, c0))  # lstm_out shape: [B, T, lstm_hidden_size]
        
        # Use the last time-step's output for classification
        last_output = lstm_out[:, -1, :]  # shape: [B, lstm_hidden_size]
        
        # Pass through fully connected layers with activation and dropout
        out = self.activation(self.fc1(last_output))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Test with dummy data: e.g. 4 samples, 16 frames, 3 channels, 224x224 resolution.
    model = ViolenceCNNLSTM(num_classes=2, lstm_hidden_size=512, num_layers=2, dropout=0.5, activation='relu')
    dummy_input = torch.randn(4, 16, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [4, 2]