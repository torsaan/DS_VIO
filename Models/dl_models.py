# Models/dl_model.py
import torch
import torch.nn as nn

class ViolenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, num_classes=2, dropout=0.5, activation='relu'):
        super(ViolenceLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # LSTM layer to capture temporal dynamics
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Choose activation function based on parameter
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()  # default
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        # x: (batch, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_length, hidden_size)
        out = out[:, -1, :]  # take last time step
        out = self.activation(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    # Test with dummy data: 4 samples, 32 time steps, 66 features (e.g., 33 keypoints*2)
    model = ViolenceLSTM(input_size=66, hidden_size=512, num_layers=2, num_classes=2, dropout=0.5, activation='relu')
    dummy_input = torch.randn(4, 32, 66)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: [4, 2]
