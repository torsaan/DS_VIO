# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders
from Models.dl_model import ViolenceLSTM
from solver import Solver
import hyperparameters as hp

def main():
    # Load data paths and labels here (e.g., from a config or custom function)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = load_data_paths()
    
    train_loader, val_loader, _ = get_dataloaders(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, hp.POSE_DIR, hp.BATCH_SIZE)
    
    model = ViolenceLSTM(
        input_size=66,
        hidden_size=hp.DL_MODEL_PARAMS['hidden_size'],
        num_layers=hp.DL_MODEL_PARAMS['lstm_layers'],
        num_classes=2,
        dropout=hp.DL_MODEL_PARAMS['dropout'],
        activation=hp.DL_MODEL_PARAMS['activation_functions'][0]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATES['adam'])
    
    solver = Solver(model, train_loader, val_loader, criterion, optimizer, device)
    solver.fit(hp.NUM_EPOCHS)

if __name__ == '__main__':
    main()
