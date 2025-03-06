# test.py
import torch
from dataloader import get_dataloaders
from Models.dl_model import ViolenceLSTM
import hyperparameters as hp

def test_dl_model(test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceLSTM(
        input_size=66,
        hidden_size=hp.DL_MODEL_PARAMS['hidden_size'],
        num_layers=hp.DL_MODEL_PARAMS['lstm_layers'],
        num_classes=2,
        dropout=hp.DL_MODEL_PARAMS['dropout'],
        activation=hp.DL_MODEL_PARAMS['activation_functions'][0]
    )
    model.load_state_dict(torch.load('Models/dl_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for _, pose_keypoints, labels in test_loader:
            inputs = pose_keypoints.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    print(f"DL Model Test Accuracy: {correct/total*100:.2f}%")

if __name__ == '__main__':
    # Replace with your test data paths/labels as needed
    _, _, test_loader = get_dataloaders([], [], [], [], ['./Data/Violence/video3.mp4'], [1], hp.POSE_DIR, hp.BATCH_SIZE)
    test_dl_model(test_loader)
