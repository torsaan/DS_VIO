import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models.video import r3d_18
from torchvision.models import resnet50
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import timm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Constants
NUM_FRAMES = 32  # Number of frames to sample from each video
FRAME_HEIGHT, FRAME_WIDTH = 224, 224  # Standard input size
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DATASET AND DATA LOADING =====

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, num_frames=32, model_type='3d_cnn', frame_size=(112,112)):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.model_type = model_type  # Used to determine preprocessing
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)
    
    def read_video(self, video_path):
        # Read video and extract frames
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}")
            return None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Warning: No frames found in {video_path}")
            return None
            
        # Calculate sampling rate to get num_frames
        if total_frames >= self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # If video is too short, loop frames
            indices = np.mod(np.arange(self.num_frames), total_frames)
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                # Convert from BGR to RGB
                frame = np.zeros((self.frame_size[0], self.frame_size[1],3), dtype=np.uint8)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #frames.append(frame)
            else:
                # If reading fails, create a black frame
                #frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)

            frames.append(frame)
        
        cap.release()
        return np.array(frames, dtype=np.uint8)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video frames
        frames = self.read_video(video_path)
        
        if frames is None or len(frames) < self.num_frames:
            frames = np.zeros((self.num_frames, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
        
        # Process frames based on model type
        if self.model_type == '3d_cnn':
            if self.transform:
                transformed_frames = [self.transform(Image.fromarray(frame)) for frame in frames]
                frames_tensor = torch.stack(transformed_frames, dim=1)  # Shape: [C, T, H, W]
            else:
                frames_tensor = torch.from_numpy(frames.transpose(3, 0, 1, 2).astype(np.float32))  # Convert to [C, T, H, W]
                frames_tensor /= 255.0  # Normalize
        
        elif self.model_type in ['2d_cnn_lstm', 'transformer', 'transfer_learning']:
            if self.transform:
                transformed_frames = [self.transform(Image.fromarray(frame)) for frame in frames]
                frames_tensor = torch.stack(transformed_frames)  # Shape: [T, C, H, W]
            else:
                frames = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0  # Convert to [T, C, H, W]
                frames_tensor = torch.from_numpy(frames)
        
        return frames_tensor, torch.tensor(label, dtype=torch.long)

# Create transforms for data augmentation
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((FRAME_HEIGHT, FRAME_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# ===== MODEL 1: 3D CNN (R3D) =====

class Model3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Model3DCNN, self).__init__()
        # Load pre-trained 3D ResNet model
        self.backbone = r3d_18(pretrained=True)
        
        # Replace the final classification layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Input shape: [B, C, T, H, W]
        return self.backbone(x)

# ===== MODEL 2: 2D CNN + LSTM =====

class Model2DCNNLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_size=512):
        super(Model2DCNNLSTM, self).__init__()
        # Load pre-trained ResNet but remove the last fully connected layer
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.backbone = nn.Sequential(*modules)
        
        # Feature dimension from resnet50
        self.feature_dim = 2048
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [B, T, C, H, W]
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Reshape for 2D CNN: [B*T, C, H, W]
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        # Extract features
        with torch.no_grad():  # Freeze the CNN backbone
            x = self.backbone(x)  # [B*T, feature_dim, 1, 1]
        
        # Reshape for LSTM: [B, T, feature_dim]
        x = x.view(batch_size, seq_length, -1)
        
        # LSTM forward
        x, _ = self.lstm(x)  # [B, T, hidden_size*2]
        
        # Use last time step output for classification
        x = x[:, -1, :]  # [B, hidden_size*2]
        
        # Classification
        x = self.classifier(x)
        
        return x

# ===== MODEL 3: TRANSFORMER =====

class VideoTransformer(nn.Module):
    def __init__(self, num_classes=2):
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [B, T, C, H, W]
        batch_size, seq_length = x.size(0), x.size(1)
        
        # Process each frame with ViT (remove classification token)
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
        
        # Use mean of temporal features for classification
        x = x.mean(dim=0)  # [B, embed_dim]
        
        # Classification
        x = self.classifier(x)
        
        return x

# ===== MODEL 4: TRANSFER LEARNING (I3D) =====

class TransferLearningI3D(nn.Module):
    def __init__(self, num_classes=2):
        super(TransferLearningI3D, self).__init__()
        
        # Load pre-trained I3D model (using torchvision's implementation)
        # Note: This is a placeholder - in a real implementation you would import
        # the actual I3D model which might require external libraries
        
        # For illustration, we'll use a simpler 3D ResNet and pretend it's I3D
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        
        # Replace the final classification layer
        self.backbone.blocks[-1].proj = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        # Input shape: [B, C, T, H, W]
        return self.backbone(x)

# ===== TRAINING AND EVALUATION FUNCTIONS =====

def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets

# ===== MAIN FUNCTION =====

def load_video_frames(video_path, num_frames=32):

    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"Warning: No frames found in {video_path}")
        cap.release()
        return None

    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Loop frames to ensure we get num_frames
        frame_indices = np.array([i % total_frames for i in range(num_frames)])

    frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((112, 112, 3), dtype=np.uint8)  # Black frame for missing frames
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return np.array(frames)

def prepare_data(data_dir, num_frames = 32, frame_size =(112,112)):

    # Collect video paths and labels
    violence_dir = os.path.join(data_dir, "Violence")
    nonviolence_dir = os.path.join(data_dir, "NonViolence")
    
    video_paths = []
    labels = []
    
    # Violence videos (label 1)
    for video_name in os.listdir(violence_dir):
        if video_name.endswith(".mp4"):
            video_paths.append(os.path.join(violence_dir, video_name))
            labels.append(1)
    
    # Non-violence videos (label 0)
    for video_name in os.listdir(nonviolence_dir):
        if video_name.endswith(".mp4"):
            video_paths.append(os.path.join(nonviolence_dir, video_name))
            labels.append(0)
    
    # Now load frames for each video and store as tensor
    video_frames = []
    for video_path in video_paths:
        frames = load_video_frames(video_path, num_frames)
        video_frames.append(frames)

    # Split into train, validation, test
    train_frames, test_frames, train_labels, test_labels = train_test_split(
        video_frames, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_frames, test_frames, val_labels, test_labels = train_test_split(
        test_frames, test_labels, test_size=0.5, random_state=42, stratify=test_labels
    )
    
    return (train_frames, train_labels), (val_frames, val_labels), (test_frames, test_labels)

def train_model(model_name, model, train_loader, val_loader, num_epochs=30, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{model_name}_model.pth")
            print(f"Model saved as best_{model_name}_model.pth")
        
        print("-" * 50)
    
    return model

def create_model_and_dataloaders(model_type, train_data, val_data, test_data):
    train_transform, val_transform = get_transforms()
    
    # Create datasets with appropriate model_type
    train_dataset = VideoDataset(
        train_data[0], train_data[1], transform=train_transform, 
        num_frames=NUM_FRAMES, model_type=model_type
    )
    val_dataset = VideoDataset(
        val_data[0], val_data[1], transform=val_transform, 
        num_frames=NUM_FRAMES, model_type=model_type
    )
    test_dataset = VideoDataset(
        test_data[0], test_data[1], transform=val_transform, 
        num_frames=NUM_FRAMES, model_type=model_type
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Initialize appropriate model
    if model_type == '3d_cnn':
        model = Model3DCNN().to(DEVICE)
    elif model_type == '2d_cnn_lstm':
        model = Model2DCNNLSTM().to(DEVICE)
    elif model_type == 'transformer':
        model = VideoTransformer().to(DEVICE)
    elif model_type == 'transfer_learning':
        model = TransferLearningI3D().to(DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, train_loader, val_loader, test_loader

def evaluate_and_compare_models(models_dict, test_loaders):
    """Evaluate all models and compare their performance"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    results = {}
    
    for model_name, model in models_dict.items():
        # Load best weights
        model.load_state_dict(torch.load(f"best_{model_name}_model.pth"))
        model.eval()
        
        test_loader = test_loaders[model_name]
        criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc, all_preds, all_targets = validate(model, test_loader, criterion, DEVICE)
        
        print(f"\n=== {model_name} Test Results ===")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        print("\nClassification Report:")
        report = classification_report(all_targets, all_preds, target_names=["NonViolence", "Violence"])
        print(report)
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_targets, all_preds)
        print(cm)
        
        results[model_name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    # Compare models
    print("\n=== Model Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: {result['accuracy']:.2f}%")
    
    # Determine best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.2f}%")
    
    return results

def ensemble_predictions(models_dict, test_loaders):
    """Create an ensemble of all models using majority voting"""
    all_model_preds = []
    targets = None
    
    for model_name, model in models_dict.items():
        # Load best weights
        model.load_state_dict(torch.load(f"best_{model_name}_model.pth"))
        model.eval()
        
        test_loader = test_loaders[model_name]
        preds = []
        
        with torch.no_grad():
            for inputs, targets_batch in test_loader:
                if targets is None:
                    targets = targets_batch.numpy()
                else:
                    targets = np.concatenate([targets, targets_batch.numpy()])
                
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
        
        all_model_preds.append(preds)
    
    # Convert to numpy arrays
    all_model_preds = np.array(all_model_preds)
    
    # Majority voting
    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), 
        axis=0, 
        arr=all_model_preds
    )
    
    # Calculate ensemble accuracy
    ensemble_acc = 100. * (ensemble_preds == targets).mean()
    
    print("\n=== Ensemble Results ===")
    print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")
    
    # Classification report
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    report = classification_report(targets, ensemble_preds, target_names=["NonViolence", "Violence"])
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(targets, ensemble_preds)
    print(cm)
    
    return ensemble_preds, targets, ensemble_acc

def main():
    print("Main function started...")

    data_dir = 'C:\Github\DS_VIO\Data\VioNonVio'
    
    # Prepare data
    train_data, val_data, test_data = prepare_data(data_dir)
    
    # Model types to train
    model_types = ['3d_cnn', '2d_cnn_lstm', 'transformer', 'transfer_learning']
    
    # Train all models
    models = {}
    test_loaders = {}
    
    for model_type in model_types:
        print(f"\n{'='*20} Training {model_type} {'='*20}\n")
        
        model, train_loader, val_loader, test_loader = create_model_and_dataloaders(
            model_type, train_data, val_data, test_data
        )
        
        trained_model = train_model(
            model_type, model, train_loader, val_loader, num_epochs=NUM_EPOCHS
        )
        
        models[model_type] = trained_model
        test_loaders[model_type] = test_loader
    
    # Evaluate and compare all models
    results = evaluate_and_compare_models(models, test_loaders)
    
    # Create ensemble
    ensemble_preds, targets, ensemble_acc = ensemble_predictions(models, test_loaders)
    
    print("\n=== Final Results ===")
    for model_type in model_types:
        print(f"{model_type} Accuracy: {results[model_type]['accuracy']:.2f}%")
    print(f"Ensemble Accuracy: {ensemble_acc:.2f}%")

if __name__ == "__main__":
    main()