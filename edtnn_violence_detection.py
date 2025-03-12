#!/usr/bin/env python
# edtnn_violence_detection.py
"""
Implementation of Entanglement-Driven Topological Neural Network (ED-TNN)
for violence detection using the existing dataloader infrastructure.

This script implements a novel neural network architecture based on knot theory
and quantum entanglement concepts for violence detection in videos.

https://github.com/RichardAragon/QGLS

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from utils.dataprep import prepare_violence_nonviolence_data
from dataloader import get_dataloaders
from train import clear_cuda_memory
from evaluations import generate_metrics_report, plot_confusion_matrix


class TopologyGenerator:
    """Defines the 3D knot structure for the network topology."""
    
    def __init__(self, knot_type='trefoil', node_density=100, strand_count=3, braid_depth=4):
        """
        Initialize the topology generator.
        
        Args:
            knot_type (str): Type of knot to generate ('trefoil', 'figure-eight')
            node_density (int): Number of nodes per strand
            strand_count (int): Number of strands in the braid
            braid_depth (int): Complexity of the braid
        """
        self.knot_type = knot_type
        self.node_density = node_density
        self.strand_count = strand_count
        self.braid_depth = braid_depth
        
        # Generate the knot structure
        self.nodes, self.paths = self._generate_topology()
        
    def _generate_topology(self):
        """Generate the knot topology based on specified parameters."""
        if self.knot_type == 'trefoil':
            return self._generate_trefoil_knot()
        elif self.knot_type == 'figure-eight':
            return self._generate_figure_eight_knot()
        else:
            raise ValueError(f"Unsupported knot type: {self.knot_type}")
    
    def _generate_trefoil_knot(self):
        """Generate a trefoil knot topology."""
        nodes = []
        t_values = np.linspace(0, 2*np.pi, self.node_density)
        
        # Parametric equations for a trefoil knot
        for t in t_values:
            x = np.sin(t) + 2 * np.sin(2*t)
            y = np.cos(t) - 2 * np.cos(2*t)
            z = -np.sin(3*t)
            nodes.append(np.array([x, y, z]))
        
        # Define entangled paths (connections between nodes)
        paths = []
        for i in range(len(nodes)):
            # Connect each node to several others based on spatial proximity and braid logic
            connections = []
            for j in range(1, self.braid_depth + 1):
                next_idx = (i + j) % len(nodes)
                prev_idx = (i - j) % len(nodes)
                connections.extend([next_idx, prev_idx])
                
                # Add some cross-strand connections for more complex entanglement
                cross_idx = (i + len(nodes)//3) % len(nodes)
                connections.append(cross_idx)
            
            paths.append(connections)
        
        return np.array(nodes), paths
    
    def _generate_figure_eight_knot(self):
        """Generate a figure-eight knot topology."""
        nodes = []
        t_values = np.linspace(0, 2*np.pi, self.node_density)
        
        # Parametric equations for a figure-eight knot
        for t in t_values:
            x = (2 + np.cos(2*t)) * np.cos(3*t)
            y = (2 + np.cos(2*t)) * np.sin(3*t)
            z = np.sin(4*t)
            nodes.append(np.array([x, y, z]))
        
        # Define entangled paths similar to trefoil but with different crossings
        paths = []
        for i in range(len(nodes)):
            connections = []
            for j in range(1, self.braid_depth + 1):
                next_idx = (i + j) % len(nodes)
                prev_idx = (i - j) % len(nodes)
                connections.extend([next_idx, prev_idx])
                
                # Figure eight has more complex crossings
                cross_idx1 = (i + len(nodes)//4) % len(nodes)
                cross_idx2 = (i + len(nodes)//2) % len(nodes)
                connections.extend([cross_idx1, cross_idx2])
            
            paths.append(connections)
        
        return np.array(nodes), paths
    
    def visualize_topology(self):
        """Visualize the generated knot topology."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the knot
        nodes = np.array(self.nodes)
        ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], 'b-', lw=2)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='r', s=50)
        
        # Plot a subset of connections to avoid visual clutter
        for i in range(0, len(self.nodes), len(self.nodes)//20):
            for j in self.paths[i][:3]:  # Only show first 3 connections per node
                ax.plot([self.nodes[i][0], self.nodes[j][0]], 
                        [self.nodes[i][1], self.nodes[j][1]], 
                        [self.nodes[i][2], self.nodes[j][2]], 'g-', alpha=0.3)
        
        ax.set_title(f"{self.knot_type.capitalize()} Knot Topology")
        plt.tight_layout()
        return fig


class EntangledConnectionLayer(nn.Module):
    """
    Implements connections with entanglement coefficients and resonance phases.
    """
    
    def __init__(self, topology, in_features, out_features):
        """
        Initialize the entangled connection layer.
        
        Args:
            topology: The TopologyGenerator instance defining the structure
            in_features: Number of input features
            out_features: Number of output features
        """
        super(EntangledConnectionLayer, self).__init__()
        
        self.topology = topology
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weights
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Entanglement coefficients (ε)
        self.entanglement_coeff = nn.Parameter(torch.Tensor(len(topology.nodes), len(topology.nodes)))
        
        # Resonance phase (ϕ)
        self.resonance_phase = nn.Parameter(torch.Tensor(len(topology.nodes), len(topology.nodes)))
        
        # Knot tension (τ) - optional dynamic variable during training
        self.knot_tension = nn.Parameter(torch.Tensor(len(topology.nodes)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the layer parameters."""
        # Standard initialization for weights
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        
        # Initialize entanglement coefficients
        nn.init.uniform_(self.entanglement_coeff, 0.1, 0.5)
        
        # Initialize resonance phase (between 0 and 2π)
        nn.init.uniform_(self.resonance_phase, 0, 2 * np.pi)
        
        # Initialize knot tension
        nn.init.ones_(self.knot_tension)
    
    def entangled_connection_function(self, i, j, signal):
        """
        Compute the signal transmission between nodes i and j based on entangled structure.
        
        Args:
            i, j: Node indices
            signal: Input signal
            
        Returns:
            Modified signal after entanglement effects
        """
        # Get entanglement parameters for this connection
        epsilon = self.entanglement_coeff[i, j]
        phi = self.resonance_phase[i, j]
        tau = self.knot_tension[i] * self.knot_tension[j]
        
        # Apply entanglement effects (using complex-valued operations to model interference)
        phase_factor = torch.exp(1j * phi)
        entangled_signal = signal * (1 + epsilon * phase_factor) / (1 + tau)
        
        # Extract the real component (could also use amplitude)
        return torch.real(entangled_signal)
    
    def forward(self, x):
        """
        Forward pass through the entangled connection layer.
        
        Args:
            x: Input tensor of shape [batch_size, in_features]
            
        Returns:
            Output tensor after applying entangled connections
        """
        # Standard linear transformation
        output = F.linear(x, self.weights)
        
        # Apply entanglement effects
        batch_size = x.shape[0]
        entangled_output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # Map input and output to topology nodes
        in_per_node = max(1, self.in_features // len(self.topology.nodes))
        out_per_node = max(1, self.out_features // len(self.topology.nodes))
        
        # Apply entanglement effects between connected nodes
        for i in range(len(self.topology.nodes)):
            in_start = i * in_per_node
            in_end = min((i+1) * in_per_node, self.in_features)
            out_start = i * out_per_node
            out_end = min((i+1) * out_per_node, self.out_features)
            
            # Process connections for this node
            for j in self.topology.paths[i]:
                j_out_start = j * out_per_node
                j_out_end = min((j+1) * out_per_node, self.out_features)
                
                if out_end > out_start and j_out_end > j_out_start:
                    # Apply entanglement function to modify the signal
                    signal_ij = output[:, out_start:out_end]
                    entangled_signal = self.entangled_connection_function(i, j, signal_ij)
                    
                    # Add entangled contribution to output
                    j_width = min(j_out_end - j_out_start, entangled_signal.shape[1])
                    entangled_output[:, j_out_start:j_out_start+j_width] += entangled_signal[:, :j_width]
        
        # Combine standard output with entangled effects
        return output + 0.5 * entangled_output


class EntanglementPropagator(nn.Module):
    """
    Propagates information across entangled paths instead of layer-by-layer.
    """
    
    def __init__(self, topology, feature_dim):
        """
        Initialize the entanglement propagator.
        
        Args:
            topology: The TopologyGenerator instance
            feature_dim: Dimension of features at each node
        """
        super(EntanglementPropagator, self).__init__()
        
        self.topology = topology
        self.feature_dim = feature_dim
        
        # Propagation weights
        self.propagation_weights = nn.Parameter(
            torch.Tensor(len(topology.nodes), len(topology.nodes), feature_dim)
        )
        
        # Phase factors for wave-based propagation
        self.phase_factors = nn.Parameter(
            torch.Tensor(len(topology.nodes), len(topology.nodes))
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the propagator parameters."""
        # Initialize propagation weights
        nn.init.xavier_uniform_(self.propagation_weights)
        
        # Initialize phase factors (between 0 and 2π)
        nn.init.uniform_(self.phase_factors, 0, 2 * np.pi)
    
    def forward(self, node_features):
        """
        Forward pass through the entanglement propagator.
        
        Args:
            node_features: Features for each node [batch_size, num_nodes, feature_dim]
            
        Returns:
            Propagated features after wave-based interference
        """
        batch_size = node_features.shape[0]
        num_nodes = len(self.topology.nodes)
        device = node_features.device
        
        # Initialize output tensor
        propagated_features = torch.zeros(
            batch_size, num_nodes, self.feature_dim, device=device
        )
        
        # Wave-based propagation with interference
        for i in range(num_nodes):
            # Get connected nodes from topology
            connections = self.topology.paths[i]
            
            # Propagate signals along entangled paths
            for j in connections:
                # Apply phase factor for wave-like propagation
                phase = self.phase_factors[i, j]
                complex_phase = torch.exp(1j * phase)
                
                # Propagate signal with wave characteristics
                propagated_signal = node_features[:, i, :] * self.propagation_weights[i, j, :]
                propagated_signal = propagated_signal * complex_phase
                
                # Add to the destination node (interference happens naturally through addition)
                propagated_features[:, j, :] += torch.real(propagated_signal)
        
        # Normalize by the number of incoming connections
        connection_counts = torch.tensor(
            [len(self.topology.paths[i]) for i in range(num_nodes)],
            device=device
        ).float()
        normalization = torch.maximum(connection_counts, torch.ones_like(connection_counts))
        
        # Apply normalization across nodes
        propagated_features = propagated_features / normalization.view(1, -1, 1)
        
        return propagated_features


class CollapseResolutionLayer(nn.Module):
    """
    Interprets multi-path propagation into a singular signal for decision-making.
    """
    
    def __init__(self, topology, feature_dim, output_dim, collapse_method='entropy'):
        """
        Initialize the collapse resolution layer.
        
        Args:
            topology: The TopologyGenerator instance
            feature_dim: Dimension of features at each node
            output_dim: Dimension of the output after collapse
            collapse_method: Method to use for collapse ('entropy', 'energy', 'tension')
        """
        super(CollapseResolutionLayer, self).__init__()
        
        self.topology = topology
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.collapse_method = collapse_method
        
        # Collapse operator weights
        self.collapse_weights = nn.Parameter(
    torch.Tensor(output_dim, len(topology.nodes) * feature_dim)
)
        
        # Energy-based collapse parameters (if using energy method)
        if collapse_method == 'energy':
            self.energy_weights = nn.Parameter(
                torch.Tensor(len(topology.nodes))
            )
        
        # Tension-based collapse parameters (if using tension method)
        elif collapse_method == 'tension':
            self.tension_weights = nn.Parameter(
                torch.Tensor(len(topology.nodes))
            )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize the collapse layer parameters."""
        # Initialize collapse weights
        nn.init.xavier_uniform_(self.collapse_weights)
        
        # Initialize method-specific parameters
        if self.collapse_method == 'energy':
            nn.init.uniform_(self.energy_weights, 0.5, 1.5)
        elif self.collapse_method == 'tension':
            nn.init.uniform_(self.tension_weights, 0.5, 1.5)
    
    def forward(self, propagated_features):
        """
        Forward pass through the collapse resolution layer.
        
        Args:
            propagated_features: Features after propagation [batch_size, num_nodes, feature_dim]
            
        Returns:
            Collapsed output for decision-making
        """
        batch_size = propagated_features.shape[0]
        num_nodes = len(self.topology.nodes)
        
        # Apply collapse method to resolve superimposed states
        if self.collapse_method == 'entropy':
            # Entropy-based collapse: focus on most uncertain nodes
            node_entropy = -torch.sum(
                F.softmax(propagated_features, dim=2) * 
                F.log_softmax(propagated_features, dim=2),
                dim=2
            )
            collapse_weights = F.softmax(node_entropy, dim=1).unsqueeze(2)
            weighted_features = propagated_features * collapse_weights
            
        elif self.collapse_method == 'energy':
            # Energy-based collapse: weight by energy distribution
            node_energy = torch.sum(propagated_features**2, dim=2)
            energy_weights = F.softmax(node_energy * self.energy_weights, dim=1).unsqueeze(2)
            weighted_features = propagated_features * energy_weights
            
        elif self.collapse_method == 'tension':
            # Tension-based collapse: minimize topological strain
            tension_weights = F.softmax(self.tension_weights, dim=0).unsqueeze(0).unsqueeze(2)
            weighted_features = propagated_features * tension_weights
            
        else:
            # Default: equal weighting
            weighted_features = propagated_features / num_nodes
        
        # Flatten and project to output dimension
        collapsed_features = weighted_features.reshape(batch_size, -1)
        output = F.linear(collapsed_features, self.collapse_weights)
        
        return output


class VideoFeatureExtractor(nn.Module):
    """
    Extracts features from video frames for use in the EDTNN model.
    Uses a 3D CNN backbone to extract spatiotemporal features.
    """
    
    def __init__(self, output_dim=1024, pretrained=True):
        """
        Initialize the feature extractor.
        
        Args:
            output_dim: Dimension of output features
            pretrained: Whether to use pretrained backbone
        """
        super(VideoFeatureExtractor, self).__init__()
        
        # Use a 3D CNN backbone (R3D_18)
        from torchvision.models.video import r3d_18
        self.backbone = r3d_18(pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace final layer with a projection to desired output dimension
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.feature_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Extract features from video frames.
        
        Args:
            x: Video frames tensor [batch_size, channels, frames, height, width]
               or [batch_size, frames, channels, height, width]
            
        Returns:
            Extracted features [batch_size, output_dim]
        """
        # Ensure input is in format [B, C, T, H, W]
        if x.dim() == 5 and x.shape[1] != 3:
            # Input is [B, T, C, H, W], permute to [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4)
        
        # Extract features using backbone
        features = self.backbone(x)
        
        return features


class PoseFeatureExtractor(nn.Module):
    """
    Extracts features from pose keypoints for use in the EDTNN model.
    """
    
    def __init__(self, input_dim=66, output_dim=256):
        """
        Initialize the pose feature extractor.
        
        Args:
            input_dim: Dimension of input pose keypoints
            output_dim: Dimension of output features
        """
        super(PoseFeatureExtractor, self).__init__()
        
        # Temporal convolutional network for pose processing
        self.pose_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Extract features from pose keypoints.
        
        Args:
            x: Pose keypoints tensor [batch_size, frames, keypoints]
            
        Returns:
            Extracted features [batch_size, output_dim]
        """
        batch_size, seq_length, pose_dim = x.shape
        
        # Process each frame
        x = x.reshape(batch_size * seq_length, pose_dim)
        x = self.pose_encoder(x)
        
        # Reshape back and average across time
        x = x.reshape(batch_size, seq_length, -1)
        x = torch.mean(x, dim=1)
        
        return x


class EDTNN_ViolenceDetection(nn.Module):
    """
    Entanglement-Driven Topological Neural Network for violence detection.
    """
    
    def __init__(self, num_classes=2, knot_type='trefoil', node_density=64, 
                 features_per_node=16, collapse_method='entropy', use_pose=False):
        """
        Initialize the ED-TNN model for violence detection.
        
        Args:
            num_classes: Number of output classes (2 for violence detection)
            knot_type: Type of knot topology ('trefoil', 'figure-eight')
            node_density: Number of nodes in the topology
            features_per_node: Number of features per node
            collapse_method: Method for the collapse layer
            use_pose: Whether to use pose data in addition to video
        """
        super(EDTNN_ViolenceDetection, self).__init__()
        
        # Generate the topology
        self.topology = TopologyGenerator(
            knot_type=knot_type,
            node_density=node_density,
            strand_count=3,
            braid_depth=4
        )
        
        # Dimensions
        self.num_classes = num_classes
        self.features_per_node = features_per_node
        self.use_pose = use_pose
        
        # Feature extractors
        self.video_extractor = VideoFeatureExtractor(
            output_dim=node_density * features_per_node
        )
        
        if use_pose:
            self.pose_extractor = PoseFeatureExtractor(
                input_dim=66,  # Assuming 33 keypoints x 2 (x,y)
                output_dim=node_density * features_per_node // 2
            )
            
            # Combined feature dimension
            combined_dim = node_density * features_per_node + node_density * features_per_node // 2
        else:
            combined_dim = node_density * features_per_node
        
        # ED-TNN specific layers
        self.entangled_layer = EntangledConnectionLayer(
            self.topology, 
            combined_dim, 
            node_density * features_per_node
        )
        
        self.propagator = EntanglementPropagator(
            self.topology,
            features_per_node
        )
        
        self.collapse_layer = CollapseResolutionLayer(
            self.topology,
            features_per_node,
            num_classes,
            collapse_method=collapse_method
        )
    
    def forward(self, inputs):
        """
        Forward pass through the ED-TNN model.
        
        Args:
            inputs: Input data (video frames) or tuple of (video frames, pose keypoints)
            
        Returns:
            Output predictions
        """
        if self.use_pose:
            # Unpack inputs
            video_frames, pose_keypoints = inputs
            
            # Extract video features
            video_features = self.video_extractor(video_frames)
            
            # Extract pose features
            pose_features = self.pose_extractor(pose_keypoints)
            
            # Combine features
            combined_features = torch.cat([video_features, pose_features], dim=1)
            
            # Apply entangled connections
            x = self.entangled_layer(combined_features)
        else:
            # Process only video frames
            video_frames = inputs
            
            # Extract video features
            video_features = self.video_extractor(video_frames)
            
            # Apply entangled connections
            x = self.entangled_layer(video_features)
        
        # Reshape for propagator
        batch_size = x.shape[0]
        x = x.view(batch_size, len(self.topology.nodes), self.features_per_node)
        
        # Apply entanglement propagation
        x = self.propagator(x)
        
        # Apply collapse resolution
        x = self.collapse_layer(x)
        
        return x


class ResonanceLoss(nn.Module):
    """
    Custom loss function that includes a resonance loss component.
    """
    
    def __init__(self, topology, base_criterion=nn.CrossEntropyLoss(), resonance_weight=0.1):
        """
        Initialize the resonance loss.
        
        Args:
            topology: The TopologyGenerator instance
            base_criterion: Base loss criterion (e.g., CrossEntropyLoss)
            resonance_weight: Weight for the resonance component
        """
        super(ResonanceLoss, self).__init__()
        
        self.topology = topology
        self.base_criterion = base_criterion
        self.resonance_weight = resonance_weight
    
    def forward(self, outputs, targets, entanglement_layer):
        """
        Compute the loss with resonance component.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            entanglement_layer: The EntangledConnectionLayer instance
            
        Returns:
            Combined loss value
        """
        # Base loss (e.g., cross-entropy)
        base_loss = self.base_criterion(outputs, targets)
        
        # Resonance loss component: penalize disharmony in signal propagation
        resonance_loss = 0.0
        
        # Compute phase disharmony across connections
        for i in range(len(self.topology.nodes)):
            for j in self.topology.paths[i]:
                # Phase difference between connected nodes
                phase_i = entanglement_layer.resonance_phase[i, j]
                phase_j = entanglement_layer.resonance_phase[j, i]
                
                # Penalize large phase differences (disharmony)
                phase_diff = torch.abs(phase_i - phase_j) % (2 * np.pi)
                if phase_diff > np.pi:
                    phase_diff = 2 * np.pi - phase_diff
                    
                resonance_loss += phase_diff
        
        # Normalize by number of connections
        total_connections = sum(len(paths) for paths in self.topology.paths)
        resonance_loss = resonance_loss / total_connections
        
        # Combine losses
        total_loss = base_loss + self.resonance_weight * resonance_loss
        
        return total_loss


def evaluate_edtnn(model, test_loader, device, output_dir="./output"):
    """
    Evaluate the ED-TNN model on test data.
    
    Args:
        model: The trained ED-TNN model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Handle different input types (with or without pose data)
            if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            else:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    report, cm = generate_metrics_report(
        all_preds, all_targets,
        output_path=os.path.join(output_dir, 'edtnn_metrics.json')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        output_path=os.path.join(output_dir, 'edtnn_confusion_matrix.png')
    )
    
    # Calculate accuracy
    accuracy = 100. * (np.array(all_preds) == np.array(all_targets)).mean()
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'report': report,
        'confusion_matrix': cm
    }


def train_edtnn(model, train_loader, val_loader, device, num_epochs=30, 
                learning_rate=0.0001, resonance_weight=0.1, output_dir="./output"):
    """
    Train the ED-TNN model for violence detection.
    
    Args:
        model: The ED-TNN model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (cpu or cuda)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        resonance_weight: Weight for resonance loss component
        output_dir: Directory to save model checkpoints and results
        
    Returns:
        Trained model and training history
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loss function with resonance component
    criterion = ResonanceLoss(
        model.topology,
        base_criterion=nn.CrossEntropyLoss(),
        resonance_weight=resonance_weight
    )
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Handle different input types (with or without pose data)
            if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                frames, pose, targets = batch
                frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                inputs = (frames, pose)
            else:  # Video + Label
                frames, targets = batch
                frames, targets = frames.to(device), targets.to(device)
                inputs = frames
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets, model.entangled_layer)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update statistics
            train_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / total
        epoch_train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different input types (with or without pose data)
                if model.use_pose and len(batch) == 3:  # Video + Pose + Label
                    frames, pose, targets = batch
                    frames, pose, targets = frames.to(device), pose.to(device), targets.to(device)
                    inputs = (frames, pose)
                else:  # Video + Label
                    frames, targets = batch
                    frames, targets = frames.to(device), targets.to(device)
                    inputs = frames
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, targets, model.entangled_layer)
                
                # Update statistics
                val_loss += loss.item() * targets.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        epoch_val_loss = val_loss / total
        epoch_val_acc = 100. * correct / total
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print epoch summary
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save checkpoint if this is the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            checkpoint_path = os.path.join(output_dir, "edtnn_best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_val_acc,
                'topology_type': model.topology.knot_type,
                'node_density': model.topology.node_density
            }, checkpoint_path)
            print(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")
        
        # Save topology visualization at specific epochs
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            fig = model.topology.visualize_topology()
            fig.savefig(os.path.join(output_dir, f"topology_epoch_{epoch+1}.png"))
            plt.close(fig)
            
            # Visualize node tensions
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            node_positions = np.array(model.topology.nodes)
            tensions = model.entangled_layer.knot_tension.detach().cpu().numpy()
            scatter = ax.scatter(
                node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
                c=tensions, cmap='viridis', s=60
            )
            ax.set_title(f"Node Tensions at Epoch {epoch+1}")
            fig.colorbar(scatter, ax=ax, label="Tension")
            fig.savefig(os.path.join(output_dir, f"tensions_epoch_{epoch+1}.png"))
            plt.close(fig)
            
        # Clear CUDA cache
        clear_cuda_memory()
    
    # Load the best model for return
    checkpoint = torch.load(os.path.join(output_dir, "edtnn_best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()
    
    return model, history


def main():
    """
    Main function to train and evaluate the ED-TNN model for violence detection.
    """
    parser = argparse.ArgumentParser(description="Train ED-TNN for violence detection")
    parser.add_argument("--data_dir", type=str, default="./Data/Processed/standardized",
                      help="Directory containing the violence detection dataset")
    parser.add_argument("--pose_dir", type=str, default=None,
                      help="Directory containing pose keypoints (optional)")
    parser.add_argument("--output_dir", type=str, default="./output/edtnn",
                      help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                      help="Learning rate for optimizer")
    parser.add_argument("--gpu", type=int, default=0,
                      help="GPU ID to use (-1 for CPU)")
    parser.add_argument("--knot_type", type=str, default="trefoil",
                      help="Type of knot topology (trefoil or figure-eight)")
    parser.add_argument("--node_density", type=int, default=64,
                      help="Number of nodes in the knot topology")
    parser.add_argument("--features_per_node", type=int, default=16,
                      help="Number of features per node")
    parser.add_argument("--collapse_method", type=str, default="entropy",
                      help="Method for collapse layer (entropy, energy, or tension)")
    parser.add_argument("--resonance_weight", type=float, default=0.1,
                      help="Weight for resonance loss component")
    parser.add_argument("--use_pose", action="store_true",
                      help="Use pose data in addition to video frames")
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
        prepare_violence_nonviolence_data(args.data_dir)
    
    # Create dataloaders
    pose_dir = args.pose_dir if args.use_pose else None
    train_loader, val_loader, test_loader = get_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        pose_dir=pose_dir,
        batch_size=args.batch_size,
        model_type='3d_cnn'  # Using 3D CNN format for the frames
    )
    
    # Initialize ED-TNN model
    print(f"Initializing ED-TNN model with {args.knot_type} knot topology...")
    model = EDTNN_ViolenceDetection(
        num_classes=2,
        knot_type=args.knot_type,
        node_density=args.node_density,
        features_per_node=args.features_per_node,
        collapse_method=args.collapse_method,
        use_pose=args.use_pose
    ).to(device)
    
    # Display model structure
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Visualize initial topology
    fig = model.topology.visualize_topology()
    fig.savefig(os.path.join(args.output_dir, "initial_topology.png"))
    plt.close(fig)
    
    # Train the model
    print("\nTraining ED-TNN model...")
    trained_model, history = train_edtnn(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        resonance_weight=args.resonance_weight,
        output_dir=args.output_dir
    )
    
    # Evaluate the model
    print("\nEvaluating ED-TNN model...")
    evaluation_results = evaluate_edtnn(
        trained_model,
        test_loader,
        device,
        output_dir=args.output_dir
    )
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Test accuracy: {evaluation_results['accuracy']:.2f}%")
    
    # Save final results summary
    results_summary = {
        'test_accuracy': evaluation_results['accuracy'],
        'best_val_accuracy': max(history['val_acc']),
        'model_parameters': {
            'knot_type': args.knot_type,
            'node_density': args.node_density,
            'features_per_node': args.features_per_node,
            'collapse_method': args.collapse_method,
            'use_pose': args.use_pose
        }
    }
    
    with open(os.path.join(args.output_dir, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=4)


if __name__ == "__main__":
    main()