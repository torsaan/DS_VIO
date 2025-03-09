# Utils/logger.py
import csv
import os
import time
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch


class CSVLogger:
    """
    Enhanced CSV Logger for training metrics with timestamp support and visualization capabilities.
    """
    def __init__(self, filepath, fieldnames, append=False):
        """
        Initialize the CSV Logger.
        
        Args:
            filepath: Path to the CSV file
            fieldnames: List of column names
            append: Whether to append to existing file (if it exists)
        """
        self.filepath = filepath
        self.fieldnames = fieldnames
        
        # Ensure timestamp is included
        if 'timestamp' not in self.fieldnames:
            self.fieldnames = ['timestamp'] + self.fieldnames
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create or append to file
        file_exists = os.path.exists(filepath)
        
        if file_exists and append:
            # Check if existing headers match
            with open(filepath, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                existing_headers = next(reader, None)
                
                if existing_headers and set(existing_headers) != set(self.fieldnames):
                    # Headers don't match, backup existing file and start new one
                    backup_path = f"{filepath}.bak_{int(time.time())}"
                    os.rename(filepath, backup_path)
                    print(f"Headers don't match. Existing file backed up to {backup_path}")
                    file_exists = False
        
        if not file_exists or not append:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
        
        # Store history for plotting
        self.history = {field: [] for field in self.fieldnames if field != 'timestamp'}
        self.epochs = []
        
        # Load existing data if appending
        if file_exists and append:
            self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing data from CSV file for plotting"""
        try:
            with open(self.filepath, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if 'epoch' in row:
                        self.epochs.append(float(row['epoch']))
                    
                    for field in self.fieldnames:
                        if field != 'timestamp' and field in row:
                            try:
                                self.history[field].append(float(row[field]))
                            except (ValueError, TypeError):
                                # Skip non-numeric values
                                pass
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
    
    def log(self, data_dict):
        """
        Log a row of data to the CSV file.
        
        Args:
            data_dict: Dictionary containing data to log
        """
        # Add timestamp
        row_dict = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        row_dict.update(data_dict)
        
        # Write to CSV
        with open(self.filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row_dict)
        
        # Update history for plotting
        if 'epoch' in data_dict:
            self.epochs.append(float(data_dict['epoch']))
            
        for field in self.fieldnames:
            if field != 'timestamp' and field in data_dict:
                try:
                    self.history[field].append(float(data_dict[field]))
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    pass
    
    def plot(self, output_dir=None):
        """
        Plot training metrics.
        
        Args:
            output_dir: Directory to save plots (if None, will save in same directory as CSV)
        """
        if not self.epochs:
            print("No data to plot")
            return
        
        if output_dir is None:
            output_dir = os.path.dirname(self.filepath)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Group related metrics
        loss_metrics = [f for f in self.history.keys() if 'loss' in f.lower()]
        acc_metrics = [f for f in self.history.keys() if 'acc' in f.lower() or 'accuracy' in f.lower()]
        other_metrics = [f for f in self.history.keys() if f not in loss_metrics and f not in acc_metrics]
        
        # Plot loss metrics
        if loss_metrics:
            plt.figure(figsize=(10, 6))
            for metric in loss_metrics:
                if self.history[metric]:
                    plt.plot(self.epochs, self.history[metric], label=metric)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
            plt.close()
        
        # Plot accuracy metrics
        if acc_metrics:
            plt.figure(figsize=(10, 6))
            for metric in acc_metrics:
                if self.history[metric]:
                    plt.plot(self.epochs, self.history[metric], label=metric)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
            plt.close()
        
        # Plot other metrics
        if other_metrics:
            plt.figure(figsize=(10, 6))
            for metric in other_metrics:
                if self.history[metric]:
                    plt.plot(self.epochs, self.history[metric], label=metric)
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Other Metrics')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'other_metrics_plot.png'))
            plt.close()


class TrainingLogger:
    """
    Comprehensive training logger that combines CSV logging, TensorBoard, and model checkpoints.
    """
    def __init__(self, log_dir, model_name, fields_to_track, use_tensorboard=True):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Base directory for logs
            model_name: Name of the model being trained
            fields_to_track: List of metrics to track
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir) / model_name
        self.model_name = model_name
        self.fields_to_track = fields_to_track
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up CSV logger
        self.csv_logger = CSVLogger(
            filepath=str(self.log_dir / f"{model_name}_log.csv"),
            fieldnames=fields_to_track,
            append=True
        )
        
        # Set up TensorBoard if requested
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
        
        # Keep track of best metric values
        self.best_metrics = {}
        
        # Config for this training run
        self.config = {
            'model_name': model_name,
            'start_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fields_tracked': fields_to_track
        }
        
        # Save config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def log_epoch(self, epoch_metrics):
        """
        Log metrics for an epoch.
        
        Args:
            epoch_metrics: Dictionary containing epoch metrics
        """
        # Log to CSV
        self.csv_logger.log(epoch_metrics)
        
        # Log to TensorBoard
        if self.writer:
            for name, value in epoch_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, epoch_metrics.get('epoch', 0))
        
        # Update best metrics
        for metric, value in epoch_metrics.items():
            if isinstance(value, (int, float)) and (metric not in self.best_metrics or 
                                                  ('loss' in metric.lower() and value < self.best_metrics[metric]) or
                                                  ('acc' in metric.lower() and value > self.best_metrics[metric])):
                self.best_metrics[metric] = value
        
        # Generate plots
        self.csv_logger.plot(str(self.log_dir))
    
    def log_batch(self, batch_metrics, global_step):
        """
        Log metrics for a batch.
        
        Args:
            batch_metrics: Dictionary containing batch metrics
            global_step: Global step (batch) number
        """
        # Only log to TensorBoard (batch logging to CSV would be too verbose)
        if self.writer:
            for name, value in batch_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"batch/{name}", value, global_step)
    
    def log_model_graph(self, model, dummy_input):
        """
        Log model graph to TensorBoard.
        
        Args:
            model: PyTorch model
            dummy_input: Dummy input tensor for tracing the model graph
        """
        if self.writer:
            try:
                self.writer.add_graph(model, dummy_input.to(next(model.parameters()).device))
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def log_confusion_matrix(self, cm, class_names, epoch):
        """
        Log confusion matrix to TensorBoard.
        
        Args:
            cm: Confusion matrix as numpy array
            class_names: List of class names
            epoch: Current epoch
        """
        if self.writer:
            try:
                import pandas as pd
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    pd.DataFrame(cm, index=class_names, columns=class_names),
                    annot=True, fmt='d', cmap='Blues', ax=ax
                )
                plt.ylabel('True')
                plt.xlabel('Predicted')
                plt.title('Confusion Matrix')
                
                self.writer.add_figure(f"confusion_matrix/epoch_{epoch}", fig, epoch)
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not log confusion matrix: {e}")
    
    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            state: Dictionary containing model state
            is_best: Whether this is the best model so far
            filename: Filename for the checkpoint
        """
        # Save checkpoint
        checkpoint_path = self.log_dir / filename
        torch.save(state, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.log_dir / f"best_{self.model_name}_model.pth"
            # Make a copy of the best model
            import shutil
            shutil.copyfile(checkpoint_path, best_path)
    
    def log_hyperparameters(self, hyperparams):
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Update config
        self.config['hyperparameters'] = hyperparams
        
        # Save updated config
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Log to TensorBoard
        if self.writer:
            try:
                # Log hyperparameters along with a performance metric if available
                if 'val_acc' in self.best_metrics:
                    metric_dict = {'best_val_accuracy': self.best_metrics['val_acc']}
                elif 'val_loss' in self.best_metrics:
                    metric_dict = {'best_val_loss': self.best_metrics['val_loss']}
                else:
                    metric_dict = {}
                
                self.writer.add_hparams(hyperparams, metric_dict)
            except Exception as e:
                print(f"Warning: Could not log hyperparameters: {e}")
    
    def close(self):
        """Close the logger, ensuring all data is saved."""
        if self.writer:
            self.writer.close()
        
        # Update end time in config
        self.config['end_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Generate final plots
        self.csv_logger.plot(str(self.log_dir))
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a summary report of the training run."""
        summary = {
            'model_name': self.model_name,
            'training_duration': self._calculate_duration(),
            'best_metrics': self.best_metrics,
            'final_epoch': len(self.csv_logger.epochs),
        }
        
        # Write summary to file
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create a markdown report
        markdown = f"# Training Summary for {self.model_name}\n\n"
        markdown += f"* **Start Time:** {self.config['start_time']}\n"
        markdown += f"* **End Time:** {self.config.get('end_time', 'N/A')}\n"
        markdown += f"* **Duration:** {self._calculate_duration()}\n"
        markdown += f"* **Total Epochs:** {len(self.csv_logger.epochs)}\n\n"
        
        markdown += "## Best Metrics\n\n"
        for metric, value in self.best_metrics.items():
            markdown += f"* **{metric}:** {value}\n"
        
        if 'hyperparameters' in self.config:
            markdown += "\n## Hyperparameters\n\n"
            for param, value in self.config['hyperparameters'].items():
                markdown += f"* **{param}:** {value}\n"
        
        # Write markdown report
        with open(self.log_dir / 'training_summary.md', 'w') as f:
            f.write(markdown)
    
    def _calculate_duration(self):
        """Calculate the duration of the training run."""
        if 'start_time' not in self.config:
            return "Unknown"
        
        start_time = datetime.datetime.strptime(self.config['start_time'], '%Y-%m-%d %H:%M:%S')
        
        if 'end_time' in self.config:
            end_time = datetime.datetime.strptime(self.config['end_time'], '%Y-%m-%d %H:%M:%S')
        else:
            end_time = datetime.datetime.now()
        
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"


# Example usage:
"""
# Basic CSV Logger
logger = CSVLogger('output/model1/training_log.csv', ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
logger.log({'epoch': 1, 'train_loss': 0.5, 'train_acc': 75.0, 'val_loss': 0.6, 'val_acc': 70.0})

# Comprehensive Training Logger
training_logger = TrainingLogger(
    log_dir='output',
    model_name='model1',
    fields_to_track=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'learning_rate']
)

# Log hyperparameters
training_logger.log_hyperparameters({
    'learning_rate': 0.001,
    'batch_size': 32,
    'optimizer': 'adam'
})

# Log each epoch
for epoch in range(1, 11):
    # Training logic here...
    
    # Log epoch metrics
    training_logger.log_epoch({
        'epoch': epoch,
        'train_loss': 0.5 - 0.04 * epoch,
        'train_acc': 75.0 + 2 * epoch,
        'val_loss': 0.6 - 0.03 * epoch,
        'val_acc': 70.0 + 1.5 * epoch,
        'learning_rate': 0.001 * (0.9 ** epoch)
    })
    
    # Save checkpoint
    training_logger.save_checkpoint(
        state={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        is_best=(epoch == 10)  # Just for example
    )

# Close logger when done
training_logger.close()
"""