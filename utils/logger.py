# utils/logger.py
import csv
import os
import logging
from datetime import datetime

class CSVLogger:
    def __init__(self, filepath, fieldnames):
        self.filepath = filepath
        self.fieldnames = fieldnames
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Initialize CSV file with header if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, data_dict):
        # Add timestamp if not present
        if 'timestamp' not in data_dict and 'timestamp' in self.fieldnames:
            data_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        with open(self.filepath, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(data_dict)


class Logger:
    """Enhanced logger with console and file output"""
    
    def __init__(self, name, log_dir="./logs", level=logging.INFO):
        self.name = name
        self.log_dir = log_dir
        
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)