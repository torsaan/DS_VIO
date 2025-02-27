# hyperparameters.py
# Contains hyperparameters and grid search settings

DATA_DIR = './Data'
POSE_DIR = './Data/pose_keypoints'
BATCH_SIZE = 8
NUM_EPOCHS = 30

LEARNING_RATES = {
    'adam': 0.0001,
    'sgd': 0.001,
}

# Deep Learning model (LSTM) parameters
DL_MODEL_PARAMS = {
    'lstm_layers': 2,
    'hidden_size': 512,
    'dropout': 0.5,
    'activation_functions': ['relu', 'gelu'],  # You can switch between these
}

# Machine Learning model parameters (for grid search)
ML_MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
    },
    'svm': {
        'C': [0.1, 1.0, 10],
        'kernel': ['rbf', 'linear'],
    }
}

# Grid for hyperparameter tuning (example for DL model)
TUNING_GRID = {
    'learning_rate': [LEARNING_RATES['adam'], LEARNING_RATES['sgd']],
    'batch_size': [BATCH_SIZE, 16],
    'lstm_layers': [1, 2],
    'hidden_size': [256, 512],
}
