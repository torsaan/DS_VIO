import sys
from main import main

if __name__ == "__main__":
    # Set up dummy sys.argv if necessary
    sys.argv = [
        "main.py",
        "--data_dir", "./Data/Processed/standardized",      # Use your dummy data directory or real one
        "--output_dir", "./dummy_output",
        "--batch_size", "2",
        "--num_epochs", "2",
        "--num_workers", "0",
        "--hp_search",  # if you want to test hyperparameter search
        "--resume",     # if testing resume functionality
        "--grad_clip", "1.0",
        "--model_types", "3d_cnn"  # For example, test with just one model
    ]
    main()
