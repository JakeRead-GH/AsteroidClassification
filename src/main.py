import sys,os
import argparse
import kagglehub
from models.baseline import run_baseline
from models.main_model import MainModel
from evaluation.eval import run_evaluation


"""
Main execution script for training, tuning, and evaluating models.

Usage:
    python src/main.py [options]
    
Options:
    -h, --hyperparameter_tune : Perform hyperparameter tuning before training the model.
    -ms, --model_save_path : Path to save the trained model. Default is 'trained_model.joblib'.
    -mp, --model_load_path : Path to load the model from. Default is 'trained_model.joblib'.
"""

# TODO : Add more command line arguments for different functionalities
def get_args():
    parser = argparse.ArgumentParser(
        description="Train, tune, and evaluate the asteroid classification model."
    )

    # Optional flag: hyperparameter tuning
    parser.add_argument(
        '--hyperparameter_tune',  # no short '-h', to avoid clash with help
        action='store_true',
        help='Perform hyperparameter tuning before training the model.'
    )

    # Optional: paths for saving/loading model
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='trained_model.joblib',
        help='Path to save the trained model.'
    )
    parser.add_argument(
        '--model_load_path',
        type=str,
        default='trained_model.joblib',
        help='Path to load the model from.'
    )

    return parser.parse_args()

def download_dataset():
    """Download dataset from Kaggle if not already in data/raw."""
    raw_path = "data/raw/dataset.csv"
    if not os.path.exists(raw_path):
        print("Downloading dataset from Kaggle...")
        dataset_dir = kagglehub.dataset_download("sakhawat18/asteroid-dataset")
        source_path = os.path.join(dataset_dir, "dataset.csv")
        os.makedirs("data/raw", exist_ok=True)
        os.system(f"cp '{source_path}' '{raw_path}'")
        print(f"Dataset saved to {raw_path}")
    else:
        print("Raw dataset already exists.")
    return raw_path

if __name__ == "__main__":
    


    download_dataset()


    #JAKE
    '''
    args = get_args()
    # Preprocess (returns X_train, y_train, X_test, y_test)
    # TODO : implement data loading and preprocessing
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    # Train models and save outputs (takes in X_train, y_train)
    mm = MainModel()

    # Hyperparameter tuning if asked for in args
    if args.hyperparameter_tune:
        mm.hyperparameter_tuning(X_train, y_train)

    # Get model (load from file if exists, else create new)
    model, is_loaded = mm.get_model(load_path=args.model_load_path)

    # Trains the model (if not loaded from file) and saves it
    if not is_loaded:
        model = mm.train(model, X_train, y_train, save_path=args.model_save_path)

    # Run models (run on X_test, returns y_pred)
    y_pred = mm.predict(model, X_test)

    # Evaluate results (takes in y_test, y_pred)
    run_evaluation(y_test, y_pred)'''
