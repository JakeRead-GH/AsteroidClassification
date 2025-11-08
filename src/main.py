from models.baseline import run_baseline
from models.main_model import run_main_model

if __name__ == "__main__":
    # Preprocess (returns X_train, y_train, X_test, y_test)

    # Train models and save outputs (takes in X_train, y_train)

    # Run models (run on X_test, returns y_pred)
    run_baseline()
    run_main_model()

    # Evaluate results (takes in y_test, y_pred)
