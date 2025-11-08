import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import os

def get_model(loss='log_loss', learning_rate=0.1, max_iter=100, max_depth=3, random_state=42):
    """
    Initialize and return a HistGradientBoostingClassifier with specified hyperparameters.

    Relevant Loss options:
    - 'log_loss': Logistic loss for classification.
    - 'auto': Automatically selects the loss function based on the data.
    - 'binary_crossentropy': Binary cross-entropy loss for binary classification.
    """

    return HistGradientBoostingClassifier(
        loss=loss,
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        random_state=random_state
    )


def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV to find the best parameters
    for the HistGradientBoostingClassifier. Returns the best parameters found.
    Randomized search is superior to grid search in all aspects, including speed and performance.

    Tuned Hyperparameters:
    - learning_rate: Controls the contribution of each tree to the final model.
    - max_depth: Maximum depth of the individual trees.
    - max_iter: Number of boosting iterations (trees).
    """

    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'max_iter': [50, 100, 150, 200]
    }

    model = get_model()

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=10,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_params_


def run_main_model():
    """
    Takes in
    """
    raw_file = 'data/raw/dataset.csv'
    preprocessed_file = 'data/preprocessed/asteroid_clean.csv'

    # Load preprocessed dataset if exists
    if os.path.exists(preprocessed_file):
        print(f"Loaded preprocessed dataset: {preprocessed_file}")
        df = pd.read_csv(preprocessed_file, low_memory=False)
    else:
        print(f"Preprocessed file not found. Loading raw dataset: {raw_file}")
        df = pd.read_csv(raw_file, low_memory=False)

        # Drop non-numeric columns
        non_numeric_cols = [
            'id', 'spkid', 'full_name', 'pdes', 'name', 'prefix',
            'orbit_id', 'epoch_cal', 'equinox', 'tp_cal', 'class'
        ]
        df.drop(columns=[col for col in non_numeric_cols if col in df.columns], inplace=True)

        # Convert categorical columns to numeric
        if 'neo' in df.columns:
            df['neo'] = df['neo'].map({'N': 0, 'Y': 1})
        if 'pha' in df.columns:
            df['pha'] = df['pha'].map({'N': 0, 'Y': 1})

        # Force everything else to numeric, strings become NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # Fill missing values
        df = df.fillna(df.mean())

        # Save preprocessed dataset for future runs
        df.to_csv(preprocessed_file, index=False)
        print(f"Saved preprocessed dataset: {preprocessed_file}")

    # Features and target
    if 'pha' not in df.columns:
        raise ValueError("Target column 'pha' not found in dataset.")
    X = df.drop('pha', axis=1)
    y = df['pha'].fillna(0).astype(int)  # <<< Force discrete 0/1

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.01, random_state=42, stratify=y
    )

    # Train Gradient Boosting model
    model = HistGradientBoostingClassifier(
        max_iter=100,      
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    print("Training Gradient Boosting model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nGradient Boosting Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    run_main_model()
