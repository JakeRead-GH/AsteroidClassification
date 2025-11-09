import os
import pandas as pd

def preprocess_data(raw_path="data/raw/dataset.csv", save_path="data/preprocessed/asteroid_clean.csv"):
    """
    Preprocessing pipeline:
    - Drops identifier/metadata columns
    - Converts binary columns (pha, neo) to 0/1
    - Converts all remaining columns to numeric (scientific notation preserved)
    - Drops columns that cannot be converted to numeric
    - Fills missing values with column mean
    - Saves cleaned dataset
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(raw_path, low_memory=False)

    # Drop identifier/metadata columns
    drop_cols = ['id', 'spkid', 'full_name', 'pdes', 'name', 'prefix', 
                 'orbit_id', 'epoch_cal' , 'epoch' ,'epoch_mjd', 'equinox', 'tp_cal' 'class']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Convert binary categorical columns to 0/1
    if 'pha' in df.columns:
        df['pha'] = df['pha'].map({'N': 0, 'Y': 1})
    if 'neo' in df.columns:
        df['neo'] = df['neo'].map({'N': 0, 'Y': 1})

    # Convert everything else to numeric (scientific notation works)
    df = df.apply(pd.to_numeric, errors='coerce')


    # Fill remaining NaNs with column mean
    df = df.fillna(df.mean())

    # Save
    df.to_csv(save_path, index=False)
    print(f"Preprocessing finished. Dataset cleaned and saved to {save_path}")
    return save_path
