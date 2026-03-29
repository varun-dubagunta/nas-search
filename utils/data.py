"""
Data loading and preprocessing for Higgs Discovery NAS.
Loads DNN_samples_v4.pkl and returns train/val numpy arrays.
Label mapping is built dynamically from the 'process' column.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_COLUMNS = [
    'mu1_eta', 'mu1_pt', 'mu2_eta', 'mu2_pt',
    'dR_mumu', 'm_mumu', 'eta_mumu', 'cosTheta_CS', 'phi_mumu',
    'pt_mumu', 'y_mumu', 'phi_CS', 'R_pt',
    'minDeltaEtaSigned', 'minDeltaPhi',
    'Zepperfield_Var', 'pt_centrality',
    'j1_eta', 'j1_pt', 'j1_btagPNetQvG',
    'j2_eta', 'j2_pt', 'j2_btagPNetQvG',
    'm_jj', 'delta_eta_jj', 'pt_jj',
    'nJet', 'nSoftActivityJet',
    'SoftActivityJetHT', 'SoftActivityJetHT2', 'SoftActivityJetHT5',
    'SoftActivityJetHT10', 'SoftActivityJetNjets2',
    'SoftActivityJetNjets5', 'SoftActivityJetNjets10'
]


def load_data(pkl_path: str, val_size: float = 0.2, seed: int = 42):
    """
    Load and preprocess the Higgs dataset.
    Label map is built from whatever unique values exist in 'process' column.

    Returns:
        X_train, X_val : np.ndarray — scaled feature matrices
        y_train, y_val : np.ndarray — integer class labels
        scaler         : fitted StandardScaler
        class_names    : list of class name strings (index = class int)
    """
    print(f"Loading {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)

    print(f"Loaded dataframe: {df.shape}")

    # Build label map dynamically from process column
    if 'Label' in df.columns:
        # If Label already exists as int, map back via process for class names
        class_names = sorted(df['process'].unique().tolist())
        label_map   = {name: i for i, name in enumerate(class_names)}
        df['_label'] = df['process'].map(label_map)
    else:
        class_names = sorted(df['process'].unique().tolist())
        label_map   = {name: i for i, name in enumerate(class_names)}
        df['_label'] = df['process'].map(label_map)

    print(f"Label map: {label_map}")

    # Drop rows with missing features or labels
    df = df.dropna(subset=DATA_COLUMNS + ['_label'])

    X = df[DATA_COLUMNS].values.astype(np.float32)
    y = df['_label'].astype(int).values

    print(f"Class distribution:")
    for name, idx in label_map.items():
        count = (y == idx).sum()
        print(f"  {name} ({idx}): {count:,} ({count/len(y)*100:.1f}%)")

    # Train/val split — stratified to preserve class balance
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=seed,
        stratify=y
    )

    # Scale — fit on train only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)

    print(f"Train: {X_train.shape} | Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val, scaler, class_names