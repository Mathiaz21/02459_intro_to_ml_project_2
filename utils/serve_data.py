import pandas as pd
import numpy as np

X_DATA_PATH: str = "local_data/breast_cancer_wisconsin_features.csv"
Y_DATA_PATH: str = "local_data/breast_cancer_wisconsin_targets.csv"

def get_features_dataset() -> np.ndarray:
  return pd.read_csv(X_DATA_PATH).to_numpy()

def get_targets() -> np.ndarray:
  return np.ravel(pd.read_csv(Y_DATA_PATH))