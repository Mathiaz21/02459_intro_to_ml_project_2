import numpy as np
import os
import utils.save_remote_data as srd

from pandas import read_csv


X_DATA_PATH: str = "local_data/breast_cancer_wisconsin_features.csv"
Y_DATA_PATH: str = "local_data/breast_cancer_wisconsin_targets.csv"

def verify_data_existence() -> None:
  if not (os.path.exists(X_DATA_PATH) and os.path.exists(Y_DATA_PATH)):
    print("Data not saved locally, importing it")
    srd.save_data()

def get_features_dataset() -> np.ndarray:
  verify_data_existence()
  return read_csv(X_DATA_PATH).to_numpy()

def get_targets() -> np.ndarray:
  verify_data_existence()
  return np.ravel(read_csv(Y_DATA_PATH))