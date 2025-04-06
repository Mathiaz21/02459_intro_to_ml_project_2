import pandas as pd

def standardize_data(X: pd.DataFrame) -> pd.DataFrame:
  return (X - X.mean(axis=0)) / X.std(axis=0)