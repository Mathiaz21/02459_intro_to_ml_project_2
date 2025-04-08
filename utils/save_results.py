import pandas as pd
import os

def save_results(results_df: pd.DataFrame, output_name: str, output_path: str = "results/") -> None:
  if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created {output_path} folder.")
  if not output_name.endswith(".csv"):
    output_name += ".csv"
  full_output_path: str = output_path + output_name
  results_df.to_csv(full_output_path, index=False)
  print(f"Saved results to {full_output_path}")