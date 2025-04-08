import numpy as np
import utils.serve_data as sd

y = sd.get_targets()

nb_benign = np.sum(y == "B")
nb_malignant = np.sum(y == "M")
ratio_benign = int(nb_benign / len(y) * 100)

print(f"There are {nb_benign} benign cells in the dataset, and {nb_malignant} malignant cells")
print(f"Out of {len(y)} cells in total")
print(f"This makes for {ratio_benign}% of benign cells.")