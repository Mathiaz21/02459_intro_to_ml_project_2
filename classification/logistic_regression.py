import pandas as pd
import numpy as np
import classification.models.logistic_regression as lr
import matplotlib.pyplot as plt
import utils.serve_data as sd

from utils.basic_operations import standardize_data

X = sd.get_features_dataset()
y = sd.get_targets()
X = standardize_data(X)


model = lr.generate_logistic_regression()
model.fit(X, y)

y_est = model.predict(X)
correct_predictions: int = np.sum(y == y_est)
total_predictions: int = len(y)
correct_percent: int = int(correct_predictions / total_predictions * 100)

print(f"The model predicted correctly {correct_predictions} predictions out of {total_predictions}, which gives us a success rate of {correct_percent}%")

y_est_benign_prob = model.predict_proba(X)[:, 0]

f = plt.figure()
classB_ids = np.nonzero(y == "B")[0].tolist()
plt.plot(classB_ids, y_est_benign_prob[classB_ids], ".y")
classM_ids = np.nonzero(y == "M")[0].tolist()
plt.plot(classM_ids, y_est_benign_prob[classM_ids], ".r")
plt.xlabel("Data object (Breast cell)")
plt.ylabel("Predicted prob. of class Benign")
plt.legend(["Benign", "Malignant"])
plt.ylim(-0.01, 1.5)

plt.show()
