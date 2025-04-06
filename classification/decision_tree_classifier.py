import pandas as pd
import utils.serve_data as sd

from sklearn import tree

X = sd.get_features_dataset()
y = sd.get_targets()


criterion = "gini"
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X, y)
