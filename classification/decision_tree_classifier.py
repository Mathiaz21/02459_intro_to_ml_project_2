import pandas as pd

from sklearn import tree

X = pd.read_csv("../breast_cancer_wisconsin_features.csv") 
y = pd.read_csv("../breast_cancer_wisconsin_targets.csv")


criterion = "gini"
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X, y)
