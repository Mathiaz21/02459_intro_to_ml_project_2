from ucimlrepo import fetch_ucirepo
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
features_df = breast_cancer_wisconsin_diagnostic.data.features 
targets_df = breast_cancer_wisconsin_diagnostic.data.targets


features_df.to_csv('breast_cancer_wisconsin_features.csv', index=False)
targets_df.to_csv('breast_cancer_wisconsin_targets.csv', index=False)
