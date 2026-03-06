
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

# Load dataset
data = load_iris()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)

print("Original Dataset:")
print(df.head())

# Variance Threshold
variance_threshold = 0.2
selector = VarianceThreshold(threshold=variance_threshold)

X_var_thresh = selector.fit_transform(X)
df_var_thresh = pd.DataFrame(X_var_thresh, columns=df.columns[selector.get_support()])

print("\nDataset after Variance Thresholding:")
print(df_var_thresh.head())

# Correlation Analysis
correlation_matrix = df.corr()
correlation_threshold = 0.9

drop_columns = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            drop_columns.add(colname)

df_reduced = df.drop(columns=drop_columns)

print("\nDataset after Correlation Analysis:")
print(df_reduced.head())
