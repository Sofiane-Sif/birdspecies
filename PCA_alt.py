import numpy as np
import pandas as pd
from pca import pca

# Load dataset
data = pd.read_csv('data.csv',index_col=0)

# Extract row labels
row_labels = data.iloc[:, 0].values

# Extract column labels
column_labels = data.columns[1:].values

# Reduce the data towards 3 PCs
model = pca(n_components=2)

# Fit transform
results = model.fit_transform(data,row_labels=row_labels, col_labels=column_labels)
# Initialize to reduce the data up to the nubmer of componentes that explains 95% of the variance.
fig, ax = model.scatter()
fig, ax = model.biplot(n_feat=4, PC=[0,1])
fig, ax = model.plot()
