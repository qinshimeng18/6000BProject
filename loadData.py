from localPath import dataPath
import numpy as np
import os
from sklearn.decomposition import PCA

x = np.load(os.path.join(dataPath, 'x.npy'))
y = np.load(os.path.join(dataPath, 'y.npy'))

x = np.transpose(x)
y = np.transpose(y)

pca_obj = PCA(n_components=100)

x_pca = pca_obj.fit_transform(x)

for i in range(y.shape[0]):
    if y[i] == 0:
        y[i] = -1.0

train_x = x_pca[0:10000, :]
train_y = y[0:10000, :]

test_x = x_pca[10000:, :]
test_y = y[10000:, :]

pass