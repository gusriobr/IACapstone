import numpy as np

data_file = "/media/data/projects/crophisto/data.npy"
data = np.load(data_file)
y = data[:, 8]
X = data[:, 0:8]

from keras.utils import to_categorical

vocab_size = len(np.unique(data))
print("vocab_size: {}".format(vocab_size))

X_cat = np.zeros((X.shape[0], X.shape[1] * vocab_size), np.int8)
for i in range(0,X.shape[1]):
    col = X[:, i]
    col_prim = to_categorical(col, num_classes=vocab_size)
    idx = i*vocab_size
    X_cat[:, idx:idx + vocab_size] = col_prim

dataset = np.hstack([X, y.reshape(-1,1)])
np.save("../resources/data_sampled", dataset)

X_cat.shape