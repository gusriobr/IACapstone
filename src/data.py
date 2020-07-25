import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.python.keras.utils.np_utils import to_categorical

BASE_FOLDER = '/home/gus/workspaces/wpy/IACapstone'


def res(path):
    return os.path.join(BASE_FOLDER, path)


def tstmp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_original_data(sample_size=10000):
    data_file = "/media/data/projects/crophisto/data.npy"
    data = np.load(data_file)
    # sample data to 10.000
    random_indices = np.random.choice(
        data.shape[0], size=sample_size, replace=False)
    sample = data[random_indices, :]
    return sample


def load_structured_sample():
    file_path = res("resources/data_sampled.npy")
    data = np.load(file_path)
    return data


def load_undersampled_data():
    file_path = res("resources/data_undersampled.npy")
    data = np.load(file_path)
    return data

def prepare_data():
    # sample = load_original_data(sample_size=10000)
    sample = load_structured_sample()
    # sample = load_undersampled_data()

    y = sample[:, 8]
    X = sample[:, 0:8]

    # X = one_hot_encoding_X(X, vocab_size=vocab_size)
    num_clasess = len(np.unique(y[:, 11]))
    y = to_categorical(y)

    print(y.shape)
    # random train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create train/test/validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    for train_index, test_index in sss.split(X_train, y_train):
        X_train, X_val = X[train_index], X[test_index]
        y_train, y_val = y[train_index], y[test_index]
    print("===Train/validation/test size: {}, {}, {}.".format(len(y_train), len(y_val), len(y_test)))

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def read_crop_list():
    df_crops = pd.read_pickle("/media/data/projects/crophisto/crop_codes.pckl")
    # transform data to get a line per each
    crop_list = df_crops["code"].to_numpy()
    vocab = {val: idx for idx, val in enumerate(crop_list)}
    return df_crops, vocab


def one_hot_encoding_X(X, vocab_size):
    X_cat = np.zeros((X.shape[0], X.shape[1] * vocab_size), np.int8)
    for i in range(0, X.shape[1]):
        col = X[:, i]
        col_prim = to_categorical(col, num_classes=vocab_size)
        idx = i * vocab_size
        X_cat[:, idx:idx + vocab_size] = col_prim

    return X_cat
