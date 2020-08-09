import os
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Model

from data import read_crop_list, load_structured_sample
from eval import eval_model_one_hot
from metrics import f1, f1_loss
from models import model_stack_lstm, model_Conv1D, model_Conv1D_LSTM
from training import train, create_training_folder
from keras.utils import plot_model
from training import train, create_training_folder, create_callbacks, step_decay_exp


from keras.models import Sequential
from tensorflow.python.keras.layers import concatenate, Embedding, LSTM, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, \
    Dropout, Input, Permute, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten, Activation, Reshape


from training import train, create_training_folder


if __name__ == '__main__':
    from training import train, create_training_folder

    tf.keras.utils.get_custom_objects()
    tf.keras.utils.get_custom_objects()['f1'] = f1
    tf.keras.utils.get_custom_objects()['f1_loss'] = f1_loss

    df_crops, vocab = read_crop_list()

    sample = load_structured_sample()
    # sample = load_undersampled_data()
    print("Using sample size: {}".format(sample.shape))

    y = sample[:, 11]
    X = sample[:, 3:11]
    sequence_length = X.shape[-1]

    # X = one_hot_encoding_X(X, vocab_size=vocab_size)
    y = to_categorical(y)
    # random train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # y_train = to_categorical(y_train, num_classes=vocab_size)
    # y_test = to_categorical(y_test, num_classes=vocab_size)
    crop_names = df_crops["description"].values.tolist()
    crop_list = df_crops["idx"].values.tolist()

    print(np.unique(y_train).shape)


    print("===Train/test size: {}, {}".format(len(y_train), len(y_test)))


    folder = "/home/gus/workspaces/wpy/IACapstone/results/tcn/20200802_085010_64"
    model = tf.keras.models.load_model('/home/gus/workspaces/wpy/IACapstone/results/tcn/20200802_085010_64/model')
    # evaluate
    y_hat = model.predict(X_test)
    eval_model_one_hot(folder, y_test, y_hat, crop_list, crop_names)

