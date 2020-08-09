import numbers

import tensorflow as tf
from tcn import TCN
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Embedding, LSTM, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, \
    Dropout, concatenate, Activation, SpatialDropout1D, Permute, Concatenate, GRU, MaxPooling1D, Flatten, Add, MaxPool1D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow_addons.utils.types import Activation
import tcn_mod
