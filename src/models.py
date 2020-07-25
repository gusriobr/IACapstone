import numbers

import tensorflow as tf
from keras.layers import concatenate, Activation
# from keras.layers import Embedding, LSTM, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, Dropout, MaxPooling1D, Permute
# from keras.layers import LSTM, Dense, Embedding, BatchNormalization, TimeDistributed, GRU
from keras.models import Sequential, Model
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Embedding, LSTM, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, \
    Dropout, \
    Permute, Concatenate, GRU, MaxPooling1D, Flatten
from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow_addons.utils.types import Activation


def activation_func(value):
    int_value = int(value)
    if int_value == 0:
        return "relu"
    elif int_value == 1:
        return "tanh"
    elif int_value == 2:
        return "sigmoid"


def regularizer(value):
    int_value = int(value)
    if int_value == 0:
        return "relu"
    elif int_value == 1:
        return "tanh"
    elif int_value == 2:
        return "sigmoid"


def model_basic_lstm(sequence_length=8, vocab_size=27, embedding_size=20, layer_size=256,
                     final_activation="relu", final_activation_f=None,
                     lstm_activation="tanh", lstm_activation_f=None,
                     batch_norm=True, dropout=0.3, recurrent_dropout=0.4, regularizers={}):
    if final_activation_f:
        final_activation = activation_func(final_activation_f)
    if lstm_activation_f:
        lstm_activation = activation_func(lstm_activation_f)

    initializer = tf.keras.initializers.Identity()
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        embeddings_initializer=initializer,
                        input_length=sequence_length))
    model.add(LSTM(int(layer_size), activation=lstm_activation,
                   dropout=dropout, recurrent_dropout=recurrent_dropout, **regularizers))

    if batch_norm is True or int(batch_norm) == 1:
        model.add(BatchNormalization())
    model.add(Dense(vocab_size, activation=final_activation))

    return model


def model_by_conf(sequence_length=8, vocab_size=27, embedding_size=20, layer_conf=[],
                  final_activation="relu", final_activation_f=None,
                  ):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=sequence_length))
    # base_conf = {"units": 256, "dropout": 0.3, "recurrent_dropout": 0, "activation": "sigmoid", "sigmoid": "tanh"}

    for i, params in enumerate(layer_conf):
        batch_norm = None
        if "batch_norm" in params:
            batch_norm = params["batch_norm"]
            del params["batch_norm"]
        layer_type = params["type"]
        del params["type"]

        print("Creating layer: " + layer_type)

        layer_k = None
        dropout = None
        if layer_type == "lstm":
            layer_k = LSTM(**params)
        elif layer_type == "dense":
            if "dropout" in params:
                dropout = params["dropout"]
                del params["dropout"]
            layer_k = Dense(**params)
        elif layer_type == "gru":
            layer_k = GRU(**params)
        elif layer_type == "conv1d":
            layer_k = Conv1D(**params)
        elif layer_type.lower() == "maxpooling1d":
            layer_k = MaxPooling1D(**params)
        elif layer_type.lower() == "flatten":
            layer_k = Flatten(**params)
        model.add(layer_k)

        if batch_norm is not None and int(batch_norm) > 0:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout))

    return model


def model_stack_lstm(sequence_length=8, vocab_size=27, embedding_size=20,
                     layer_size=(32, 64), l1_size=256, l2_size=256,
                     final_activation="relu", final_activation_f=None,
                     lstm_activation="tanh", lstm_activation_f=None,
                     layer_normalization=False,
                     dropout=0.4, dropout_rec=0.3,
                     regularizers=None,
                     reg_all_layers=False, batch_norm=True):
    if layer_size is None:
        layer_size = (int(l1_size), int(l2_size))

    if layer_normalization:
        layer_normalization = int(layer_normalization) == 1
    if final_activation_f:
        final_activation = activation_func(final_activation_f)
    if lstm_activation_f:
        lstm_activation = activation_func(lstm_activation_f)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=sequence_length))

    for i, units in enumerate(layer_size):
        current_regularizers = regularizers
        last_layer = i == len(layer_size) - 1
        first_layer = i == 0
        if layer_normalization and last_layer:
            lstm = LayerNormLSTMCell(units, dropout=dropout, recurrent_dropout=dropout_rec,
                                     activation='relu')
        else:
            if first_layer and not reg_all_layers:
                current_regularizers = {}
            lstm = LSTM(units, dropout=dropout, recurrent_dropout=dropout_rec,
                        activation=lstm_activation,
                        return_sequences=not last_layer, **current_regularizers
                        )
        model.add(lstm)
    if isinstance(batch_norm, numbers.Number):
        batch_norm = (int(batch_norm) == 1)

    if batch_norm:
        model.add(BatchNormalization())

    model.add(Dense(vocab_size, activation=final_activation))

    return model


def model_Conv1D(sequence_length, embeddings):
    vocab_size, embedding_size = embeddings
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=sequence_length))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(128, 10, activation='relu'))
    # model.add(Conv1D(128, 10, activation='relu'))
    # model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(vocab_size, activation='relu'))

    return model


def model_Conv1D_LSTM_skip(sequence_length, embeddings, layer_size=32, filters=32):
    vocab_size, embedding_size = embeddings
    input = Input(shape=(sequence_length,))

    embeddings = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    conv1 = Conv1D(filters=filters, kernel_size=2, padding="same", strides=1, activation='relu')(embeddings)
    # # conv3 = Conv1D(filters=32, kernel_size=7, padding="same", strides=1, activation='relu')(embeddings)
    # lstm1 = LSTM(64, dropout=0.3, recurrent_dropout=0.2, return_sequences = True)(embeddings)

    concat = Concatenate(axis=2)([conv1, embeddings])
    lstm = LSTM(layer_size, dropout=0.3, recurrent_dropout=0.2)(concat)

    output = Dense(vocab_size, activation='relu')(lstm)
    model = Model(input, output, name="conv-lstm_parallel")
    return model


def model_Conv1D_LSTM(sequence_length, embeddings):
    vocab_size, embedding_size = embeddings
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=sequence_length))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Permute((2, 1)))
    model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Dense(vocab_size, activation='relu'))

    return model


def model_Rest_Stacked_lstm(sequence_length, embeddings):
    vocab_size, embedding_size = embeddings

    input = Input(shape=(sequence_length,))
    embeddings = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    conv1 = Conv1D(filters=32, kernel_size=2, padding="same", strides=1, activation='relu')(embeddings)
    conv2 = Conv1D(filters=32, kernel_size=4, padding="same", strides=1, activation='relu')(embeddings)
    # # conv3 = Conv1D(filters=32, kernel_size=7, padding="same", strides=1, activation='relu')(embeddings)
    lstm1 = LSTM(64, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)(embeddings)

    concat = Concatenate(axis=2)([conv1, conv2, lstm1])
    concat = Dropout(0.3)(concat)
    lstm2 = LSTM(64, dropout=0.3, recurrent_dropout=0.2)(concat)

    btch = BatchNormalization()(lstm2)
    output = Dense(vocab_size, activation='relu')(btch)
    # output = concat
    model = Model(input, output, name="conv-lstm_parallel")
    # tag = "conv_lstm_restnet"
    return model


#


def model_basic_NN(input_dim, output_dim, layers=(64, 32)):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(layers[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='sigmoid'))

    return model


def model_time_distributed(sequence_length, vocab_size):
    embedding_size = vocab_size  # round(vocab_size / 2)

    input = Input(shape=(sequence_length,))

    # embeddings = Embedding(vocab_size, embedding_size,input_length=sequence_length)

    embeddings = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    conv1 = Conv1D(filters=32, kernel_size=3, padding="same", strides=1, activation='relu')(embeddings)
    conv2 = Conv1D(filters=32, kernel_size=5, padding="same", strides=1, activation='relu')(embeddings)

    concat = Concatenate(axis=2)([conv1, conv2, embeddings])

    lstm = LSTM(32, dropout=0.3, recurrent_dropout=0.2, return_sequences=True)(concat)

    btch = BatchNormalization()(lstm)
    output = Dense(vocab_size, activation='relu')(btch)

    model = Model(input, output, name="conv-lstm_parallel")
    return model


# def model_time_distributed(sequence_length, vocab_size):
#     embedding_size = vocab_size  # round(vocab_size / 2)
#
#     model = Sequential()
#     model.add(Embedding(vocab_size, embedding_size,
#                         input_length=sequence_length))
#     model.add(LSTM(30, return_sequences=True))
#     # model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(TimeDistributed(Dense(1, activation='relu')))  # , recurrent_dropout=0.2
#     # model.add(Dropout(0.2))
#     # model.add(BatchNormalization())
#     # model.add(Dense(vocab_size, activation='relu'))
#
#     return model


def model_gru(sequence_length, vocab_size):
    embedding_size = vocab_size
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=sequence_length))
    model.add(GRU(32, dropout=0.1, recurrent_dropout=0.5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(vocab_size, activation='relu'))

    return model

def lstmfcn(sequence_length, embeddings, layer_size = 64):
    vocab_size, embedding_size = embeddings

    ip = Input(shape=(1, sequence_length))  # tensor

    x = LSTM(layer_size)(ip)  # lstm block
    x = Dropout(0.8)(x)

    # permutes the first and second dimension of the input (connecting RNNs and convnets together)
    y = Permute((2, 1))(ip)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)  # for temporal convolution
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(vocab_size, activation='relu')(x)  # classification

    model = Model(ip, out)
    model.summary()

    return model


def lstm_conv(sequence_length, embeddings):
    vocab_size, embedding_size = embeddings

    input = Input(shape=(sequence_length,))
    embeddings1 = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    input2 = Input(shape=(sequence_length,))
    embeddings2 = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    input3 = Input(shape=(sequence_length,))
    embeddings3 = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)

    # conv
    conv1 = Conv1D(filters=64, kernel_size=3, padding="same", strides=1, activation='relu')(embeddings1)
    conv2 = Conv1D(filters=66, kernel_size=4, padding="same", strides=1, activation='relu')(embeddings2)
    lstm = LSTM(128, dropout=0.3, recurrent_dropout=0.4, return_sequences=True)(embeddings3)
    # # conv3 = Conv1D(filters=32, kernel_size=7, padding="same", strides=1, activation='relu')(embeddings)

    concat = Concatenate(axis=2)([conv1, conv2, lstm])

    #     concat = Dropout(0.3)(concat)
    lstm2 = LSTM(128, dropout=0.3, recurrent_dropout=0.4)(concat)

    btch = BatchNormalization()(lstm2)
    output = Dense(vocab_size, activation='relu')(btch)
    # output = concat
    model = Model(input, output, name="conv-lstm_parallel")
    # tag = "conv_lstm_restnet"
    return model


def fcn(sequence_length, embeddings, layer_size=64):
    vocab_size, embedding_size = embeddings

    input = Input(shape=(sequence_length,))

    # permutes the first and second dimension of the input (connecting RNNs and convnets together)
    y = Embedding(vocab_size, embedding_size, input_length=sequence_length)(input)
    y = Permute((2, 1))(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)  # for temporal convolution
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    out = Dense(vocab_size, activation='relu')(y)  # classification

    model = Model(input, out)
    model.summary()

    return model



def basic_lstm_flatten(sequence_length, embeddings):
    input = Input(shape=(sequence_length,))
    embeddings = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=sequence_length)(input)
    embeddings = Flatten()(embeddings)
    embeddings = Reshape((-1, embedding_size))(embeddings)
    lstm = LSTM(128, dropout=0.3, recurrent_dropout=0.5)(embeddings)
    btch = BatchNormalization()(lstm)
    output = Dense(vocab_size, activation='relu')(btch)
    # output = concat
    model = Model(input, output)
    # tag = "conv_lstm_restnet"
    return model