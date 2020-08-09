import numbers

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.layers import Embedding, LSTM, BatchNormalization, Dense, Conv1D, GlobalAveragePooling1D, \
    Dropout, concatenate, Activation, SpatialDropout1D, Permute, Concatenate, GRU, MaxPooling1D, Flatten, Add, MaxPool1D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow_addons.rnn import LayerNormLSTMCell
from tensorflow_addons.utils.types import Activation
import tcn_mod


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
    # model.add(Embedding(vocab_size, embedding_size,
    #                     input_length=sequence_length))
    # base_conf = {"units": 256, "dropout": 0.3, "recurrent_dropout": 0, "activation": "sigmoid", "sigmoid": "tanh"}

    for i, params in enumerate(layer_conf):
        batch_norm = None
        if "batch_norm" in params:
            batch_norm = params["batch_norm"]
            del params["batch_norm"]
        layer_type = params["type"].lower()
        del params["type"]

        print("Creating layer: " + layer_type)

        layer_k = None
        dropout = None
        if layer_type == "lstm":
            layer_k = LSTM(**params)
        elif layer_type == "embeddings":
            layer_k = Embedding(**params)
        elif layer_type == "dense":
            if "dropout" in params:
                dropout = params["dropout"]
                del params["dropout"]
            layer_k = Dense(**params)
        elif layer_type == "batchnormalization":
            layer_k = BatchNormalization(**params)
        elif layer_type == "globalaveragepooling1d":
            layer_k = BatchNormalization(**params)
        elif layer_type == "spatialdropout1d":
            layer_k = SpatialDropout1D(**params)
        elif layer_type == "gru":
            layer_k = GRU(**params)
        elif layer_type == "conv1d":
            layer_k = Conv1D(**params)
        elif layer_type == "maxpooling1d":
            layer_k = MaxPooling1D(**params)
        elif layer_type == "flatten":
            layer_k = Flatten(**params)
        # elif layer_type == "tcn":
        #     layer_k = TCN(**params)
        # elif layer_type == "tcn_mod":
        #     layer_k = tcn_mod.TCN(**params)
        else:
            raise Exception("Type not found: " + layer_type)
        model.add(layer_k)

        if batch_norm is not None and int(batch_norm) > 0:
            model.add(BatchNormalization())
        if dropout:
            model.add(Dropout(dropout))

    return model


def model_inceptionTime(input_shape, nb_classes, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6,
                        kernel_size=41):
    def _inception_module(input_tensor, stride=1, activation='linear', bottleneck_size=True):

        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        # kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        kernel_size_s = [1, 2, 4, 8]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                    strides=stride, padding='same',
                                    activation=activation, use_bias=False)(input_inception))

        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv_list.append(conv_6)

        x = Concatenate(axis=2)(conv_list)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        return x

    def _shortcut_layer(input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                            padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)
        x = Add()([shortcut_y, out_tensor])
        x = Activation('relu')(x)
        return x

    input_layer = Input(input_shape)
    x = input_layer
    input_res = input_layer

    for d in range(depth):
        x = _inception_module(x)
        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)
    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
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


def lstm_conv1_mutibranch(sequence_length, vocab_size, embedding_size, layer_size=64):
    input = Input(shape=(1, sequence_length))  # tensor

    conv_emb = Embedding(vocab_size, embedding_size, input_length=sequence_length)(input)

    conv2 = Conv1D(64, 2, padding='same')(conv_emb)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(64, 3, padding='same')(conv_emb)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv5 = Conv1D(64, 5, padding='same')(conv_emb)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    lstm_emb = Embedding(vocab_size, embedding_size, input_length=sequence_length)(input)
    lstm = LSTM(layer_size)(lstm_emb)  # lstm block
    lstm = SpatialDropout1D(0.3)(lstm)
    lstm = LSTM(units=128, recurrent_dropout=0.3)(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)

    x = Concatenate(axis=2)([conv2, conv3, conv5, lstm])
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Dense(vocab_size, activation='relu')(x)


def lstmfcn(sequence_length, embeddings, layer_size=64):
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


def inception(input):
    conv1 = Conv1D(32, 2, padding='same')(input)
    conv2 = Conv1D(32, 3, padding='same')(input)
    conv3 = Conv1D(32, 5, padding='same')(input)

    pool = MaxPooling1D(2, strides=1, padding='same')(input)
    conv4 = Conv1D(32, 1, padding='same')(pool)

    output = Concatenate(axis=2)([conv1, conv2, conv3, conv4])
    return output


def lstm_inception(sequence_length, embeddings, num_modules=2, layer_size=256):
    vocab_size, embedding_size = embeddings

    input = Input(shape=(sequence_length,))

    conv_emb = Embedding(vocab_size, embedding_size, input_length=sequence_length)(input)

    for i in range(0, num_modules):
        x = inception(conv_emb)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    lstm_emb = Embedding(vocab_size, embedding_size, input_length=sequence_length)(input)
    lstm = SpatialDropout1D(0.3)(lstm_emb)
    lstm = LSTM(units=64, recurrent_dropout=0.3, return_sequences=True)(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(0.3)(lstm)

    x = Concatenate(axis=2)([x, lstm])
    #     x = BatchNormalization()(x)
    #     x = Activation(activation='relu')(x)

    #     x = SpatialDropout1D(0.3)(x)
    x = LSTM(units=layer_size, recurrent_dropout=0.3, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Flatten()(x)

    output = Dense(vocab_size, activation='relu')(x)

    return Model(input, output)