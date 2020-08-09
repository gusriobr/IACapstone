from tensorflow.python.keras.regularizers import l2, l1, l1_l2

from models import model_by_conf


def tcn_models():
    return [
         ["256_stacks2",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn", 'nb_filters': 128, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear'},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["128_stacks2",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear'},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l2_0001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l2(0.0001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l2_00001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l2(0.00001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l2_000001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l2(0.000001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],

        ["256_stacks2_reg_l1_0001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1(0.0001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l1_00001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1(0.00001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l1_000001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1(0.000001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],

        ["256_stacks2_reg_l1l2_0001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1_l2(0.0001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l1l2_00001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1_l2(0.00001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["256_stacks2_reg_l1l2_000001",
         model_by_conf(layer_conf=[
             {"type": "embeddings", 'input_dim': 27, 'output_dim': 20, 'input_length': 8},
             {"type": "batchnormalization"},
             {"type": "tcn_mod", 'nb_filters': 256, 'kernel_size': 2, 'nb_stacks': 2,
              'dilations': [1, 2, 4], 'dropout_rate': 0.3, 'return_sequences': False, 'kernel_initializer': 'he_normal',
              'use_batch_norm': True, 'activation': 'linear', 'kernel_regularizer': l1_l2(0.000001)},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
    ]


def conv1d_models():
    return [
        ["1d32",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 32, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])
         ],
        ["1d64",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["1d64_64",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["1d64_64_d128_fl",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "dense", "units": 128, "activation": "relu"},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["1d64_64_fl_d128",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "dense", "units": 128, "activation": "relu"},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])],
        ["1d64_64_fl_d128_sfm",
         model_by_conf(layer_conf=[
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "conv1d", 'filters': 64, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
             {"type": "MaxPooling1D", 'pool_size': 4},
             {"type": "dense", "units": 128, "activation": "relu"},
             {"type": "flatten"},
             {"type": "dense", "units": 27, "activation": "softmax"},
         ])],

        # quitar capa flatten, añadirla solo si pasamos de 128 unidades

    ]


def lst_models():
    return [
        ["lstm128",

         model_by_conf(layer_conf=[
             {"type": "lstm", "units": 256, "return_sequences": True},
             {"type": "SpatialDropOut1D", "rate": 0.5},
             {"type": "lstm", "units": 256, "batch_norm": 1, "recurrent_dropout": 0.4,
              "dropout": 0.3, "recurrent_regularizer": l2(0.001), "activation": "relu"},
             # {"type": "batchnormalization"},
             {"type": "dense", "units": 27, "activation": "relu"},
         ])

         ],
        # ["lstm128-dense",
        #  model_by_conf(layer_conf=[
        #      {"type": "lstm", "units": 128, "batch_norm": 1, "recurrent_dropout": 0.3,
        #       "dropout": 0.4, "recurrent_regularizer": l2(0.001)},
        #      {"type": "dense", "units": 256, "activation": "relu", "dropout": 0.5},
        #      {"type": "dense", "units": 27, "activation": "relu"},
        #  ])],
        # ["lstm256-d256",
        #  model_by_conf(layer_conf=[
        #      {"type": "lstm", "units": 256, "batch_norm": 1, "recurrent_dropout": 0.3,
        #       "dropout": 0.4, "recurrent_regularizer": l2(0.001)},
        #      {"type": "dense", "units": 256, "activation": "relu"},
        #      {"type": "dense", "units": 27, "activation": "relu"},
        #  ])],
        # ["lstm64_256",
        #  model_by_conf(layer_conf=[
        #      {"type": "lstm", "units": 64, "batch_norm": 1, "recurrent_dropout": 0.3,
        #       "dropout": 0.4, "recurrent_regularizer": l2(0.001)},
        #      {"type": "dense", "units": 256, "activation": "relu"},
        #      {"type": "dense", "units": 27, "activation": "relu"},
        #  ])],
        # # dos capas
        # ["lstm256-lstm256",
        #  model_by_conf(layer_conf=[
        #      {"type": "lstm", "units": 256, "batch_norm": 1, "return_sequences": True},
        #      {"type": "lstm", "units": 256, "batch_norm": 1, "batch_norm": 1},
        #      {"type": "dense", "units": 256, "activation": "relu"},
        #      {"type": "dense", "units": 27, "activation": "relu"},
        #  ])],
        # ["lstm64-lstm64",
        #  model_by_conf(layer_conf=[
        #      {"type": "lstm", "units": 64, "batch_norm": 1, "return_sequences": True},
        #      {"type": "lstm", "units": 64, "batch_norm": 1, "batch_norm": 1},
        #      {"type": "dense", "units": 256, "activation": "relu"},
        #      {"type": "dense", "units": 27, "activation": "relu"},
        #  ])]
    ]
