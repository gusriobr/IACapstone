from tensorflow.python.keras.regularizers import l2

from models import model_by_conf

conv1d_models = [
    ["1d32",
     model_by_conf(layer_conf=[
         {"type": "conv1d", 'filters': 32, 'kernel_size': 3, 'strides': 1, 'activation': 'relu'},
         {"type": "MaxPooling1D", 'pool_size': 4},
         {"type": "flatten"},
         {"type": "dense", "units": 27, "activation": "relu"},
     ])],
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

lstm_models = [
    ["lstm128",
     model_by_conf(layer_conf=[
         {"type": "lstm", "units": 256, "batch_norm": 1, "recurrent_dropout": 0.4,
          "dropout": 0.3, "recurrent_regularizer": l2(0.001), "activation": "relu"},
         {"type": "dense", "units": 27, "activation": "relu"},
     ])],
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
