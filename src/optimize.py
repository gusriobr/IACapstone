import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit

from data import read_crop_list, load_structured_sample
from eval import eval_model_one_hot
from metrics import f1, f1_loss
from models import model_basic_lstm, model_stack_lstm
from training import train, create_training_folder, create_callbacks


def load_data():
    global vocab_size, crop_list, crop_names, sequence_length, X_train, X_test, y_train, y_test
    df_crops, vocab = read_crop_list()
    # sample = load_original_data(sample_size=10000)
    sample = load_structured_sample()
    # sample = load_undersampled_data()
    print("Using sample size: {}".format(sample.shape))
    vocab_size = len(vocab)
    crop_list = np.unique(sample)
    crop_names = df_crops["description"].values.tolist()
    y = sample[:, 8]
    X = sample[:, 0:8]
    sequence_length = X.shape[-1]
    # X = one_hot_encoding_X(X, vocab_size=vocab_size)
    y = to_categorical(y)
    # random train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    print(np.unique(y_train).shape)
    return X_train, y_train, X_test, y_test, crop_list, crop_names


def load_model(file_name):
    loaded_model = keras.models.load_model(file_name)
    return loaded_model


BASE_TRAINING_PARAMS = {
    'loss': f1_loss,
    # 'loss': 'categorical_crossentropy',
    # 'optimizer': 'rmsprop',
    'optimizer': tf.optimizers.RMSprop(lr=0.001, clipvalue=0.3),
    'metrics': [f1],
    'run_eagerly': False
}

evaluate = True
epochs = 120
evaluate = True
exp_base = "regularizers2"


def fit_closure(model_func):
    def partial_fit(**params):
        model_params = {k: v for (k, v) in params.items() if not k.startswith("opt_")}
        opt_params = {k.replace("opt_", ""): v for (k, v) in params.items() if k.startswith("opt_")}

        print("fitting with model params: {}".format(model_params))
        print("Optimizer params: {}".format(opt_params))

        model = model_func(**model_params)
        tag = '_'.join('{}_{}'.format(key, round(value, 4)) for key, value in params.items())
        tag = tag.replace(".","_")

        K.clear_session()
        # tf.reset_default_graph()
        training_params = BASE_TRAINING_PARAMS.copy()
        if opt_params:
            training_params['optimizer'] = tf.optimizers.RMSprop(**opt_params)

        folder = create_training_folder(exp_base, tag)

        model.compile(**training_params)
        print(model.summary())

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                               patience=5, min_lr=0.001)
        # lr_scheduller = tf.keras.callbacks.LearningRateScheduler(step_decay_exp)
        # lr_scheduller = tf.keras.callbacks.ReduceLROnPlateau(step_decay_exp)
        lr_scheduller = None
        callbacks = create_callbacks(folder, tensor_board=True, monitor_metric="val_f1", monitor_mode="max",
                                     lr_scheduller=lr_scheduller)
        try:
            print("========== Training {} ========".format(tag))
            train(model, X_train, y_train, X_test, y_test, epochs=epochs, callbacks=callbacks)
            # save model
            model_folder = '{}/model'.format(folder)
            model.save(model_folder)
            if evaluate:
                # run prediction
                y_hat = model.predict(X_test)
                eval_model_one_hot(folder, y_test, y_hat, crop_list, crop_names)

            score = model.evaluate(X_test, y_test)
            print("{}, score: {}".format(tag, score))
            return score[1]

        except:
            print("======== Error ========== " + tag)
            return 0

    return partial_fit


def opt_1l_lstm():
    # Bounded region of parameter space
    pbounds = {'dropout': (0.1, 0.5), 'recurrent_dropout': (0.1, 0.5),
               "opt_lr": (0.001, 0.0001), "opt_clipvalue": (0.15, 0.5)}
    return model_basic_lstm, pbounds


def opt_2l_lstm():
    # Bounded region of parameter space
    pbounds = {'dropout': (0.1, 0.5), 'recurrent_dropout': (0.1, 0.5),
               'l1_size': (64, 512), 'l2_size': (64, 512)
               }
    return model_stack_lstm, pbounds


if __name__ == '__main__':
    tf.keras.utils.get_custom_objects()
    tf.keras.utils.get_custom_objects()['f1'] = f1
    tf.keras.utils.get_custom_objects()['f1_loss'] = f1_loss

    X_train, y_train, X_test, y_test, crop_list, crop_names = load_data()
    vocab_size = len(crop_list)

    model_func, pbounds = opt_1l_lstm()
    opt_func = fit_closure(model_func)

    optimizer = BayesianOptimization(f=opt_func, pbounds=pbounds,
                                     verbose=2,
                                     # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                                     random_state=1,
                                     )
    optimizer.maximize(init_points=10, n_iter=20)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("=========Max iteration===============")
    print(optimizer.max)
