import math
import os
import numpy as np

import keras
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint

from data import res, tstmp


def step_decay_exp(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


# learning rate schedule
def step_decay(epoch, lr):
    initial_lrate = 0.05
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def create_training_folder(exp_base, tag):
    folder_path = res("results/" + exp_base + "/" + tstmp() + "_" + tag)
    os.makedirs(folder_path, )
    return folder_path


def create_callbacks(folder, tensor_board=False, lr_scheduller=None,
                     csv_logger=True, early_stopper=True, patience=20, checkpoint_model=True, monitor_metric="val_loss",
                     monitor_mode="max"):
    callbacks = []
    if tensor_board:
        logdir = "{}/tfboard".format(folder)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True,
                                                           write_grads=True)
        callbacks.append(tensorboard_callback)
    if csv_logger:
        callbacks.append(CSVLogger('{}/training_history.csv'.format(folder), append=True, separator=';'))
    if lr_scheduller is not None:
        callbacks.append(lr_scheduller)
    if checkpoint_model:
        if not monitor_metric:
            raise Exception("Define the monitor metric!");
        checkpoint = ModelCheckpoint(folder, monitor=monitor_metric, verbose=1, save_best_only=True, mode=monitor_mode)
        callbacks.append(checkpoint)

    if early_stopper:
        if not monitor_metric:
            raise Exception("Define the monitor metric!");
        stopper = keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=0.001,
            mode=monitor_mode,
            # "no longer improving" being further defined as "for at least 8 epochs"
            patience=patience,
            verbose=1)
        callbacks.append(stopper)
    return callbacks


def train(model, X_train, y_train, X_test, y_test, epochs=200, batch_size=64, callbacks=[], calculate_steps=False):
    # define the model
    steps_per_epoch = num_val_steps = None
    if calculate_steps:
        steps_per_epoch = len(y_train) // batch_size
        num_val_steps = len(y_test) // batch_size

    # fit network
    model.fit(X_train, y_train,
              epochs=epochs,
              verbose=2,
              validation_data=(X_test, y_test),
              steps_per_epoch=steps_per_epoch,
              validation_steps=num_val_steps,
              callbacks=callbacks
              )
    loss = model.evaluate(X_test, y_test, verbose=0)
    print("Loss on test data: {}".format(loss))
