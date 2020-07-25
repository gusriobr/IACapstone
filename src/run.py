import keras
import keras.backend as K
import tensorflow as tf

import models_1d
from data import read_crop_list, prepare_data
from eval import eval_model_one_hot
from metrics import f1, f1_loss
from models import model_basic_lstm
from training import train, create_training_folder, create_callbacks


def load_model(file_name):
    loaded_model = keras.models.load_model(file_name)
    return loaded_model


if __name__ == '__main__':

    tf.keras.utils.get_custom_objects()
    tf.keras.utils.get_custom_objects()['f1'] = f1
    tf.keras.utils.get_custom_objects()['f1_loss'] = f1_loss

    df_crops, vocab = read_crop_list()

    vocab_size = len(vocab)
    # crop_list = np.unique(sample)
    crop_names = df_crops["description"].values.tolist()
    crop_list = df_crops["code"].values.tolist()

    (X_train, y_train), (X_test, y_test), (X_val, y_val) = prepare_data()

    models = models_1d.lstm_models
    models = [
        ["lstm128", model_basic_lstm()]
    ]

    # models = [
    #     ["lstm_64_depth_5", load_model(res('results/20200719_193831_lstm_64_depth_5'))]
    # ]
    training_params = {
        'loss': f1_loss,
        # 'loss': 'categorical_crossentropy',
        # 'optimizer': 'rmsprop',
        'optimizer': tf.optimizers.RMSprop(lr=0.001, clipvalue=0.3),
        'metrics': [f1],
        'run_eagerly': False
    }

    epochs = 100
    evaluate = True
    exp_base = "conv1d"
    for reg in models:
        tag = reg[0]
        model = reg[1]

        K.clear_session()
        # tf.reset_default_graph()
        folder = create_training_folder(exp_base, tag)
        model.compile(**training_params)
        print(model.summary())
        # exit(0)
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                               patience=5, min_lr=0.001)

        # lr_scheduller = tf.keras.callbacks.LearningRateScheduler(step_decay_exp)
        # lr_scheduller = tf.keras.callbacks.ReduceLROnPlateau(step_decay_exp)
        lr_scheduller = None
        callbacks = create_callbacks(folder, tensor_board=True, monitor_metric="val_f1", monitor_mode="max",
                                     lr_scheduller=lr_scheduller)
        try:
            print("========== Training {} ========".format(tag))
            train(model, X_train, y_train, X_val, y_val, epochs=epochs, callbacks=callbacks)
            # save model
            model_folder = '{}/model'.format(folder)
            model.save(model_folder)

            if evaluate:
                # run prediction
                results = model.evaluate(X_test, y_test, batch_size=128)
                print("Model evaluation: {}".format(results))
                y_hat = model.predict(X_test)
                eval_model_one_hot(folder, y_test, y_hat, crop_list, crop_names)
        except Exception as e:
            print(str(e))
