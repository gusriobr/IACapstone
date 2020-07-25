import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tpot import TPOTClassifier

from data import read_crop_list, load_structured_sample
from eval import eval_model
from training import create_training_folder

#
# digits = load_digits()
# X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
#                                                     train_size=0.75, test_size=0.25, random_state=42)

if __name__ == '__main__':

    df_crops, vocab = read_crop_list()

    data_file = "/media/data/projects/crophisto/data.npy"
    data = np.load(data_file)
    vocab_size = len(vocab)
    crop_list = np.unique(data)
    crop_names = df_crops["description"].values.tolist()

    # sample = load_original_data(sample_size=10000)
    sample = load_structured_sample()
    # sample = load_undersampled_data()
    print("Using sample size: {}".format(sample.shape))

    y = sample[:, 8]
    X = sample[:, 0:8]
    sequence_length = X.shape[-1]

    # X = one_hot_encoding_X(X, vocab_size=vocab_size)
    # y = to_categorical(y)
    # random train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

    # run prediction
    y_hat = tpot.predict(X_test)

    # eval model
    folder = create_training_folder("tpot")
    eval_model(folder, y_test, y_hat, crop_list, crop_names)
    # save model
    # model_folder = '{}/model'.format(folder)
    # model.save(model_folder)
    tpot.export('{}/model.py')
