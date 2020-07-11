import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from recipes import plot_confusion_matrix


def load_original_data(sample_size=10000):
    data_file = "/media/data/projects/crophisto/data.npy"
    data = np.load(data_file)
    # sample data to 10.000
    random_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
    sample = data[random_indices, :]
    return sample


def load_structure_sample():
    data = np.load("../resources/data_sampled.npy")
    return data


def read_crop_list():
    crop_codes = '/media/data/projects/crophisto/crop_codes.csv'

    # load crop info
    df_crops = pd.read_csv(crop_codes, sep=";")
    # transform data to get a line per each
    crop_list = df_crops["code"].to_numpy()
    vocab = {val: idx for idx, val in enumerate(crop_list)}
    return df_crops, vocab


def train_embedding(X_train, y_train, X_test, y_test, sequence_length, vocab_size, epochs=200):
    embedding_size = vocab_size
    # define the model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=sequence_length))
    model.add(LSTM(50))
    model.add(Dense(vocab_size, activation='sigmoid'))
    print(model.summary())

    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, y_train, epochs=epochs, verbose=2)
    # evaluate
    # print(generate_seq(model, tokenizer, 'Jack', 6))

    loss = model.evaluate(X_test, y_test, verbose=0)
    print("Loss on test data: {}".format(loss))

    y_hat = model.predict(X_test)

    class_test = np.argmax(y_test, axis=1)
    class_predicted = np.argmax(y_hat, axis=1)

    crop_list = np.unique(data)
    crop_names = df_crops["description"].values.tolist()
    cfm = confusion_matrix(class_test, class_predicted, crop_list)
    plot_confusion_matrix(cfm, classes=crop_names, figsize=(20, 20),
                          output_file="../resources/output/basic_embedding_{}.png".format(sequence_length))
    report = classification_report(class_test, class_predicted)
    report_path = "../resources/output/report_{}.txt".format(sequence_length)

    text_file = open(report_path, "w")
    text_file.write(report)
    text_file.close()


df_crops, vocab = read_crop_list()

data_file = "/media/data/projects/crophisto/data.npy"
data = np.load(data_file)

vocab_size = len(vocab)

# sample = load_original_data(sample_size=10000)
sample = load_structure_sample()
print("Using sample size: {}".format(sample.shape))

y = sample[:, 8]
X = sample[:, 0:8]
sequence_length = X.shape[-1]

# random train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

y_train = to_categorical(y_train, num_classes=vocab_size)
y_test = to_categorical(y_test, num_classes=vocab_size)

print("Train/test size: {}, {}".format(len(y_train), len(y_test)))

epochs = 2

for sequence_length in range(3, 9):
    print("Training with sequence: " + str(sequence_length))
    X_train_cut = X_train[:, -sequence_length:]
    X_test_cut = X_test[:, -sequence_length:]
    train_embedding(X_train_cut, y_train, X_test_cut, y_test, sequence_length, vocab_size, epochs=epochs)
