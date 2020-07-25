from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from recipes import plot_confusion_matrix


def eval_model(folder, y_true, y_pred, classes, labels):
    # crop_list = np.unique(data)
    # crop_names = df_crops["description"].values.tolist()
    cfm = confusion_matrix(y_true, y_pred, classes)
    plot_confusion_matrix(cfm, classes=labels, figsize=(20, 20),
                          output_file="{}/cfm.png".format(folder))
    report = classification_report(y_true, y_pred)
    report_path = "{}/report.txt".format(folder)

    text_file = open(report_path, "w")
    text_file.write(report)
    text_file.close()

def eval_model_one_hot(folder, y_true, y_pred, classes, labels):
    class_test = np.argmax(y_true, axis=1)
    class_predicted = np.argmax(y_pred, axis=1)

    eval_model(folder, class_predicted, class_test, classes, labels)
