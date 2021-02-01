# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
from packages.ann_models import MLPModel
from packages.preprocessing import BOW
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time


def hl_actFctXout_actFct(DEVICE, SEED, hl_actFcts, out_actFcts,
                         data_url, verbose):

    searching_accScore = np.empty(
        shape=(len(hl_actFcts), len(out_actFcts)),
        dtype=float
    )

    searching_prcScore = np.empty(
        shape=(len(hl_actFcts), len(out_actFcts)),
        dtype=float
    )

    searching_rclScore = np.empty(
        shape=(len(hl_actFcts), len(out_actFcts)),
        dtype=float
    )

    searching_f1Score = np.empty(
        shape=(len(hl_actFcts), len(out_actFcts)),
        dtype=float
    )

    training_time = np.empty(
        shape=(len(hl_actFcts), len(out_actFcts)),
        dtype=float
    )

    tensors = BOW(
        bow_type="counter",
        vocab_size=0,
        device=DEVICE,
        verbose=verbose,
        random_state=SEED
    )

    tensors.fit_transform(
        url=data_url
    )

    X_train, X_test, y_train, y_test = train_test_split(
        tensors.input_tensor,
        tensors.target,
        test_size=0.25,
        random_state=SEED,
        shuffle=True,
        stratify=tensors.target
    )

    for c_idx, col in enumerate(hl_actFcts):
        for r_idx, row in enumerate(out_actFcts):

            model = MLPModel(
                n_hl=1,
                num_features=X_train.shape[1],
                n_classes=7,
                dropout=0.2,
                epochs=10,
                units=10,
                bias=0.1,
                lr=0.01,
                momentum=0.9,
                device=DEVICE,
                weights_init="xavier_normal",
                hl_actfunct=col,
                out_actfunct=row,
                loss_funct="cross-entropy",
                random_state=SEED
            )

            t0 = time.time()
            model.fit(
                input_tensor=X_train,
                target=y_train,
                batch_size=1
            )
            t1 = time.time()

            y_pred = model.predict_classes(X_test)
            y_true = y_test

            training_time[c_idx][r_idx] = t1-t0

            searching_accScore[c_idx][r_idx] = \
                accuracy_score(y_true, y_pred)

            searching_prcScore[c_idx][r_idx] = precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0)

            searching_rclScore[c_idx][r_idx] = recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0)

            searching_f1Score[c_idx][r_idx] = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0)

    htm_arrays = [training_time,
                  searching_accScore,
                  searching_prcScore,
                  searching_rclScore,
                  searching_f1Score]

    htm_titles = ["Training elapsed time in seconds",
                  "Accuracy Score Heatmap",
                  "Macro Precision Score Heatmap",
                  "Macro Recall Score Heatmap",
                  "Macro F1 Score Heatmap"]

    for array, title in zip(htm_arrays, htm_titles):
        sns.heatmap(
            data=array,
            annot=True,
            fmt=".4g",
            xticklabels=out_actFcts,
            yticklabels=hl_actFcts
        )
        plt.title(title)
        plt.ylabel("Activation Function for hidden layers.")
        plt.xlabel("Activation Function for output layer")
        plt.show()


def vocab_numXbow_type(DEVICE, SEED, vocab_sizes, bow_types, data_url,
                       verbose):

    searching_accScore = np.empty(
        shape=(len(vocab_sizes), len(bow_types)),
        dtype=float
    )

    searching_prcScore = np.empty(
        shape=(len(vocab_sizes), len(bow_types)),
        dtype=float
    )

    searching_rclScore = np.empty(
        shape=(len(vocab_sizes), len(bow_types)),
        dtype=float
    )

    searching_f1Score = np.empty(
        shape=(len(vocab_sizes), len(bow_types)),
        dtype=float
    )

    training_time = np.empty(
        shape=(len(vocab_sizes), len(bow_types)),
        dtype=float
    )

    for c_idx, col in enumerate(vocab_sizes):
        for r_idx, row in enumerate(bow_types):

            tensors = BOW(
                bow_type=row,
                vocab_size=col,
                device=DEVICE,
                verbose=verbose,
                random_state=SEED
            )

            tensors.fit_transform(
                url=data_url
            )

            X_train, X_test, y_train, y_test = train_test_split(
                tensors.input_tensor,
                tensors.target,
                test_size=0.25,
                random_state=SEED,
                shuffle=True,
                stratify=tensors.target
            )

            model = MLPModel(
                n_hl=1,
                num_features=X_train.shape[1],
                n_classes=7,
                dropout=0.2,
                epochs=10,
                units=10,
                bias=0.1,
                lr=0.01,
                momentum=0.9,
                device=DEVICE,
                weights_init="xavier_normal",
                hl_actfunct="tanh",
                out_actfunct="softmax",
                loss_funct="cross-entropy",
                random_state=SEED
            )

            t0 = time.time()
            model.fit(
                input_tensor=X_train,
                target=y_train,
                batch_size=1
            )
            t1 = time.time()

            y_pred = model.predict_classes(X_test)
            y_true = y_test

            training_time[c_idx][r_idx] = t1-t0

            searching_accScore[c_idx][r_idx] = \
                accuracy_score(y_true, y_pred)

            searching_prcScore[c_idx][r_idx] = precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0)

            searching_rclScore[c_idx][r_idx] = recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0)

            searching_f1Score[c_idx][r_idx] = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0)

    htm_arrays = [training_time,
                  searching_accScore,
                  searching_prcScore,
                  searching_rclScore,
                  searching_f1Score]

    htm_titles = ["Training elapsed time in seconds",
                  "Accuracy Score Heatmap",
                  "Macro Precision Score Heatmap",
                  "Macro Recall Score Heatmap",
                  "Macro F1 Score Heatmap"]

    for array, title in zip(htm_arrays, htm_titles):
        sns.heatmap(
            data=array,
            annot=True,
            fmt=".4g",
            xticklabels=bow_types,
            yticklabels=vocab_sizes
        )
        plt.title(title)
        plt.ylabel("Vocabulary size in quantity of words. "
                   "0 means all words.")
        plt.xlabel("Types of Bag of words.")
        plt.show()


if __name__ == '__main__':

    hl_actFctXout_actFct(
        DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        SEED=1,
        hl_actFcts=["sigmoid", "tanh", "relu"],
        out_actFcts=["sigmoid", "tanh", "relu", "softmax"],
        data_url='data/sample.tsv',
        verbose=False
    )

    vocab_numXbow_type(
        DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        SEED=1,
        vocab_sizes=[0, 1000, 750, 500, 250, 100, 50, 10],
        bow_types=["counter", "binary", "tfidf"],
        data_url='data/sample.tsv',
        verbose=False
    )
