# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from packages.ann_models import MLPModel_wl
from packages.preprocessing import BOW
import pandas as pd
import numpy as np
import torch
import time


class ActivationFunctionStudy:

    @staticmethod
    def _generate_metrics(par_1, par_2):

        training_time = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_accScore = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_prcScore = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_rclScore = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_f1Score = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        return (training_time, searching_accScore, searching_prcScore,
                searching_rclScore, searching_f1Score)

    @staticmethod
    def _load_data(url, pos, bow_type, vocab_size, verbose, random_state,
                   train_size):

        tensors = BOW(
            bow_type=bow_type,
            vocab_size=vocab_size,
            verbose=verbose,
            random_state=random_state
        )

        tensors.fit_transform(
            url=url,
            pos=pos,
            train_prop=train_size
        )

        return (tensors.X_train, tensors.X_test, tensors.y_train,
                tensors.y_test)

    @staticmethod
    def _fit_predict(X_train, y_train, X_test, batch_size, n_hl,
                     num_features, n_classes, dropout, epochs, units,
                     bias, lr, momentum, device, weights_init,
                     hl_actfunct, out_actfunct, loss_funct,
                     verbose, random_state):

        model = MLPModel_wl(
            n_hl=n_hl,
            num_features=num_features,
            n_classes=n_classes,
            dropout=dropout,
            epochs=epochs,
            units=units,
            bias=bias,
            lr=lr,
            momentum=momentum,
            device=device,
            weights_init=weights_init,
            hl_actfunct=hl_actfunct,
            out_actfunct=out_actfunct,
            loss_funct=loss_funct,
            random_state=random_state,
            verbose=verbose
        )

        print("Fitting data...") if verbose is True else None
        t0 = time.time()
        model.fit(
            loader=(X_train, y_train),
            verbose=verbose,
            batch_size=batch_size
        )
        t1 = time.time()
        print(f"Fitting done in {t1-t0}s.") if verbose is True else None

        y_pred = model.predict_classes(X_test.to(device))
        y_pred = y_pred.to(torch.device("cpu"))

        return model, y_pred, t0, t1

    @staticmethod
    def _save_metrics(metrics, par_1, par_2, path_filename):

        for idx, metric in enumerate(metrics):
            df = pd.DataFrame(
                metric,
                index=par_1,
                columns=par_2
            )

            df.to_csv(f'{path_filename}_{idx}.csv')

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename):

        # seeding
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.out_path_filename = out_path_filename
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.par_1 = par_1
        self.par_2 = par_2

        (self.searching_accScore, self.searching_prcScore,
         self.searching_rclScore, self.searching_f1Score,
         self.training_time) = self._generate_metrics(
            par_1=par_1,
            par_2=par_2
        )

    def _store_metrics(self, c_idx, r_idx, t0, t1, y_test, y_pred):

        self.training_time[c_idx][r_idx] = t1 - t0

        test_acc = accuracy_score(y_test, y_pred)
        print(f"Test Acc: {test_acc}.") if self.verbose is True else None
        self.searching_accScore[c_idx][r_idx] = test_acc

        self.searching_prcScore[c_idx][r_idx] = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0)

        self.searching_rclScore[c_idx][r_idx] = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0)

        f1_s = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0)

        print(f"Test F1: {f1_s}.") if self.verbose is True else None
        self.searching_f1Score[c_idx][r_idx] = f1_s

    def run(self, url, pos, bow_type, vocab_size, batch_size,
            train_size, n_hl, dropout, epochs, units, bias, lr,
            momentum, weights_init, loss_funct):

        X_train, X_test, y_train, y_test = self._load_data(
            url=url,
            pos=pos,
            bow_type=bow_type,
            vocab_size=vocab_size,
            verbose=self.verbose,
            random_state=self.random_state,
            train_size=train_size,
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):

                model, y_pred, t0, t1 = self._fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    batch_size=batch_size,
                    n_hl=n_hl,
                    num_features=X_train.shape[1],
                    n_classes=20,
                    dropout=dropout,
                    epochs=epochs,
                    units=units,
                    bias=bias,
                    lr=lr,
                    momentum=momentum,
                    device=self.device,
                    weights_init=weights_init,
                    hl_actfunct=col,
                    out_actfunct=row,
                    loss_funct=loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    y_test=y_test,
                    y_pred=y_pred
                )

        metrics_ = (self.training_time, self.searching_accScore,
                    self.searching_prcScore, self.searching_rclScore,
                    self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


class BoWStudy(ActivationFunctionStudy):

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename)

    def run(self, url, pos, batch_size, train_size, hl_actfunct, out_actfunct,
            n_hl, dropout, epochs, units, bias, lr, momentum,
            weights_init, loss_funct, bow_type=None, vocab_size=None):

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):

                X_train, X_test, y_train, y_test = self._load_data(
                    url=url,
                    pos=pos,
                    bow_type=row,
                    vocab_size=col,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    train_size=train_size,
                )

                model, y_pred, t0, t1 = self._fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    batch_size=batch_size,
                    n_hl=n_hl,
                    num_features=X_train.shape[1],
                    n_classes=20,
                    dropout=dropout,
                    epochs=epochs,
                    units=units,
                    bias=bias,
                    lr=lr,
                    momentum=momentum,
                    device=self.device,
                    weights_init=weights_init,
                    hl_actfunct=hl_actfunct,
                    out_actfunct=out_actfunct,
                    loss_funct=loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    y_test=y_test,
                    y_pred=y_pred
                )

        metrics_ = (self.training_time, self.searching_accScore,
                    self.searching_prcScore, self.searching_rclScore,
                    self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


class HlUnitsStudy(ActivationFunctionStudy):

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename)

    def run(self, url, pos, batch_size, train_size, hl_actfunct, out_actfunct,
            dropout, epochs, bias, lr, momentum, weights_init,
            loss_funct, bow_type, vocab_size, n_hl=None, units=None):

        X_train, X_test, y_train, y_test = self._load_data(
            url=url,
            pos=pos,
            bow_type=bow_type,
            vocab_size=vocab_size,
            verbose=self.verbose,
            random_state=self.random_state,
            train_size=train_size,
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):

                model, y_pred, t0, t1 = self._fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    batch_size=batch_size,
                    n_hl=col,
                    num_features=X_train.shape[1],
                    n_classes=20,
                    dropout=dropout,
                    epochs=epochs,
                    units=row,
                    bias=bias,
                    lr=lr,
                    momentum=momentum,
                    device=self.device,
                    weights_init=weights_init,
                    hl_actfunct=hl_actfunct,
                    out_actfunct=out_actfunct,
                    loss_funct=loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    y_test=y_test,
                    y_pred=y_pred
                )

        metrics_ = (self.training_time, self.searching_accScore,
                    self.searching_prcScore, self.searching_rclScore,
                    self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


class LrMmtStudy(ActivationFunctionStudy):

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename)

    def run(self, url, pos, batch_size, train_size, hl_actfunct, out_actfunct,
            dropout, epochs, bias, weights_init, loss_funct,
            bow_type, vocab_size, n_hl, units, lr=None, momentum=None):

        X_train, X_test, y_train, y_test = self._load_data(
            url=url,
            pos=pos,
            bow_type=bow_type,
            vocab_size=vocab_size,
            verbose=self.verbose,
            random_state=self.random_state,
            train_size=train_size,
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    batch_size=batch_size,
                    n_hl=n_hl,
                    num_features=X_train.shape[1],
                    n_classes=20,
                    dropout=dropout,
                    epochs=epochs,
                    units=units,
                    bias=bias,
                    lr=col,
                    momentum=row,
                    device=self.device,
                    weights_init=weights_init,
                    hl_actfunct=hl_actfunct,
                    out_actfunct=out_actfunct,
                    loss_funct=loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    y_test=y_test,
                    y_pred=y_pred
                )

        metrics_ = (self.training_time, self.searching_accScore,
                    self.searching_prcScore, self.searching_rclScore,
                    self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


class EpochsBatchesStudy(ActivationFunctionStudy):

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename)

    def run(self, url, pos, train_size, hl_actfunct, out_actfunct,
            dropout, bias, weights_init, loss_funct,
            bow_type, vocab_size, n_hl, units, lr, momentum,
            epochs=None, batch_size=None):

        X_train, X_test, y_train, y_test = self._load_data(
            url=url,
            pos=pos,
            bow_type=bow_type,
            vocab_size=vocab_size,
            verbose=self.verbose,
            random_state=self.random_state,
            train_size=train_size,
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    batch_size=row,
                    n_hl=n_hl,
                    num_features=X_train.shape[1],
                    n_classes=20,
                    dropout=dropout,
                    epochs=col,
                    units=units,
                    bias=bias,
                    lr=lr,
                    momentum=momentum,
                    device=self.device,
                    weights_init=weights_init,
                    hl_actfunct=hl_actfunct,
                    out_actfunct=out_actfunct,
                    loss_funct=loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    y_test=y_test,
                    y_pred=y_pred
                )

        metrics_ = (self.training_time, self.searching_accScore,
                    self.searching_prcScore, self.searching_rclScore,
                    self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self
