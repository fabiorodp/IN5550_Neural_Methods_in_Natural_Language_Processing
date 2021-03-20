# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig2.packages.preprocessing import pad_batches, process_raw_data
    from Oblig2.packages.preprocessing import load_embedding, TSVDataset
    from Oblig2.packages.preprocessing import RNNDataset, collate_fn
    from Oblig2.packages.ann_models import MLPModel, RNNModel
except:
    from packages.preprocessing import pad_batches, process_raw_data
    from packages.preprocessing import load_embedding, TSVDataset
    from packages.preprocessing import RNNDataset, collate_fn
    from packages.ann_models import MLPModel, RNNModel

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import time


pos_tagged = [
    '0',
    '3',
    '4',
    '5',
    '7',
    '9',
    '11',
    '13',
    '15',
    '17',
    '19',
    '21',
    '23',
    '24',
    '25',
    '26',
    '27',
    '28',
    '29',
    '75',
    '192',
    '200'
]

non_pos_tagged = [
    '1',
    '6',
    '8',
    '10',
    '12',
    '14',
    '16',
    '18',
    '20',
    '22',
    '40',
    '82'
]


# only interested in bidirectional
class RnnTypeBiDirectional:

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
    def _load_data(vocab_dir, train_data_url, vocab, pos, batch_size,
                   device, train_prop, verbose, random_state):

        print('\n\n\nload embeddings...') if verbose else None
        embedder = load_embedding(vocab_dir + vocab + '.zip')
        embedder.add('<unk>', weights=torch.rand(embedder.vector_size))
        embedder.add('<pad>', weights=torch.zeros(embedder.vector_size))
        pad_token = embedder.vocab['<pad>'].index

        print('\n\n\nloading data set...') if verbose else None
        df_train, df_test = process_raw_data(
            data_url=train_data_url,
            train_prop=train_prop,
            verbose=verbose,
            pos_tagged=pos,
            random_state=random_state
        )

        train_data = RNNDataset(
            embedder=embedder,
            df=df_train,
            device=device,
            random_state=random_state,
            label_vocab=None,
            verbose=verbose
        )

        test_data = RNNDataset(
            embedder=embedder,
            df=df_test,
            device=device,
            random_state=random_state,
            label_vocab=train_data.label_vocab,
            verbose=verbose
        )

        del df_test, df_train

        print('\n\n\nloading data loader...') if verbose else None
        # -> X[0] : [num_samples, longest_sentence] := words indices,
        # -> X[1] : [num_samples] := words_lengths,
        # -> y : [num_samples, longest_sentence] := labels.
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, pad_token, device='cpu')
        )

        test_loader = DataLoader(
            test_data,
            batch_size=len(test_data),
            collate_fn=lambda x: collate_fn(x, pad_token, device='cpu')
        )

        return (embedder, train_loader, test_loader)

    @staticmethod
    def _fit_predict(train_loader, test_loader, embedder, n_hl, dropout,
                     epochs, units, lr, momentum, device, loss_funct,
                     random_state, verbose, rnn_type, bidirectional,
                     freeze, lr_scheduler, factor, patience, pool_type,
                     need_valid):

        print('\n\n\nloading model...') if verbose else None
        model = RNNModel(
            emb=embedder,
            n_hl=n_hl,
            num_features=embedder.vector_size,
            n_classes=2,
            dropout=dropout,
            epochs=epochs,
            units=units,
            lr=lr,
            momentum=momentum,
            device=device,
            loss_funct=loss_funct,
            random_state=random_state,
            verbose=verbose,
            rnn_type=rnn_type,              # <- 'rnn', 'lstm', 'gru'
            bidirectional=bidirectional,    # <- True or False
            freeze=freeze,                  # <- True or False
            lr_scheduler=lr_scheduler,      # <- True or False
            factor=factor,
            patience=patience,
            pool_type=pool_type,
        )

        print("Fitting data...") if verbose else None
        t0 = time.time()
        model.fit(
            loader=train_loader,
            verbose=verbose,
            test=test_loader if need_valid else None
        )
        t1 = time.time()
        print(f"Fitting done in {t1 - t0}s.") if verbose else None

        y_test, y_pred = model.predict_classes(test_loader)
        y_test = y_test.to(torch.device("cpu"))
        y_pred = y_pred.to(torch.device("cpu"))

        return model, y_test, y_pred, t0, t1

    @staticmethod
    def _save_metrics(metrics, par_1, par_2, path_filename):

        for idx, metric in enumerate(metrics):
            df = pd.DataFrame(
                metric,
                index=par_1,
                columns=par_2
            )

            df.to_csv(f'{path_filename}_{idx}.csv')

    def _best_params(self):
        # best parameters according to accuracy (not f1-score)
        best_1 = np.argmax(self.searching_accScore, axis=0)
        best_2 = np.argmax(self.searching_accScore, axis=1)
        return (self.par_1[best_1[0]], self.par_2[best_2[0]])

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, epochs, units, lr, momentum, device,
                 loss_funct, batch_size, vocab, freeze, lr_scheduler,
                 factor, patience, pool_type, rnn_type=None,
                 bidirectional=None):

        # searching parameters:
        self.par_1 = par_1
        self.par_2 = par_2

        # creating matrices to store the searching scores
        (self.searching_accScore, self.searching_prcScore,
         self.searching_rclScore, self.searching_f1Score,
         self.training_time) = self._generate_metrics(
            par_1=par_1,
            par_2=par_2
        )

        self.vocab_dir = vocab_dir
        self.train_data_url = train_data_url
        self.out_path_filename = out_path_filename
        self.need_valid = need_valid
        self.verbose = verbose

        # seeding
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.n_hl = n_hl
        self.dropout = dropout
        self.epochs = epochs
        self.units = units
        self.lr = lr
        self.momentum = momentum
        self.device = device
        self.loss_funct = loss_funct
        self.batch_size = batch_size
        self.vocab = vocab
        self.freeze = freeze
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.pool_type = pool_type

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

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

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # rnn_type
            for r_idx, row in enumerate(self.par_2):  # bi-directional

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=col,
                    bidirectional=row,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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


# only want best vocab
class VocabRnnType(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, epochs, units, lr, momentum,
                 device, loss_funct, batch_size, freeze,
                 lr_scheduler, factor, patience, pool_type,
                 bidirectional, vocab=None, rnn_type=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        for c_idx, col in enumerate(self.par_1):  # vocab

            if col in pos_tagged:
                pos = True

            else:
                pos = False

            embedder, train_loader, test_loader = self._load_data(
                vocab_dir=self.vocab_dir,
                train_data_url=self.train_data_url,
                vocab=col,
                pos=pos,
                batch_size=self.batch_size,
                device=self.device,
                train_prop=0.75,
                verbose=self.verbose,
                random_state=self.random_state
            )

            for r_idx, row in enumerate(self.par_2):  # rnnType

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=row,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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


# interested in pooling type
class PoolTypeRNNType(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, epochs, units, lr, momentum,
                 device, loss_funct, batch_size, vocab, freeze,
                 lr_scheduler, factor, patience, bidirectional,
                 pool_type=None, rnn_type=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # pool
            for r_idx, row in enumerate(self.par_2):  # rnn type

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=row,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=col,
                    need_valid=self.need_valid
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


# only interested in best num_of_epochs and rnn_type
class BestEpochRNNType(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, units, lr, momentum,
                 device, loss_funct, batch_size, vocab, freeze,
                 lr_scheduler, factor, patience, pool_type,
                 bidirectional, rnn_type=None, epochs=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

        self.num_epochs = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=int
        )

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # large epoch
            for r_idx, row in enumerate(self.par_2):  # rnn_types

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=col,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=row,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
                )

                # np.argmax(self.val_losses) -> int + 1
                # np.max(self.val_losses) -> float

                num_epochs, score = model.best_epoch()
                self.training_time[c_idx][r_idx] = t1 - t0
                self.searching_accScore[c_idx][r_idx] = score
                self.num_epochs[c_idx][r_idx] = num_epochs

        metrics_ = (self.training_time, self.searching_accScore,
                    self.num_epochs)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


# interested in bach_size and epochs
class BatchEpoch(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, units, lr, momentum,
                 device, loss_funct, vocab, freeze,
                 lr_scheduler, factor, patience, pool_type,
                 rnn_type, bidirectional, batch_size=None, epochs=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        for c_idx, col in enumerate(self.par_1):  # batches

            embedder, train_loader, test_loader = self._load_data(
                vocab_dir=self.vocab_dir,
                train_data_url=self.train_data_url,
                vocab=self.vocab,
                pos=pos,
                batch_size=col,
                device=self.device,
                train_prop=0.75,
                verbose=self.verbose,
                random_state=self.random_state
            )

            for r_idx, row in enumerate(self.par_2):  # epochs

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=row,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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


# interested in n_hl and units
class HLUnits(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 dropout, epochs, lr, momentum,
                 device, loss_funct, batch_size, vocab, freeze,
                 lr_scheduler, factor, patience, pool_type,
                 rnn_type, bidirectional, n_hl=None, units=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # HL
            for r_idx, row in enumerate(self.par_2):  # Units

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=col,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=row,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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


# interested in lr and momentum
class LrMmt(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, epochs, units,
                 device, loss_funct, batch_size, vocab, freeze,
                 lr_scheduler, factor, patience, pool_type,
                 rnn_type, bidirectional, lr=None, momentum=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # Lr
            for r_idx, row in enumerate(self.par_2):  # Mmt

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    lr=col,
                    momentum=row,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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


# interested in factor and patience
class FactorPatience(RnnTypeBiDirectional):

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, need_valid, verbose, random_state,
                 n_hl, dropout, epochs, units, lr, momentum,
                 device, loss_funct, batch_size, vocab, freeze,
                 lr_scheduler, pool_type, rnn_type, bidirectional,
                 factor=None, patience=None):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                         out_path_filename, need_valid, verbose, random_state,
                         n_hl, dropout, epochs, units, lr, momentum,
                         device, loss_funct, batch_size, vocab, freeze,
                         lr_scheduler, factor, patience, pool_type,
                         rnn_type, bidirectional)

    def run(self):

        if self.vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedder, train_loader, test_loader = self._load_data(
            vocab_dir=self.vocab_dir,
            train_data_url=self.train_data_url,
            vocab=self.vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state
        )

        for c_idx, col in enumerate(self.par_1):  # Factor
            for r_idx, row in enumerate(self.par_2):  # Patience

                model, y_test, y_pred, t0, t1 = self._fit_predict(
                    train_loader=train_loader,
                    test_loader=test_loader,
                    embedder=embedder,
                    n_hl=self.n_hl,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                    freeze=self.freeze,
                    lr_scheduler=self.lr_scheduler,
                    factor=col,
                    patience=row,
                    pool_type=self.pool_type,
                    need_valid=self.need_valid
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
