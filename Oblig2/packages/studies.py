# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig2.packages.preprocessing import load_embedding, TSVDataset
    from Oblig2.packages.preprocessing import pad_batches
    from Oblig2.packages.ann_models import MLPModel
except:
    from packages.preprocessing import load_embedding, TSVDataset
    from packages.preprocessing import pad_batches
    from packages.ann_models import MLPModel

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import time


DIR = "/cluster/shared/nlpl/data/vectors/latest/"
# DIR = "saga/"
URL = 'data/stanford_sentiment_binary.tsv.gz'

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


class VocabLossFunct:

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
    def _load_data(vocab, pos, batch_size, device):

        print('\n\n\nload embeddings...')
        embedding = load_embedding(DIR + vocab + '.zip')

        embedding.add(
            '<pad>',
            weights=torch.zeros(embedding.vector_size)
        )

        pad_idx = embedding.vocab['<pad>'].index

        print('\n\n\nloading data set...')
        data = TSVDataset(
            embedder=embedding,
            url=URL,
            pos_tagged=pos,
            device=device
        )

        print('\n\n\nloading data loader...')
        loader = DataLoader(
            data,
            batch_size=batch_size,
            collate_fn=lambda x: pad_batches(x, pad_idx)
        )

        return (embedding, data, loader)

    @staticmethod
    def _fit_predict(data, loader, embedding, n_hl,
                     num_features, n_classes, dropout, epochs, units,
                     bias, lr, momentum, device, weights_init,
                     hl_actfunct, out_actfunct, loss_funct,
                     verbose, random_state, embedding_type):

        print('\n\n\nloading model...')
        model = MLPModel(
            emb=embedding,
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
            verbose=verbose,
            embedding_type=embedding_type
        )

        print("Fitting data...") if verbose is True else None
        t0 = time.time()
        model.fit(
            loader=loader,
            verbose=verbose
        )
        t1 = time.time()
        print(f"Fitting done in {t1 - t0}s.") if verbose is True else None

        y_pred = model.predict_classes(data.get_embedded_test_tensor())
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

    def _best_params(self):
        # best parameters according to accuracy (not f1-score)
        best_1 = np.argmax(self.searching_accScore, axis=0)
        best_2 = np.argmax(self.searching_accScore, axis=1)
        return (self.par_1[best_1[0]], self.par_2[best_2[0]])

    def __init__(self, par_1, par_2, random_state, verbose, device,
                 out_path_filename, n_hl, dropout, epochs, units,
                 lr, momentum, weights_init, hl_actfunct, out_actfunct,
                 batch_size, embedding_type):

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

        self.n_hl = n_hl
        self.dropout = dropout
        self.epochs = epochs
        self.units = units
        self.lr = lr
        self.momentum = momentum
        self.weights_init = weights_init
        self.hl_actfunct = hl_actfunct
        self.out_actfunct = out_actfunct
        self.batch_size = batch_size
        self.embedding_type = embedding_type

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

        for c_idx, col in enumerate(self.par_1):

            if col in pos_tagged:
                pos = True

            else:
                pos = False

            embedding, data, loader = self._load_data(
                vocab=col,
                pos=pos,
                batch_size=self.batch_size,
                device=self.device
            )

            for r_idx, row in enumerate(self.par_2):

                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=self.n_hl,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    bias=0.1,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    loss_funct=row,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=self.embedding_type
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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


class EbdTypeNewVocab(VocabLossFunct):

    def __init__(self, par_1, par_2, loss_funct, random_state, verbose,
                 device, out_path_filename, n_hl, dropout, epochs, units,
                 lr, momentum, weights_init, batch_size, hl_actfunct,
                 out_actfunct, pre_trained_vocab=None, embedding_type=None):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename, n_hl, dropout, epochs, units,
                         lr, momentum, weights_init, hl_actfunct,
                         out_actfunct, batch_size, embedding_type)

        self.pre_trained_vocab = pre_trained_vocab
        self.loss_funct = loss_funct

    def run(self):

        for c_idx, col in enumerate(self.par_1):

            if col in pos_tagged:
                pos = True

            else:
                pos = False

            embedding, data, loader = self._load_data(
                vocab=col,
                pos=pos,
                batch_size=self.batch_size,
                device=self.device
            )

            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=self.n_hl,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    bias=0.1,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    loss_funct=self.loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=row
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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


class ActivationFunctionStudy(VocabLossFunct):

    def __init__(self, par_1, par_2, pre_trained_vocab, loss_funct,
                 random_state, verbose, device, out_path_filename,
                 n_hl, dropout, epochs, units, lr, momentum, weights_init,
                 embedding_type, batch_size, hl_actfunct=None,
                 out_actfunct=None):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename, n_hl, dropout, epochs, units,
                         lr, momentum, weights_init, hl_actfunct,
                         out_actfunct, batch_size, embedding_type)

        self.pre_trained_vocab = pre_trained_vocab
        self.loss_funct = loss_funct

    def run(self):

        if self.pre_trained_vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedding, data, loader = self._load_data(
            vocab=self.pre_trained_vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=self.n_hl,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    bias=0.1,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=col,
                    out_actfunct=row,
                    loss_funct=self.loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=self.embedding_type
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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


class EpochsBatchesStudy(VocabLossFunct):

    def __init__(self, par_1, par_2, pre_trained_vocab, loss_funct,
                 random_state, verbose, device, out_path_filename,
                 n_hl, dropout, units, lr, momentum, weights_init,
                 hl_actfunct, out_actfunct, embedding_type,
                 epochs=None, batch_size=None):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename, n_hl, dropout, epochs, units,
                         lr, momentum, weights_init, hl_actfunct,
                         out_actfunct, batch_size, embedding_type)

        self.pre_trained_vocab = pre_trained_vocab
        self.loss_funct = loss_funct

    def run(self):

        if self.pre_trained_vocab in pos_tagged:
            pos = True

        else:
            pos = False

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):

                embedding, data, loader = self._load_data(
                    vocab=self.pre_trained_vocab,
                    pos=pos,
                    batch_size=row,
                    device=self.device
                )

                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=self.n_hl,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=col,
                    units=self.units,
                    bias=0.1,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    loss_funct=self.loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=self.embedding_type
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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


class HlUnitsStudy(VocabLossFunct):

    def __init__(self, par_1, par_2, pre_trained_vocab, loss_funct,
                 random_state, verbose, device, out_path_filename,
                 dropout, lr, momentum, weights_init, hl_actfunct,
                 out_actfunct, epochs, batch_size, embedding_type,
                 n_hl=None, units=None):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename, n_hl, dropout, epochs, units,
                         lr, momentum, weights_init, hl_actfunct,
                         out_actfunct, batch_size, embedding_type)

        self.pre_trained_vocab = pre_trained_vocab
        self.loss_funct = loss_funct

    def run(self):

        if self.pre_trained_vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedding, data, loader = self._load_data(
            vocab=self.pre_trained_vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=col,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=row,
                    bias=0.1,
                    lr=self.lr,
                    momentum=self.momentum,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    loss_funct=self.loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=self.embedding_type
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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


class LrMmtStudy(VocabLossFunct):

    def __init__(self, par_1, par_2, pre_trained_vocab, loss_funct,
                 random_state, verbose, device, out_path_filename,
                 dropout, weights_init, hl_actfunct, out_actfunct,
                 epochs, batch_size, n_hl, units, embedding_type,
                 lr=None, momentum=None):

        super().__init__(par_1, par_2, random_state, verbose, device,
                         out_path_filename, n_hl, dropout, epochs, units,
                         lr, momentum, weights_init, hl_actfunct,
                         out_actfunct, batch_size, embedding_type)

        self.pre_trained_vocab = pre_trained_vocab
        self.loss_funct = loss_funct

    def run(self):

        if self.pre_trained_vocab in pos_tagged:
            pos = True

        else:
            pos = False

        embedding, data, loader = self._load_data(
            vocab=self.pre_trained_vocab,
            pos=pos,
            batch_size=self.batch_size,
            device=self.device
        )

        for c_idx, col in enumerate(self.par_1):
            for r_idx, row in enumerate(self.par_2):
                model, y_pred, t0, t1 = self._fit_predict(
                    data=data,
                    loader=loader,
                    embedding=embedding,
                    n_hl=self.n_hl,
                    num_features=embedding.vector_size,
                    n_classes=2,
                    dropout=self.dropout,
                    epochs=self.epochs,
                    units=self.units,
                    bias=0.1,
                    lr=col,
                    momentum=row,
                    device=self.device,
                    weights_init=self.weights_init,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    loss_funct=self.loss_funct,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    embedding_type=self.embedding_type
                )

                y_test = [torch.LongTensor([y]) for y in data.label_test]
                y_test = torch.stack(y_test)
                y_test = model.transform_target_for_loss(y_test, row)

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
