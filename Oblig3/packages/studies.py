# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.model import Transformer, BertRNN, BertMLP

except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.model import Transformer, BertRNN, BertMLP

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import torch
import time


class FreezerStudy:
    @staticmethod
    def _generate_metrics(par_1, par_2):

        training_time = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_f1Score = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_epoch = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        return (training_time, searching_f1Score, searching_epoch)

    @staticmethod
    def _load_data(NORBERT, train_data_url, batch_size, device,
                   train_prop, verbose, random_state, min_entities):

        print('\n\n\nload tokenizer...') if verbose else None
        tokenizer = BertTokenizer.from_pretrained(NORBERT)

        print('\n\n\nloading data set...') if verbose else None
        # loading raw data
        con_df = load_raw_data(datapath=train_data_url)
        con_df = filter_raw_data(df=con_df, min_entities=min_entities)

        # splitting data
        train_df, val_df = train_test_split(
            con_df,
            test_size=1 - train_prop,
            random_state=random_state,
            shuffle=True,
        )

        # creating data sets
        train_dataset = OurCONLLUDataset(
            df=train_df,
            tokenizer=tokenizer,
            device=device
        )

        val_dataset = OurCONLLUDataset(
            df=val_df,
            tokenizer=tokenizer,
            label_vocab=train_dataset.label_vocab,
            device=device
        )
        del train_df, val_df, con_df

        print('\n\n\nloading data loader...') if verbose else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
        )

        num_labels = len(train_dataset.label_indexer)
        NOT_ENTITY_ID = train_dataset.label_indexer['O']

        return (train_loader, val_loader, num_labels, NOT_ENTITY_ID)

    @staticmethod
    def _save_metrics(metrics, par_1, par_2, path_filename):

        for idx, metric in enumerate(metrics):
            df = pd.DataFrame(
                metric,
                index=par_1,
                columns=par_2
            )

            df.to_csv(f'{path_filename}_{idx}.csv')

    @staticmethod
    def _freezer(model, keyword, max_count=1):
        if keyword is None:
            return model
        
        count = 0
        if type(keyword) == list:
            for (i, j) in model.named_parameters():
                for kwrd in keyword:
                    if kwrd in i:
                        j.requires_grad = False
                        count += 1
                    if count >= max_count:
                        break
        else:
            for (i, j) in model.named_parameters():
                if keyword in i:
                    j.requires_grad = False
                    count += 1
                if count >= max_count:
                    break
        return model

    def __init__(self, par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, verbose, random_state,
                 epochs, device, loss_funct, batch_size, freeze,
                 lr_scheduler, factor, patience, min_entities,
                 lr, momentum, epoch_patience):

        # searching parameters:
        self.par_1 = par_1  # list of key words to freeze
        self.par_2 = par_2  # max number of layers to freeze

        # creating matrices to store the searching scores
        self.training_time, self.searching_f1Score, self.searching_epoch = \
            self._generate_metrics(
                par_1=par_1,
                par_2=par_2
            )

        self.vocab_dir = vocab_dir
        self.train_data_url = train_data_url
        self.out_path_filename = out_path_filename
        self.verbose = verbose

        # seeding
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.epochs = epochs
        self.device = device
        self.loss_funct = loss_funct
        self.batch_size = batch_size
        self.freeze = freeze
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.min_entities = min_entities
        self.lr = lr
        self.momentum = momentum
        self.epoch_patience = epoch_patience

    def _store_metrics(self, c_idx, r_idx, t0, t1, model):

        print(f"Elapsed time: {t1 - t0}.") if self.verbose is True else None
        self.training_time[c_idx][r_idx] = t1 - t0

        print(f"Valid F1: {model.val_f1_scores[model.early_stop_epoch]}.") \
            if self.verbose is True else None
        self.searching_f1Score[c_idx][r_idx] = \
            model.val_f1_scores[model.early_stop_epoch]

        self.searching_epoch[c_idx][r_idx] = model.early_stop_epoch

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
         NOT_ENTITY_ID) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # keyword
            for r_idx, row in enumerate(self.par_2):  # num frozen layers

                print('\n\n\nloading model...') if self.verbose else None
                model = Transformer(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    factor=self.factor,
                    patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience
                )

                model = self._freezer(model, col, row)

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


############  BertRNN studies  ############
class RNNTypeFineTuneBert:
    @staticmethod
    def _generate_metrics(par_1, par_2):

        training_time = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_f1Score = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_epoch = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        return (training_time, searching_f1Score, searching_epoch)

    @staticmethod
    def _load_data(NORBERT, train_data_url, batch_size, device, train_prop, 
                   verbose, random_state, min_entities, small_data=None):

        print('\n\n\nload tokenizer...') if verbose else None
        tokenizer = BertTokenizer.from_pretrained(NORBERT)

        print('\n\n\nloading data set...') if verbose else None
        # loading raw data
        con_df = load_raw_data(datapath=train_data_url)
        con_df = filter_raw_data(df=con_df, min_entities=min_entities)
        if small_data:
            con_df, _ = train_test_split(con_df, train_size=small_data)
            del _

        # splitting data
        train_df, val_df = train_test_split(
            con_df,
            test_size=1 - train_prop,
            random_state=random_state,
            shuffle=True,
        )

        # creating data sets
        train_dataset = OurCONLLUDataset(
            df=train_df,
            tokenizer=tokenizer,
            device=device
        )

        val_dataset = OurCONLLUDataset(
            df=val_df,
            tokenizer=tokenizer,
            label_vocab=train_dataset.label_vocab,
            device=device
        )
        del train_df, val_df, con_df

        print('\n\n\nloading data loader...') if verbose else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
        )

        num_labels = len(train_dataset.label_indexer)
        NOT_ENTITY_ID = train_dataset.label_indexer['O']

        return (train_loader, val_loader, num_labels, 
                NOT_ENTITY_ID, train_dataset.label_indexer)

    @staticmethod
    def _save_metrics(metrics, par_1, par_2, path_filename):

        for idx, metric in enumerate(metrics):
            df = pd.DataFrame(
                metric,
                index=par_1,
                columns=par_2
            )

            df.to_csv(f'{path_filename}_{idx}.csv')

    def __init__(
        self, par_1, par_2, 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=6,            # hidden layers of rnn
        units=512,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        rnn_type='gru',    # rnn, lstm, gru
        nonlinearity: str = 'tanh',    # tanh, relu, ..?
        pool_type: str = 'cat',        # TODO check if this is even used
        bidirectional: bool = True,
        fine_tune_bert=False, 
        small_data=None,   
        ):
        
        # searching parameters:
        self.par_1 = par_1  # list of rnn types
        self.par_2 = par_2  # fine tune bert true or false

        # creating matrices to store the searching scores
        self.training_time, self.searching_f1Score, self.searching_epoch = \
            self._generate_metrics(
                par_1=par_1,
                par_2=par_2
            )

        self.vocab_dir = vocab_dir                  # NORBERT
        self.train_data_url = train_data_url        # datafile
        self.out_path_filename = out_path_filename  # outputs/RNNBiDir
        self.verbose = verbose
        self.small_data = small_data

        # seeding
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.epochs = epochs
        self.device = device
        self.loss_funct = loss_funct
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.factor = lr_factor
        self.patience = lr_patience
        self.min_entities = min_entities
        self.lr = lr
        self.momentum = momentum
        self.epoch_patience = epoch_patience
        self.n_hl = n_hl                        # <- num_layers
        self.input_size = input_size            # <- input_size
        self.dropout = dropout
        self.input_size = input_size
        self.units = units
        self.pool_type = pool_type
        self.rnn_type = rnn_type
        self.nonlinearity = nonlinearity
        self.bidirectional = bidirectional
        self.fine_tune_bert = fine_tune_bert

    def _store_metrics(self, c_idx, r_idx, t0, t1, model):

        print(f"Elapsed time: {t1 - t0}.") if self.verbose is True else None
        self.training_time[c_idx][r_idx] = t1 - t0

        print(f"Valid F1: {model.val_f1_scores[model.last_epoch]}.") \
            if self.verbose is True else None
        self.searching_f1Score[c_idx][r_idx] = \
            model.val_f1_scores[model.last_epoch]

        self.searching_epoch[c_idx][r_idx] = model.last_epoch

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
         NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities,
            small_data=self.small_data,
        )

        for c_idx, col in enumerate(self.par_1):  # rnn type
            for r_idx, row in enumerate(self.par_2):  # fine tune bert

                print('\n\n\nloading model...') if self.verbose else None
                model = BertRNN(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=self.n_hl,
                    units=self.units,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    rnn_type=col,
                    nonlinearity=self.nonlinearity,
                    pool_type=self.pool_type,
                    bidirectional=self.bidirectional,
                    fine_tune_bert=row,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self # want to return best params

class RNNHLUnits(RNNTypeFineTuneBert):

    def __init__(
        self, 
        par_1, # hidden layers 
        par_2, # Units 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=None,            # hidden layers of rnn
        units=None,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        rnn_type='gru',    # rnn, lstm, gru
        nonlinearity= 'tanh',    # tanh, relu, ..?
        pool_type= 'cat',        # TODO check if this is even used
        bidirectional = True, 
        fine_tune_bert = False,
        small_data=None
    ):

        super().__init__(
            par_1, par_2, vocab_dir, train_data_url,
            out_path_filename, verbose, random_state,
            epochs, device, loss_funct, batch_size, 
            lr_scheduler, lr_factor, lr_patience, min_entities,
            lr, momentum, epoch_patience,
            n_hl, units, dropout, input_size, rnn_type, 
            nonlinearity, pool_type, bidirectional, 
            fine_tune_bert, small_data
        )

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
         NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # rnn type
            for r_idx, row in enumerate(self.par_2):  # fine tune bert

                print('\n\n\nloading model...') if self.verbose else None
                model = BertRNN(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=col,
                    units=row,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    rnn_type=self.rnn_type,
                    nonlinearity=self.nonlinearity,
                    pool_type=self.pool_type,
                    bidirectional=self.bidirectional,
                    fine_tune_bert=self.fine_tune_bert,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self

class RNNLrMmt(RNNTypeFineTuneBert):
    def __init__(self, 
                par_1, # hidden layers 
                par_2, # Units 
                vocab_dir, train_data_url,
                out_path_filename, verbose, random_state,
                epochs, device, loss_funct, batch_size, 
                lr_scheduler, lr_factor, lr_patience, min_entities,
                lr, momentum, epoch_patience,
                n_hl=None,            # hidden layers of rnn
                units=None,         # units per layer of rnn (default BERT val)
                dropout=0.1,       # rnn dropout
                input_size=768,    # make sure model is BertModel not ForToken 
                rnn_type='gru',    # rnn, lstm, gru
                nonlinearity= 'tanh',    # tanh, relu, ..?
                pool_type= 'cat',        # TODO check if this is even used
                bidirectional = True, 
                fine_tune_bert = False,
                small_data = None
    ):

        super().__init__(par_1, par_2, vocab_dir, train_data_url,
                 out_path_filename, verbose, random_state,
                 epochs, device, loss_funct, batch_size, 
                 lr_scheduler, lr_factor, lr_patience, min_entities,
                 lr, momentum, epoch_patience,
                 n_hl, units, dropout, input_size, rnn_type, 
                 nonlinearity, pool_type, bidirectional, 
                 fine_tune_bert, small_data)

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
         NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # rnn type
            for r_idx, row in enumerate(self.par_2):  # fine tune bert

                print('\n\n\nloading model...') if self.verbose else None
                model = BertRNN(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=col,
                    momentum=row,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=self.n_hl,
                    units=self.units,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    rnn_type=self.rnn_type,
                    nonlinearity=self.nonlinearity,
                    pool_type=self.pool_type,
                    bidirectional=self.bidirectional,
                    fine_tune_bert=self.fine_tune_bert,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self



############ BertMLP Studies ##############
class MLPFilterSizeFineTuneBert(RNNTypeFineTuneBert):

    def __init__(
        self, par_1, par_2, 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=6,            # hidden layers of rnn
        units=512,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        fine_tune_bert=False,    
        small_data = 0.25,

        ### change these for MLP specific parameters
        bias=0.1,
        hl_actfunct='tanh',
        out_actfunct='relu',
        weights_init='xavier_normal',
    ):

        # searching parameters:
        self.par_1 = par_1  # list of rnn types
        self.par_2 = par_2  # fine tune bert true or false

        # creating matrices to store the searching scores
        self.training_time, self.searching_f1Score, self.searching_epoch = \
            self._generate_metrics(
                par_1=par_1,
                par_2=par_2
            )

        self.vocab_dir = vocab_dir                  # NORBERT
        self.train_data_url = train_data_url        # datafile
        self.out_path_filename = out_path_filename  
        self.verbose = verbose
        self.small_data = small_data

        # seeding
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.epochs = epochs
        self.device = device
        self.loss_funct = loss_funct
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.factor = lr_factor
        self.patience = lr_patience
        self.min_entities = min_entities
        self.lr = lr
        self.momentum = momentum
        self.epoch_patience = epoch_patience
        self.n_hl = n_hl                        # <- num_layers
        self.input_size = input_size            # <- input_size
        self.dropout = dropout
        self.input_size = input_size
        self.units = units
        self.fine_tune_bert = fine_tune_bert

        # change to MLP specific 
        self.bias=bias
        self.hl_actfunct=hl_actfunct
        self.out_actfunct=out_actfunct
        self.weights_init=weights_init
    
    def run(self):

        for c_idx, col in enumerate(self.par_1):  # min_entities
            for r_idx, row in enumerate(self.par_2):  # fine tune bert

                # load data
                (train_loader, val_loader, num_labels,
                NOT_ENTITY_ID, label_indexer) = self._load_data(
                    NORBERT=self.vocab_dir,
                    train_data_url=self.train_data_url,
                    batch_size=self.batch_size,
                    device=self.device,
                    train_prop=0.75,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    min_entities=col,
                    small_data=self.small_data
                )

                print('\n\n\nloading model...') if self.verbose else None
                model = BertMLP(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=self.n_hl,
                    units=self.units,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    fine_tune_bert=row,

                    # MLP
                    bias=self.bias,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    weights_init=self.weights_init,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self # want to return best params

class MLPActivationFunction(MLPFilterSizeFineTuneBert):

    def __init__(
        self, par_1, par_2, 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=6,            # hidden layers of rnn
        units=512,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        fine_tune_bert=False,    
        small_data = 0.25,

        ### MLP specific parameters
        bias=0.1,
        hl_actfunct='tanh',
        out_actfunct='relu',
        weights_init='xavier_normal'
    ):

        super().__init__(
            par_1, par_2, 
            vocab_dir, train_data_url,
            out_path_filename, verbose, random_state,
            epochs, device, loss_funct, batch_size, 
            lr_scheduler, lr_factor, lr_patience, min_entities,
            lr, momentum, epoch_patience,
            n_hl, units, dropout, input_size, fine_tune_bert, small_data,  
            bias, hl_actfunct, out_actfunct, weights_init, 
        )

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
        NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # hidden activation 
            for r_idx, row in enumerate(self.par_2):  # output activation


                print('\n\n\nloading model...') if self.verbose else None
                model = BertMLP(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=self.n_hl,
                    units=self.units,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    fine_tune_bert=True,

                    # MLP
                    bias=self.bias,
                    hl_actfunct=col,
                    out_actfunct=row,
                    weights_init=self.weights_init,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self # want to return best params

class MLPHlUnits(MLPFilterSizeFineTuneBert):

    def __init__(
        self, par_1, par_2, 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=6,            # hidden layers of rnn
        units=512,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        fine_tune_bert=False,    
        small_data = 0.25,

        ### MLP specific parameters
        bias=0.1,
        hl_actfunct='tanh',
        out_actfunct='relu',
        weights_init='xavier_normal'
    ):

        super().__init__(
            par_1, par_2, 
            vocab_dir, train_data_url,
            out_path_filename, verbose, random_state,
            epochs, device, loss_funct, batch_size, 
            lr_scheduler, lr_factor, lr_patience, min_entities,
            lr, momentum, epoch_patience,
            n_hl, units, dropout, input_size, fine_tune_bert, small_data ,  
            bias, hl_actfunct, out_actfunct, weights_init, 
        )

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
        NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # hidden layers 
            for r_idx, row in enumerate(self.par_2):  # units per layer


                print('\n\n\nloading model...') if self.verbose else None
                model = BertMLP(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=self.lr,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=col,
                    units=row,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    fine_tune_bert=True,

                    # MLP
                    bias=self.bias,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    weights_init=self.weights_init,
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self # want to return best params

class MLPLrOpt(MLPFilterSizeFineTuneBert):

    def __init__(
        self, par_1, par_2, 
        vocab_dir, train_data_url,
        out_path_filename, verbose, random_state,
        epochs, device, loss_funct, batch_size, 
        lr_scheduler, lr_factor, lr_patience, min_entities,
        lr, momentum, epoch_patience,
        n_hl=6,            # hidden layers of rnn
        units=512,         # units per layer of rnn (default BERT val)
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForToken 
        fine_tune_bert=False,    
        small_data = 0.25,

        ### MLP specific parameters
        bias=0.1,
        hl_actfunct='tanh',
        out_actfunct='relu',
        weights_init='xavier_normal'
    ):

        super().__init__(
            par_1, par_2, 
            vocab_dir, train_data_url,
            out_path_filename, verbose, random_state,
            epochs, device, loss_funct, batch_size, 
            lr_scheduler, lr_factor, lr_patience, min_entities,
            lr, momentum, epoch_patience,
            n_hl, units, dropout, input_size, fine_tune_bert, small_data,  
            bias, hl_actfunct, out_actfunct, weights_init, 
        )

    def run(self):

        # load data
        (train_loader, val_loader, num_labels,
        NOT_ENTITY_ID, label_indexer) = self._load_data(
            NORBERT=self.vocab_dir,
            train_data_url=self.train_data_url,
            batch_size=self.batch_size,
            device=self.device,
            train_prop=0.75,
            verbose=self.verbose,
            random_state=self.random_state,
            min_entities=self.min_entities
        )

        for c_idx, col in enumerate(self.par_1):  # learning rate 
            for r_idx, row in enumerate(self.par_2):  # optimizer


                print('\n\n\nloading model...') if self.verbose else None
                model = BertMLP(
                    NORBERT=self.vocab_dir,
                    num_labels=num_labels,
                    NOT_ENTITY_ID=NOT_ENTITY_ID,
                    device=self.device,
                    epochs=self.epochs,
                    lr_scheduler=self.lr_scheduler,
                    lr_factor=self.factor,
                    lr_patience=self.patience,
                    loss_funct=self.loss_funct,
                    random_state=self.random_state,
                    verbose=self.verbose,
                    lr=col,
                    momentum=self.momentum,
                    epoch_patience=self.epoch_patience,
                    label_indexer=label_indexer,
                    n_hl=self.n_hl,
                    units=self.units,
                    dropout=self.dropout,
                    input_size=self.input_size,
                    fine_tune_bert=True,

                    # MLP
                    bias=self.bias,
                    hl_actfunct=self.hl_actfunct,
                    out_actfunct=self.out_actfunct,
                    weights_init=self.weights_init,
                    optimizer=row
                )

                print("Fitting data...") if self.verbose else None
                t0 = time.time()
                model.fit(
                    loader=train_loader,
                    test=val_loader,
                    verbose=self.verbose
                )
                t1 = time.time()
                print(f"Fitting done in {t1 - t0}s.") if self.verbose else \
                    None

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    model=model
                )

        metrics_ = (self.training_time, self.searching_f1Score)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self # want to return best params

