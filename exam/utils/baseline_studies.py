# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from utils.datasets import Vocab, ConllDataset
    from utils.wordembs import WordVecs
    from utils.models import BiLSTM
except:
    from exam.utils.datasets import Vocab, ConllDataset
    from exam.utils.wordembs import WordVecs
    from exam.utils.models import BiLSTM

from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import time


class EMBEDDINGvsTRAIN_EMBEDDINGS:

    @staticmethod
    def _generate_metrics(par_1, par_2):

        training_time = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_binary_f1 = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        searching_propor_f1 = np.empty(
            shape=(len(par_1), len(par_2)),
            dtype=float
        )

        return training_time, searching_binary_f1, searching_propor_f1

    @staticmethod
    def _load_data(EMBEDDINGS, TRAIN_DATA, DEV_DATA, TEST_DATA,
                   BATCH_SIZE, EMBEDDINGS_DIR=None):
        """
        EMBEDDINGS_DIR: "/cluster/shared/nlpl/data/vectors/latest/",
        EMBEDDINGS: "58.zip",
        TRAIN_DATA: "data/train.conll"
        DEV_DATA: "data/dev.conll"
        TEST_DATA: "data/test.conll"
        BATCH_SIZE: 32
        """
        # can load dir from saga
        if EMBEDDINGS_DIR:
            EMBEDDINGS = EMBEDDINGS_DIR+EMBEDDINGS+".zip"

        file_type = EMBEDDINGS.split('.')[-1]
        # Get embeddings
        embeddings = WordVecs(EMBEDDINGS, file_type)

        # make sure embeddings dimensions match model file
        EMBEDDING_DIM = embeddings._matrix.shape[1]

        # Create shared vocabulary for tasks
        vocab = Vocab(train=True)

        # ???
        with_unk = {}
        for word, idx in embeddings._w2idx.items():
            with_unk[word] = idx + 2
        vocab.update(with_unk)

        # Import datasets
        # This will update vocab with words not found in embeddings
        dataset = ConllDataset(vocab)

        train_iter = dataset.get_split(TRAIN_DATA)
        dev_iter = dataset.get_split(DEV_DATA)
        test_iter = dataset.get_split(TEST_DATA)

        # Create a new embedding matrix which includes the pretrained
        # embeddings as well as new embeddings for PAD UNK and tokens not
        # found in the pretrained embeddings.
        diff = len(vocab) - embeddings.vocab_length - 2
        PAD_UNK_embeddings = np.zeros((2, EMBEDDING_DIM))
        new_embeddings = np.zeros((diff, EMBEDDING_DIM))
        new_matrix = np.concatenate(
            (PAD_UNK_embeddings, embeddings._matrix, new_embeddings))

        # Set up the data iterators for the LSTM model.
        # The batch size for the dev and test loader is set to 1 for the
        # predict() and evaluate() methods
        train_loader = DataLoader(train_iter,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=train_iter.collate_fn,
                                  shuffle=True)

        dev_loader = DataLoader(dev_iter,
                                batch_size=1,
                                collate_fn=dev_iter.collate_fn,
                                shuffle=False)

        test_loader = DataLoader(test_iter,
                                 batch_size=1,
                                 collate_fn=test_iter.collate_fn,
                                 shuffle=False)

        return vocab, new_matrix, train_loader, dev_loader, test_loader

    @staticmethod
    def _fit_predict(vocab, new_matrix, EMBEDDING_DIM, HIDDEN_DIM, device,
                     output_dim, NUM_LAYERS, DROPOUT, LEARNING_RATE,
                     TRAIN_EMBEDDINGS, EPOCHS, train_loader, dev_loader,
                     test_loader, verbose):

        model = BiLSTM(word2idx=vocab,
                       embedding_matrix=new_matrix,
                       embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM,
                       device=device,
                       output_dim=output_dim,
                       num_layers=NUM_LAYERS,
                       word_dropout=DROPOUT,
                       learning_rate=LEARNING_RATE,
                       train_embeddings=TRAIN_EMBEDDINGS)

        print("Fitting data...") if verbose else None
        t0 = time.time()
        model.fit(train_loader, dev_loader, epochs=EPOCHS)
        t1 = time.time()
        print(f"Fitting done in {t1 - t0}s.") if verbose else None

        binary_f1, propor_f1 = model.evaluate(test_loader)
        return model, binary_f1, propor_f1, t0, t1

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
        # best parameters according to searching_propor_f1
        best_1 = np.argmax(self.searching_propor_f1, axis=0)
        best_2 = np.argmax(self.searching_propor_f1, axis=1)
        return (self.par_1[best_1[0]], self.par_2[best_2[0]])

    def __init__(self, par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                 out_path_filename, verbose, random_state, BATCH_SIZE,
                 HIDDEN_DIM, device, output_dim, NUM_LAYERS, DROPOUT,
                 LEARNING_RATE, EPOCHS, EMBEDDINGS_DIR=None,
                 EMBEDDINGS=None, TRAIN_EMBEDDINGS=None):

        # seeding
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # searching parameters:
        self.par_1 = par_1
        self.par_2 = par_2

        # creating matrices to store the searching scores
        (self.training_time, self.searching_binary_f1,
         self.searching_propor_f1) = self._generate_metrics(
            par_1=par_1,
            par_2=par_2
        )

        # global parameters
        self.EMBEDDINGS_DIR = EMBEDDINGS_DIR
        self.EMBEDDINGS = EMBEDDINGS
        self.TRAIN_DATA = TRAIN_DATA
        self.DEV_DATA = DEV_DATA
        self.TEST_DATA = TEST_DATA
        self.out_path_filename = out_path_filename
        self.verbose = verbose
        self.BATCH_SIZE = BATCH_SIZE
        self.HIDDEN_DIM = HIDDEN_DIM
        self.device = device
        self.output_dim = output_dim
        self.NUM_LAYERS = NUM_LAYERS
        self.DROPOUT = DROPOUT
        self.LEARNING_RATE = LEARNING_RATE
        self.TRAIN_EMBEDDINGS = TRAIN_EMBEDDINGS
        self.EPOCHS = EPOCHS

    def _store_metrics(self, c_idx, r_idx, t0, t1, binary_f1, propor_f1):
        print(f"Time: {t1 - t0}.") if self.verbose is True else None
        self.training_time[c_idx][r_idx] = t1 - t0

        print(f"Binary F1: {binary_f1}.") if self.verbose is True else None
        self.searching_binary_f1[c_idx][r_idx] = binary_f1

        print(f"Propor F1: {propor_f1}.") if self.verbose is True else None
        self.searching_propor_f1[c_idx][r_idx] = propor_f1

    def run(self):

        for c_idx, col in enumerate(self.par_1):  # EMBEDDING

            # cleaning cuda's cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            vocab, new_matrix, train_loader, dev_loader, test_loader = \
                self._load_data(
                    EMBEDDINGS=col,
                    TRAIN_DATA=self.TRAIN_DATA,
                    DEV_DATA=self.DEV_DATA,
                    TEST_DATA=self.TEST_DATA,
                    BATCH_SIZE=self.BATCH_SIZE,
                    EMBEDDINGS_DIR=self.EMBEDDINGS_DIR,
                )

            for r_idx, row in enumerate(self.par_2):  # TRAIN_EMBEDDINGS

                model, binary_f1, propor_f1, t0, t1 = self._fit_predict(
                    vocab=vocab,
                    new_matrix=new_matrix,
                    EMBEDDING_DIM=new_matrix.shape[1],
                    HIDDEN_DIM=self.HIDDEN_DIM,
                    device=self.device,
                    output_dim=self.output_dim,
                    NUM_LAYERS=self.NUM_LAYERS,
                    DROPOUT=self.DROPOUT,
                    LEARNING_RATE=self.LEARNING_RATE,
                    TRAIN_EMBEDDINGS=row,
                    EPOCHS=self.EPOCHS,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    verbose=self.verbose
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    binary_f1=binary_f1,
                    propor_f1=propor_f1
                )

        metrics_ = (self.training_time, self.searching_binary_f1,
                    self.searching_propor_f1)

        self._save_metrics(
            metrics=metrics_,
            par_1=self.par_1,
            par_2=self.par_2,
            path_filename=self.out_path_filename
        )

        return self


class NUM_LAYERSvsHIDDEN_DIM(EMBEDDINGvsTRAIN_EMBEDDINGS):

    def __init__(self, par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                 out_path_filename, verbose, random_state, BATCH_SIZE,
                 device, output_dim, DROPOUT, LEARNING_RATE, TRAIN_EMBEDDINGS,
                 EPOCHS, EMBEDDINGS, NUM_LAYERS=None, HIDDEN_DIM=None,
                 EMBEDDINGS_DIR=None):

        super().__init__(par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                         out_path_filename, verbose, random_state, BATCH_SIZE,
                         HIDDEN_DIM, device, output_dim, NUM_LAYERS, DROPOUT,
                         LEARNING_RATE, TRAIN_EMBEDDINGS, EPOCHS,
                         EMBEDDINGS_DIR, EMBEDDINGS)

    def run(self):

        # cleaning cuda's cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        vocab, new_matrix, train_loader, dev_loader, test_loader = \
            self._load_data(
                EMBEDDINGS=self.EMBEDDINGS,
                TRAIN_DATA=self.TRAIN_DATA,
                DEV_DATA=self.DEV_DATA,
                TEST_DATA=self.TEST_DATA,
                BATCH_SIZE=self.BATCH_SIZE,
                EMBEDDINGS_DIR=self.EMBEDDINGS_DIR,
            )

        for c_idx, col in enumerate(self.par_1):  # NUM_LAYERS
            for r_idx, row in enumerate(self.par_2):  # HIDDEN_DIM

                model, binary_f1, propor_f1, t0, t1 = self._fit_predict(
                    vocab=vocab,
                    new_matrix=new_matrix,
                    EMBEDDING_DIM=new_matrix.shape[1],
                    HIDDEN_DIM=row,
                    device=self.device,
                    output_dim=self.output_dim,
                    NUM_LAYERS=col,
                    DROPOUT=self.DROPOUT,
                    LEARNING_RATE=self.LEARNING_RATE,
                    TRAIN_EMBEDDINGS=self.TRAIN_EMBEDDINGS,
                    EPOCHS=self.EPOCHS,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    verbose=self.verbose
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    binary_f1=binary_f1,
                    propor_f1=propor_f1
                )

            metrics_ = (self.training_time, self.searching_binary_f1,
                        self.searching_propor_f1)

            self._save_metrics(
                metrics=metrics_,
                par_1=self.par_1,
                par_2=self.par_2,
                path_filename=self.out_path_filename
            )

            return self


class LEARNING_RATEvsDROPOUT(EMBEDDINGvsTRAIN_EMBEDDINGS):

    def __init__(self, par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                 out_path_filename, verbose, random_state, BATCH_SIZE,
                 device, output_dim, TRAIN_EMBEDDINGS, EPOCHS, EMBEDDINGS,
                 NUM_LAYERS, HIDDEN_DIM, LEARNING_RATE=None,
                 DROPOUT=None, EMBEDDINGS_DIR=None):

        super().__init__(par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                         out_path_filename, verbose, random_state, BATCH_SIZE,
                         HIDDEN_DIM, device, output_dim, NUM_LAYERS, DROPOUT,
                         LEARNING_RATE, TRAIN_EMBEDDINGS, EPOCHS,
                         EMBEDDINGS_DIR, EMBEDDINGS)

    def run(self):

        # cleaning cuda's cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        vocab, new_matrix, train_loader, dev_loader, test_loader = \
            self._load_data(
                EMBEDDINGS=self.EMBEDDINGS,
                TRAIN_DATA=self.TRAIN_DATA,
                DEV_DATA=self.DEV_DATA,
                TEST_DATA=self.TEST_DATA,
                BATCH_SIZE=self.BATCH_SIZE,
                EMBEDDINGS_DIR=self.EMBEDDINGS_DIR,
            )

        for c_idx, col in enumerate(self.par_1):  # LEARNING_RATE
            for r_idx, row in enumerate(self.par_2):  # DROPOUT

                model, binary_f1, propor_f1, t0, t1 = self._fit_predict(
                    vocab=vocab,
                    new_matrix=new_matrix,
                    EMBEDDING_DIM=new_matrix.shape[1],
                    HIDDEN_DIM=self.HIDDEN_DIM,
                    device=self.device,
                    output_dim=self.output_dim,
                    NUM_LAYERS=self.NUM_LAYERS,
                    DROPOUT=row,
                    LEARNING_RATE=col,
                    TRAIN_EMBEDDINGS=self.TRAIN_EMBEDDINGS,
                    EPOCHS=self.EPOCHS,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    verbose=self.verbose
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    binary_f1=binary_f1,
                    propor_f1=propor_f1
                )

            metrics_ = (self.training_time, self.searching_binary_f1,
                        self.searching_propor_f1)

            self._save_metrics(
                metrics=metrics_,
                par_1=self.par_1,
                par_2=self.par_2,
                path_filename=self.out_path_filename
            )

            return self


class BATCH_SIZEvsEPOCHS(EMBEDDINGvsTRAIN_EMBEDDINGS):

    def __init__(self, par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                 out_path_filename, verbose, random_state,
                 device, output_dim, TRAIN_EMBEDDINGS, EMBEDDINGS,
                 NUM_LAYERS, HIDDEN_DIM, LEARNING_RATE, DROPOUT,
                 BATCH_SIZE=None, EPOCHS=None, EMBEDDINGS_DIR=None):

        super().__init__(par_1, par_2, TRAIN_DATA, DEV_DATA, TEST_DATA,
                         out_path_filename, verbose, random_state, BATCH_SIZE,
                         HIDDEN_DIM, device, output_dim, NUM_LAYERS, DROPOUT,
                         LEARNING_RATE, TRAIN_EMBEDDINGS, EPOCHS,
                         EMBEDDINGS_DIR, EMBEDDINGS)

    def run(self):

        for c_idx, col in enumerate(self.par_1):  # BATCH_SIZE

            # cleaning cuda's cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            vocab, new_matrix, train_loader, dev_loader, test_loader = \
                self._load_data(
                    EMBEDDINGS=self.EMBEDDINGS,
                    TRAIN_DATA=self.TRAIN_DATA,
                    DEV_DATA=self.DEV_DATA,
                    TEST_DATA=self.TEST_DATA,
                    BATCH_SIZE=col,
                    EMBEDDINGS_DIR=self.EMBEDDINGS_DIR,
                )

            for r_idx, row in enumerate(self.par_2):  # EPOCHS

                model, binary_f1, propor_f1, t0, t1 = self._fit_predict(
                    vocab=vocab,
                    new_matrix=new_matrix,
                    EMBEDDING_DIM=new_matrix.shape[1],
                    HIDDEN_DIM=self.HIDDEN_DIM,
                    device=self.device,
                    output_dim=self.output_dim,
                    NUM_LAYERS=self.NUM_LAYERS,
                    DROPOUT=self.DROPOUT,
                    LEARNING_RATE=self.LEARNING_RATE,
                    TRAIN_EMBEDDINGS=self.TRAIN_EMBEDDINGS,
                    EPOCHS=row,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    verbose=self.verbose
                )

                self._store_metrics(
                    c_idx=c_idx,
                    r_idx=r_idx,
                    t0=t0,
                    t1=t1,
                    binary_f1=binary_f1,
                    propor_f1=propor_f1
                )

            metrics_ = (self.training_time, self.searching_binary_f1,
                        self.searching_propor_f1)

            self._save_metrics(
                metrics=metrics_,
                par_1=self.par_1,
                par_2=self.par_2,
                path_filename=self.out_path_filename
            )

            return self
