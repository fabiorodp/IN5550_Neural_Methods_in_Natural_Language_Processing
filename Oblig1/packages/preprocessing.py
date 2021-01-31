# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Christopher Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind
# E-mail:

from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class BOW:

    @staticmethod
    def _reading_data(url: str, verbose: bool) -> pd.DataFrame:

        print("Reading data...") if verbose else None
        df = pd.read_csv(url, sep='\t', header=0, compression='infer')

        return df

    @staticmethod
    def _extracting(df: pd.DataFrame, verbose: bool) -> tuple:

        print("Extracting text and source...") if verbose else None
        text, sources = df.text.values, df.source.values

        return text, sources

    def __init__(self,
                 bow_type: str = "counter",  # options: "binary", "embedded"
                 vocab_size: int = 0,  # 0 means all words as features
                 device: torch.device = torch.device("cpu"),
                 verbose: bool = False,
                 random_state: int = None) -> None:

        # seeding
        torch.manual_seed(random_state)

        # initializing objects
        self.verbose = verbose
        self.device = device
        self.vocab_size = vocab_size

        if bow_type == "counter":
            self.vectorizer_classes = OneHotEncoder()

            if vocab_size == 0:
                self.vectorizer_features = CountVectorizer()

            else:
                self.vectorizer_features = \
                    CountVectorizer(max_features=vocab_size)

        elif bow_type == "binary":
            self.vectorizer_classes = OneHotEncoder()

            if vocab_size == 0:
                self.vectorizer_features = CountVectorizer(binary=True)

            else:
                self.vectorizer_features = \
                    CountVectorizer(max_features=vocab_size, binary=True)

        elif bow_type == "embedded":
            pass

        self.df, self.text, self.sources = None, None, None
        self.features_names, self.classes_names = None, None
        self.input_tensor, self.target = None, None

    def fit_transform(self, url: str):

        self.df = self._reading_data(url, self.verbose)
        self.text, self.sources = self._extracting(self.df, self.verbose)

        print("Creating tensors...") if self.verbose else None
        x = self.text
        y = self.sources.reshape(-1, 1)

        X_train = self.vectorizer_features.fit_transform(x)
        X_train_array = X_train.toarray()

        Y_train = self.vectorizer_classes.fit_transform(y)
        Y_train_array = Y_train.toarray()

        self.get_feature_names()

        self.input_tensor = torch.from_numpy(X_train_array).float()
        self.target = torch.from_numpy(Y_train_array.argmax(axis=1)).long()

        print("Finalized.") if self.verbose else None

    def get_feature_names(self):
        self.features_names = \
            {i for i in self.vectorizer_features.get_feature_names()}

        self.classes_names = \
            {i for i in self.vectorizer_classes.get_feature_names().tolist()}

        return self.features_names, self.classes_names


class TSVDataset(Dataset):

    @staticmethod
    def _reading_data(url: str, verbose: bool) -> pd.DataFrame:
        print("Reading data...") if verbose else None
        df = pd.read_csv(url, sep='\t', header=0, compression='infer')

        return df

    @staticmethod
    def _dev_split(df: pd.DataFrame, prop: float, verbose: bool) -> tuple:
        print("Splitting train and test data...") if verbose else None
        n = len(df) * prop
        train = resample(df, replace=False, stratify=df.source, n_samples=n)
        valid = df[~df.index.isin(train.index)]

        return train, valid

    @staticmethod
    def _extracting(df: pd.DataFrame, train: pd.DataFrame,
                    valid: pd.DataFrame, verbose: bool) -> tuple:
        print("Extracting text and source...") if verbose else None
        text = list(df.text.str.split())
        train_text = list(train.text.str.split())
        valid_text = list(valid.text.str.split())
        source = list(df.source)
        train_source = list(train.source)
        valid_source = list(valid.source)

        return (text, train_text, valid_text, source, train_source,
                valid_source)

    @staticmethod
    def _counting(train_text: list, verbose: bool) -> Counter:
        print("Counting...") if verbose else None
        flat_text = [item for sublist in train_text for item in sublist]
        tokens = Counter(flat_text)

        return tokens

    @staticmethod
    def _storing_vocab(train_source: list, tokens: Counter,
                       vocab_size: int, verbose: bool) -> tuple:
        print("Storing vocabulary counts...") if verbose else None

        if vocab_size == 0:
            vocab_size = len(tokens.keys())

        # get unique words in set of training data
        text_vocab = [i[0] for i in tokens.most_common(vocab_size)]

        # store source_vocab (i.e. titles of classes)
        # all possible sources in training data
        source_vocab = list(set(train_source))

        return text_vocab, source_vocab

    @staticmethod
    def _indexing(text_vocab: list, source_vocab: list, verbose: bool):
        print("Generating indexers...") if verbose else None

        # indexable vocab list (dict with words as key and index as value)
        text_indexer = {i: n for n, i in enumerate(text_vocab)}

        # indexable source list
        source_indexer = {i: n for n, i in enumerate(source_vocab)}

        return text_indexer, source_indexer

    def __init__(self, url: str, vocab_size: int = 0, prop: float = 0.8,
                 device: torch.device = torch.device("cpu"),
                 verbose: bool = False, random_state: int = None) -> None:

        super().__init__()

        # seeding
        torch.manual_seed(random_state)

        # initializing objects
        self.verbose = verbose
        self.vocab_size = vocab_size
        self.prop = prop
        self.device = device

        self.df = self._reading_data(url, self.verbose)

        self.train, self.valid = self._dev_split(
            self.df,
            self.prop,
            self.verbose
        )

        (self.text, self.train_text, self.valid_text, self.source,
         self.train_source, self.valid_source) = self._extracting(
            self.df,
            self.train,
            self.valid,
            self.verbose
        )

        self.tokens = self._counting(self.train_text, self.verbose)

        self.text_vocab, self.source_vocab = self._storing_vocab(
            self.train_source,
            self.tokens,
            self.vocab_size,
            self.verbose
        )

        self.text_indexer, self.source_indexer = self._indexing(
            self.text_vocab,
            self.source_vocab,
            self.verbose
        )

        print("Data initialized.") if self.verbose else None

    def __getitem__(self, index: int) -> tuple:
        """
        Magic method called when using [ ] indexing on class object.

        Current features are:
            0: counts (bag of words)
            1: binary (exists in bow)
            2: what should be the third feature?

        Return: tuple (X, y)
            X: all features as tensor
            y: this items gold source as tensor
        """
        # load text and source from data loaded
        current_text = self.text[index]

        # current_text_list = current_text.split()
        current_source = self.source[index]

        # define features
        feature_count = 2
        feature = list(range(feature_count))

        # Vectorized b.o.w.
        feature[0] = [self.tokens[word] if word in current_text else 0 for
                      word in self.text_vocab]

        # Binary Vectorized b.o.w.
        feature[1] = list(np.where(np.array(feature[0]) > 0, 1, 0))

        # third feature..? word embeddings?
        # feature[2] = None

        # build tensors
        X = torch.FloatTensor(feature)
        y = self.source_indexer[current_source]
        y = torch.LongTensor([y])

        return X, y

    def __len__(self):
        """
        Magic method to return the number of samples.
        """
        return len(self.text)


def source_dist_plot(data):
    plt.figure(figsize=(8, 10))
    c = 1
    for dataset in data:
        plt.subplot(len(data), 1, c)
        dataset.source.value_counts(normalize=True).plot(kind='barh')
        c += 1


def source_dist_prop(data, index=None):
    table = []
    for dataset in data:
        table.append(dataset.source.value_counts(normalize=True))
    if not index:
        index = [i for i in range(len(data))]
    return pd.DataFrame(table, index=index)
