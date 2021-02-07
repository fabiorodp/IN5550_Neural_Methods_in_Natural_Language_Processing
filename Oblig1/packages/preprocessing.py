# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


class Signal20Dataset(Dataset):
    # NOTE: This class is never used. See line 373 for BOW preprocessor
    # "static" variables (once set, holds for all future inits of class)
    # validation indexes
    v_idx = None
    one = None
    source_indexer = None
    # features
    count = None
    binry = None
    tfidf = None

    @staticmethod
    def _read_data(url: str, verbose: bool) -> pd.DataFrame:
        """
        Reads the data specified in the url into a pandas dataframe.
        ______________________________________________________________
        Parameters
        url: str
            The path to the file to be read
        verbose: bool
            if True: prints progressive output
        ______________________________________________________________
        Returns:
        df: pd.DataFrame
            Dataframe containing the read data
        """
        print("Reading data...") if verbose else None
        df = pd.read_csv(url, sep='\t', compression='infer')

        return df

    @staticmethod
    def _dev_split(df: pd.DataFrame, prop: float, train: bool,
                   verbose: bool):
        """
        Splits the dataframe into training- and validation data in
        the proportion spcified by prop.
        ______________________________________________________________
        Parameters
        df: pd.DataFrame
            The data to be split
        prop: float
            The proportion in which to split the data
        train: bool
            If True: resamples the data to the proportion
        verbose: bool
            if True: print progressive output
        ______________________________________________________________
        Returns:
        this: pd.DataFrame
            A dataframe split in the specified proportion for train
            and test sets.
        """
        print("Splitting train and test data...") if verbose else None
        # get count for proportion
        n = len(df) * prop

        # build return df
        if train:
            # split train/valid
            train = resample(
                df,
                replace=False,
                stratify=df.source,
                n_samples=n
            )

            valid = df[~df.index.isin(train.index)]

            # set valid indexes for future dataset inits
            Signal20Dataset.v_idx = valid.index

            # df to return
            this = train

        else:
            # df to return
            this = df[df.index.isin(Signal20Dataset.v_idx)]

        return this

    @staticmethod
    def _extract(df: pd.DataFrame, verbose: bool):
        """
        Extracts the text and source from the dataframe and returns
        them individually.
        ______________________________________________________________
        Parameters
        df: pd.DataFrame
            The dataframe to operate on
        verbose:
            if True: prints progressive output
        ______________________________________________________________
        Returns:
        text: list
            The text from the dataframe
        source: pd.DataFrame
            the sources from the dataframe
        """
        print("Extract text...") if verbose else None
        text = list(df.text)
        source = df.source.values.reshape(-1, 1)
        return text, source

    @staticmethod
    def _encode(data: list, encoder: OneHotEncoder, verbose: bool):
        """
        One-hot-encodes the given list.
        ______________________________________________________________
        Parameters
        data: list
            The input data
        encoder: OneHotEncoder
            The OneHotEncoder to use
        verbose: bool
            If True: prints progressive output
        ______________________________________________________________
        Returns:
        data: csr_matrix
            The one-hot-encoded text
        """
        print("OneHotEncode sources...") if verbose else None
        return encoder.transform(data)

    @staticmethod
    def _fit_encode(data: list, verbose: bool):
        """
        Fits the one-hot-encoder on the data.
        ______________________________________________________________
        Parameters
        data: list
            The input data
        verbose: bool
            If true: prints progressive output
        ______________________________________________________________
        Returns:
        one: OneHotEncoder
            The fitted OnehotEncoder
        one.get_feature_names: ndarray
            Array of feature names
        """
        print("Fit OneHotEncoder on sources") if verbose else None
        one = OneHotEncoder()
        one.fit(data)
        return one, one.get_feature_names([''])

    @staticmethod
    def _fit_counter(counter: CountVectorizer,
                     x: pd.Series, verbose: bool) -> dict:
        """
        Fits the CountVectorizer on the training data.
        ______________________________________________________________
        Parameters
        counter: CountVectorizer
            The CountVectorizer to fit
        x: pd.Series
            The data to fit the CountVectorizer on
        verbose: bool
            if True: prints progressive output
        ______________________________________________________________
        Returns:
        counter: dict
            The fitted CountVectorizer
        """
        print("Generating count vector...") if verbose else None
        counter.fit(x)
        return counter

    @staticmethod
    def _get_counter(bow_type: str = '', cntrargs: dict = {}):
        """
        Chooses the counter version based on the string
        representation.
        ______________________________________________________________
        Parameters
        bow_type: str = ""
            String representation of the bow type
        cntrargs: dict = {}
            Arguments for the CountVectorizer
        ______________________________________________________________
        Returns:
        counter: class
            CountVectorizer or TfidfVectorizer
        """
        counter = CountVectorizer
        if bow_type == 'binary':
            cntrargs['binary'] = True

        elif bow_type == 'tfidf':
            counter = TfidfVectorizer

        return counter(**cntrargs)

    @staticmethod
    def _count(x: list, count: CountVectorizer, verbose: bool) -> np.array:
        """
        Transforms the CountVectorizer
        ______________________________________________________________
        Parameters
        x: list
            The data to transform
        count: CountVectorizer
            The CountVectorizer to transform
        verbose: bool
            If True: prints progressive output
        ______________________________________________________________
        Returns:
        count.transform: np.array
            The transformed data
        """
        return count.transform(x)

    def __init__(self, url: str, vocab_size: int = 5000, prop: float = 0.7,
                 device: torch.device = torch.device("cpu"),
                 train: bool = True, verbose: bool = False,
                 random_state: int = 1, bow_type: str = 'counter',
                 cntrargs: dict = {}) -> None:
        """
        Creates an instance of Signal20Dataset from the provided
        arguments.
        ______________________________________________________________
        Parameters
        url: str
            The URL to the data to read
        vocab_size: int = 1
            The size of the input vectors, defaults to 5000
        prop: float = 0.7
            Proportion to split training and validaton data to,
            defaults to 0.8
        device: torch.device = torch.device("cpu")
            Specifies how the model is run, defaults to "cpu"
        train: bool = True
            Specifies whether you are training orvalidating, defaults
            to True
        verbose: bool = False
            If True: prints progressive output, defaults to False
        random_state: int = None
            The seed for the random state, defaults to 1
        bow_type: str = 'counter'
            The type of CountVectorizer to use, defaults to "counter"
        cntrargs: dict = {}
            Arguments for the countvectorizer
        ______________________________________________________________
        Returns:
        None
        """

        # seeding
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # initialize objects
        self.counter = self._get_counter(bow_type, cntrargs)
        cntrargs['max_features'] = vocab_size
        self.vocab_size = vocab_size
        self.verbose = verbose
        self.device = device
        self.prop = prop

        # reading data from file
        self.data = self._read_data(url, self.verbose)

        # handling splits
        self.df = self._dev_split(
            self.data,
            self.prop,
            train,  # bool: specifies train or valid
            self.verbose
        )

        # extract for indexing in __getitem__
        self.text, self.source = self._extract(
            self.df,
            self.verbose
        )

        # generate vocab with counts
        if train:
            # collect feature values
            self.count = self._fit_counter(
                self.counter,
                self.df.text,
                self.verbose
            )

            # generating onehot encoding + source indexer
            self.onehot, self.sources = self._fit_encode(
                self.source,
                self.verbose
            )
            self.source_indexer = {s.strip('_'): i
                                   for i, s in enumerate(self.sources)}
            self.n_classes = len(self.source_indexer)

            # "static" var (once set, holds for all future inits of class)
            Signal20Dataset.count = self.count
            Signal20Dataset.source_indexer = self.source_indexer
            Signal20Dataset.onehot = self.onehot

        # encode sources
        self.source_vec = self._encode(self.source, self.onehot, self.verbose)

    def __getitem__(self, index: int) -> tuple:
        """
        Magic method called when using [ ] indexing on class object.

        Return: tuple (X, y)
            X: all features as tensor
            y: this items gold source as tensor (i.e. the number referring
               to that source)
        """
        # load text and source from data loaded
        s = self.df.source.iloc[index]
        current_source = self.source_indexer[s]

        # get features
        text = self.df.text.iloc[index]
        feature = self._count([text], self.count, self.verbose).toarray()[0]

        # build tensors
        X = torch.FloatTensor(feature)
        y = current_source

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


class BOW:
    """Pre-Processing data without using Torch Dataset and DataLoader."""

    @staticmethod
    def bow_selector(bow_type: str, vocab_size: int):

        if bow_type == "counter":

            if vocab_size == 0:
                return CountVectorizer(dtype=np.float32), \
                    OneHotEncoder(dtype=np.uint8)

            else:
                return CountVectorizer(max_features=vocab_size,
                                       dtype=np.float32), \
                    OneHotEncoder(dtype=np.uint8)

        elif bow_type == "binary":

            if vocab_size == 0:
                return CountVectorizer(binary=True, dtype=np.uint8), \
                    OneHotEncoder(dtype=np.uint8)

            else:
                return CountVectorizer(max_features=vocab_size,
                                       binary=True, dtype=np.uint8), \
                    OneHotEncoder(dtype=np.uint8)

        elif bow_type == "tfidf":

            if vocab_size == 0:
                return TfidfVectorizer(dtype=np.float32), \
                    OneHotEncoder(dtype=np.uint8)

            else:
                return TfidfVectorizer(max_features=vocab_size,
                                       dtype=np.float32), \
                    OneHotEncoder(dtype=np.uint8)

    @staticmethod
    def _filter_vocab(df, pos):

        df_splited = df.text.str.split()
        vocab_final = np.empty(shape=(len(df_splited)), dtype=list)

        for row_idx, article in enumerate(df_splited.values):

            vocab = ""

            for word in article:
                for p in pos:

                    if word[-len(p):] == p:
                        vocab += word + " "

            vocab.strip()
            vocab_final[row_idx] = vocab

        del df_splited

        df2 = pd.DataFrame([df.source, vocab_final]).T
        df2.columns = ["source", "text"]

        return df2

    @staticmethod
    def _train_test_split(df, train_prop):

        n = len(df) * train_prop

        # split train/valid
        df_train = resample(
            df,
            replace=False,
            stratify=df.source,
            n_samples=n
        )

        df_test = df[~df.index.isin(df_train.index)]

        return df_train, df_test

    def __init__(self, bow_type: str = "counter", vocab_size: int = 1000,
                 verbose: bool = False, random_state: int = None) -> None:
        """
        Pre-Processing data without using Torch Dataset and DataLoader
        ______________________________________________________________
        Parameters:
        bow_type: str = "counter"
            The type of BOW to use
        vocab_size: int = 1000
            The size of the input tensors
        verbose: bool = False
            If True: prints progressional output, defaults to False
        random_state: int = None
            The seed for the random state        
        ______________________________________________________________
        Returns:
        None
        """

        # seeding
        if random_state is not None:
            np.random.seed(random_state)

        # initializing objects
        self.verbose = verbose

        self.vectorizer_features, self.vectorizer_classes = \
            self.bow_selector(
                bow_type=bow_type,
                vocab_size=vocab_size
            )

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

    def fit_transform(self, url: str, pos: list = None,
                      train_prop: float = 0.7):
        """
        Transforms and fits the BOW.
        ______________________________________________________________
        Parameters:
        url: str
            The path to the dataset to read
        pos: list=None
            List of pos_tags to use, defaults to None
        train_prop: float=0.7
            The proportion of the train and validation data, defaults
            to 0.7        
        ______________________________________________________________
        Returns:
        None
        """

        print("Preprocessing Starting...") if self.verbose else None

        # reading file
        df = pd.read_csv(url, sep='\t', compression='infer')

        df = self._filter_vocab(df, pos) if pos is not None else df

        # splitting train and test data
        df_train, df_test = self._train_test_split(
            df=df,
            train_prop=train_prop
        )

        self.vectorizer_features.fit(df_train.text)
        X_train = self.vectorizer_features.transform(df_train.text)
        self.X_train = torch.from_numpy(X_train.toarray()).float()

        del X_train

        self.vectorizer_classes.fit(df_train.source.values.reshape(-1, 1))
        y_train = self.vectorizer_classes.transform(
            df_train.source.values.reshape(-1, 1))
        self.y_train = \
            torch.from_numpy(y_train.toarray().argmax(axis=1)).long()

        del y_train

        X_test = self.vectorizer_features.transform(df_test.text)
        self.X_test = torch.from_numpy(X_test.toarray()).float()

        del X_test

        y_test = self.vectorizer_classes.transform(
            df_test.source.values.reshape(-1, 1))
        self.y_test = \
            torch.from_numpy(y_test.toarray().argmax(axis=1)).long()

        del y_test

        print("Preprocessing Finalized.") if self.verbose else None

        return self
