# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
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
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

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

        elif bow_type == "tfidf":
            self.vectorizer_classes = OneHotEncoder()

            if vocab_size == 0:
                self.vectorizer_features = TfidfVectorizer(binary=True)

            else:
                self.vectorizer_features = \
                    TfidfVectorizer(max_features=vocab_size, binary=True)

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
