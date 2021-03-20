# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from sklearn.utils import resample
import pandas as pd
import logging
import zipfile
import gensim
import random
import torch
import nltk
import json
import re


# def make_embedding(filename=None):
#     tsv_data = TSVDataset()
#     text = tsv_data.text
#
#     conll_data = CoNLLDataset(partition='train')
#     text += conll_data.text
#
#     ##### these are found in meta file
#     skipgram = 0  # args.sg
#     # Context window size (e.g., 2 words to the right and to the left)
#     window = 5  # args.window
#     # How many words types we want to be considered (sorted by frequency)?
#     vocabsize = 100000  # args.vocab
#     # cores
#     cores = 1
#     vectorsize = 300  # Dimensionality of the resulting word embeddings.
#     # For how many epochs to train a model (how many passes over corpus)?
#     iterations = 2
#     ####################################
#
#     model = gensim.models.Word2Vec(
#         text,
#         size=vectorsize,
#         window=window,
#         workers=cores,
#         sg=skipgram,
#         max_final_vocab=vocabsize,
#         iter=iterations,
#         sample=0,
#     )
#
#     # intending on training the model further
#     if filename:
#         model.wv.save(filename)
#     return model.wv


def pad_batches(batch, pad_idx):
    longest_sentence = max([X.size(0) for X, y in batch])

    new_X, new_y = [], []

    for X, y in batch:
        new_X.append(torch.nn.functional.pad(
            X,
            (0, longest_sentence - X.size(0)),
            value=pad_idx)
        )

        new_y.append(y)

    new_X = torch.stack(new_X)
    new_y = torch.stack(new_y)

    return new_X, new_y


def train_test_split(df, train_prop, random_state=1):
    n = len(df) * train_prop

    # split train/valid
    df_train = resample(
        df,
        replace=False,
        stratify=df.label,
        n_samples=n,
        random_state=random_state
    )

    df_test = df[~df.index.isin(df_train.index)]

    return df_train, df_test


def process_raw_data(data_url, train_prop=0.75, verbose=True,
                     pos_tagged=False, random_state=1):

    # ########## importing data
    df = pd.read_csv(data_url, sep='\t')
    # ##########
    print(f'data read with columns {df.columns}') \
        if verbose is True else None

    # ########## Initial checks
    # checking classes
    print("Labels/Targets/Classes:\n", df.label.unique(), "\n") \
        if verbose is True else None

    print('dropping unnecessary row') if verbose is True else None
    # dropping wrong row
    df = df.drop(
        index=df[
            (df.label != "negative") & (df.label != "positive")].index
    )

    # checking balance of the classes
    print("Balance of the Classes:\n", df.label.value_counts(), "\n") \
        if verbose is True else None

    # fixing DF's indexes
    df.index = [i for i in range(len(df))]
    # ##########

    print('train test splitting') if verbose is True else None
    # ########## splitting train and unseen test data
    df_train, df_test = train_test_split(
        df=df,
        train_prop=train_prop,
        random_state=random_state
    )
    del df
    # ##########

    print('encoding') if verbose is True else None
    # ########## encoding labels
    # encoder = LabelEncoder()
    # encoder.fit(df_train.label)  # fit encoder to training labels
    # df_train.label = encoder.transform(
    #     df_train.label)  # actually transform
    # df_test.label = encoder.transform(df_test.label)
    # print(df_train.label) if verbose is True else None
    # ##########
    #
    print('tokenizing...') if verbose is True else None
    # ########### Tokenizing - standard
    df_train.tokens = df_train.tokens.apply(
        lambda x: nltk.word_tokenize(x))
    df_test.tokens = df_test.tokens.apply(lambda x: nltk.word_tokenize(x))

    if pos_tagged:
        print('POS tagging...') if verbose is True else None
        df_train.tokens = df_train.tokens.apply(
            lambda x: [z + '_' + y for z, y in
                       nltk.pos_tag(x, tagset='universal')])
        df_test.tokens = df_test.tokens.apply(
            lambda x: [z + '_' + y for z, y in
                       nltk.pos_tag(x, tagset='universal')])

    return df_train, df_test


def load_embedding(modelfile):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
            modelfile.endswith(".txt.gz")
            or modelfile.endswith(".txt")
            or modelfile.endswith(".vec.gz")
            or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


def load_embedded_model(url):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    embeddings_file = url

    logger.info("Loading the embedding model...")
    model = load_embedding(embeddings_file)
    logger.info("Finished loading the embedding model...")

    logger.info(f"Model vocabulary size: {len(model.vocab)}")

    logger.info(f"Random example of a word in the model: "
                f"{random.choice(model.index2word)}")

    return model

# for generating new embeddings
# def make_embedding(filename=None):
#     tsv_data = TSVDataset()
#     text = tsv_data.text
#
#     conll_data = CoNLLDataset(partition='train')
#     text += conll_data.text
#
#     ##### these are found in meta file
#     skipgram = 0 #args.sg
#     # Context window size (e.g., 2 words to the right and to the left)
#     window = 5 #args.window
#     # How many words types we want to be considered (sorted by frequency)?
#     vocabsize = 100000 #args.vocab
#     # cores
#     cores=1
#     vectorsize = 300  # Dimensionality of the resulting word embeddings.
#     # For how many epochs to train a model (how many passes over corpus)?
#     iterations = 2
#     ####################################
#
#     model = gensim.models.Word2Vec(
#         text,
#         size=vectorsize,
#         window=window,
#         workers=cores,
#         sg=skipgram,
#         max_final_vocab=vocabsize,
#         iter=iterations,
#         sample=0,
#     )
#
#     # intending on training the model further
#     if filename:
#         model.wv.save(filename)
#     return model.wv


class TSVDataset(Dataset):

    def __init__(self, embedder, url, pos_tagged=False,
                 random_state=1, device="cpu") -> None:

        self.embedder = embedder
        self.url = url
        self.pos_tagged = pos_tagged
        self.random_state = random_state
        self.device = device

        print(f'processing data file {url}')
        df_train, df_test = process_raw_data(
            data_url=self.url,
            train_prop=0.75,
            verbose=False,
            pos_tagged=self.pos_tagged,
            random_state=self.random_state
        )

        # str.split done @ process_raw:nltk.tokenize
        self.text = list(df_train['tokens'])
        self.label = list(df_train['label'])

        self.text_test = list(df_test['tokens'])
        self.label_test = list(df_test['label'])

        self.label_vocab = sorted(list(set(self.label)))
        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}

        self.encoded_train_labels = \
            [self.label_indexer[i] for i in self.label]

        self.encoded_test_labels = \
            [self.label_indexer[i] for i in self.label_test]

    def __getitem__(self, index: int):
        current_text = self.text[index]
        current_label = self.label[index]

        # long/float will be dependent on embedder
        X = torch.LongTensor([
            self.embedder.vocab[token].index
            if token in self.embedder.vocab else 0
            for token in current_text
        ])

        y = self.label_indexer[current_label]
        y = torch.LongTensor([y])

        # assigning cuda for X and y
        if self.device == "cuda":
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

        return X, y

    def __len__(self):
        """
        Magic method to return the number of samples.
        """
        return len(self.text)

    def get_embedded_test_tensor(self):
        longest_sentence = max([len(s) for s in self.text_test])

        new_X = []
        for sentence in self.text_test:
            s = torch.LongTensor([
                self.embedder.vocab[token].index
                if token in self.embedder.vocab else 0
                for token in sentence])

            new_X.append(
                torch.nn.functional.pad(s, (0, longest_sentence - s.size(0)),
                                        value=self.embedder.vocab[
                                            '<pad>'].index))
        return torch.stack(new_X)


def simple_regex_fix(string):
    # removing: 's, n't, 're, 'll, 've:
    patterns = [r"'s", r"n't", r"'re", r"'ll", r"'ve"]

    for p in patterns:
        string = re.sub(p, "", string)

    return string


def regex_fixes(string):
    # fixing quote marks, 's, n't, 're, 'll, 've, removing comma and dot.
    patterns = [r"``\s*(.*?)\s*''", r"`\s*(.*?)\s*'",
                r"\s*('s)", r"\s*(n't)", r"\s*('re)", r"\s*('ll)",
                r"\s*('ve)",
                r"\s[\.\,\;\:\-\_\<\>\\\/\=\+\*]\s"]

    for p in patterns:
        match = re.search(p, string)

        if match is not None:

            if (p == r"``\s*(.*?)\s*''") or (p == r"`\s*(.*?)\s*'"):
                sub = f"''{match.group(1)}''"
            elif p == r"\s[\.\,\;\:\-\_\<\>\\\/\=\+\*]\s":
                sub = " "
            else:
                sub = f"{match.group(1)}"

            string = re.sub(p, sub, string)

    return string


def collate_fn(batch, pad_X, device='cpu'):
    longest_sentence = max([X.size(0) for X, y in batch])

    new_X = torch.stack([torch.nn.functional.pad(
        X, (0, longest_sentence - X.size(0)), value=pad_X
    ) for X, y in batch])

    new_y = torch.stack([torch.nn.functional.pad(
        y, (0, longest_sentence - y.size(0)), value=-1
    ) for X, y in batch])

    lengths = torch.LongTensor([X.size(0) for X, y in batch])

    # assigning cuda for X and y
    if device == "cuda":
        new_X = new_X.to(torch.device("cuda"))
        lengths = lengths.to(torch.device("cuda"))
        new_y = new_y.to(torch.device("cuda"))

    return (new_X, lengths), new_y


class OurCoNLLDataset(Dataset):

    def __init__(self, embedder, df, device="cpu", upos_vocab=None):

        self.embedder = embedder
        self.device = device

        self.forms = []
        for r in list(df['tokens']):
            sentence = []

            for c in r:
                sentence.append(c.split("_")[0])

            self.forms.append(sentence)

        self.upos = []
        for r in list(df['tokens']):
            sentence = []

            for c in r:
                sentence.append(c.split("_")[1])

            self.upos.append(sentence)

        if upos_vocab:
            self.upos_vocab = upos_vocab

        else:
            self.upos_vocab = list(set(
                    [item for sublist in self.upos for item in sublist]))

            self.upos_vocab.extend(['@UNK'])

        self.upos_indexer = {i: n for n, i in enumerate(self.upos_vocab)}

    def __getitem__(self, index):
        forms = self.forms[index]
        upos = self.upos[index]

        X = torch.LongTensor(
            [self.embedder.vocab[i].index if i in self.embedder.vocab
             else self.embedder.vocab['<unk>'].index
             for i in forms]
        )

        y = torch.LongTensor(
            [self.upos_indexer[i] if i in self.upos_vocab
             else self.upos_indexer['@UNK']
             for i in upos]
        )

        # assigning cuda for X and y
        if self.device == "cuda":
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

        return X, y

    def __len__(self):
        return len(self.forms)


class RNNDataset(Dataset):

    def __init__(self, embedder, df, device="cpu", random_state=1,
                 label_vocab=None, verbose=False) -> None:

        self.embedder = embedder
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.text = list(df['tokens'])
        self.label = list(df['label'])

        if label_vocab:
            self.label_vocab = label_vocab

        else:
            self.label_vocab = sorted(list(set([i for i in self.label])))

        self.label_indexer = {i: n for n, i in enumerate(self.label_vocab)}

        self._unk = embedder.vocab['<unk>'].index

        self.encoded_labels = [self.label_indexer[i] for i in self.label]

    def __getitem__(self, index: int):
        current_text = self.text[index]
        current_label = self.label[index]

        # long/float will be dependent on embedder
        X = torch.LongTensor([
            self.embedder.vocab[token].index
            if token in self.embedder.vocab else self._unk
            for token in current_text
        ])

        y = self.label_indexer[current_label]
        y = torch.LongTensor([y])

        # assigning cuda for X and y
        if self.device == "cuda":
            X = X.to(torch.device("cuda"))
            y = y.to(torch.device("cuda"))

        return X, y

    def __len__(self):
        """
        Magic method to return the number of samples.
        """
        return len(self.text)
