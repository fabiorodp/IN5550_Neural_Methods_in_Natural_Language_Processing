# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


# from nltk.tokenize import WordPunctTokenizer
from torch.utils.data import Dataset
from conllu import parse_incr
import pandas as pd
import torch


def load_raw_data(datapath):
    with open(datapath, "r", encoding="utf-8") as data_file:
        slow_parse = [
            {
                'TokenList': tokenlist,
                'sentence': tokenlist.metadata['text'],
                'labels': [t['misc']['name'] for t in tokenlist]
            }
            for tokenlist in parse_incr(data_file)]

    return pd.DataFrame(slow_parse)


def filter_raw_data(df, min_entities=1, max_entities=10000000):
    """
    Removing sentences with 'min' and 'max' labels other than 'O'.

    Args:
        df: pd.DataFrame containing all sentences of the data.
        min_entities: int representing the min number of labels other than
        'O'.
        max_entities: int representing the max number of labels other than
        'O'.

    Returns:
        filtered_df: pd.DataFrame containing only
    """
    row_idxs_to_drop = []
    for row_idx, row in enumerate(df.iloc[:, 2]):
        labels = []  # appending all entity indices

        for e in row:
            if e != 'O':
                labels.append(e)

        if (len(labels) < min_entities) or (len(labels) >= max_entities):
            row_idxs_to_drop.append(row_idx)

    return df.drop(row_idxs_to_drop, axis=0)


def pad(batch, IGNORE_ID):
    longest_sentence = max([X.size(0) for X, y, z in batch])

    new_X, new_y, new_z = [], [], []

    for X, y, z in batch:
        new_X.append(torch.nn.functional.pad(
            X,
            (0, longest_sentence - X.size(0)),
            value=0)  # find padding index in bert
        )
        new_y.append(torch.nn.functional.pad(
            y,
            (0, longest_sentence - y.size(0)),
            value=IGNORE_ID)
        )
        new_z.append(torch.nn.functional.pad(
            z,
            (0, longest_sentence - z.size(0)),
            value=0)
        )

    new_X = torch.stack(new_X).long()
    new_y = torch.stack(new_y).long()
    new_z = torch.stack(new_z).long()

    return new_X, new_y, new_z


class OurCONLLUDataset(Dataset):

    def __init__(self, df, tokenizer=None, label_vocab=None, device='cpu',
                 label_indexer=None):
        """
        # what is needed from this dataset
        # every word as token
        # the BIO tag for each word
        # the sentence structure, to convert back to CONLLU format
        """
        self._tokenizer = tokenizer
        self.device = device

        self.TokenList = list(df['TokenList'])
        self.text = list(df['sentence'])  # list of sentence strings
        self.label = list(df['labels'])  # list of list of labels

        self.label_vocab = label_vocab if label_vocab else set([
            e
            for label in self.label
            for e in label
        ])

        if label_indexer:
            self.label_indexer = label_indexer
            self.IGNORE_ID = self.label_indexer['[MASK]']

        else:
            self.label_indexer = \
                {i: n for n, i in enumerate(self.label_vocab)}

            # len is one more than highest index
            self.IGNORE_ID = len(self.label_vocab)  # can not be 0 or 1
            self.label_indexer['[MASK]'] = self.IGNORE_ID

    def __getitem__(self, index):
        current_sentence = self.text[index]
        current_label = self.label[index]

        input_ids = self._tokenizer(current_sentence)['input_ids']
        attn_mask = self._tokenizer(current_sentence)['attention_mask']

        y = [int(self.label_indexer[t]) for t in current_label]
        y_masks = self._build_y_masks(input_ids)
        self.y_extended = self._extend_labels(y, y_masks)

        return (torch.LongTensor(input_ids).to(torch.device(self.device)),
                self.y_extended.to(torch.device(self.device)),
                torch.tensor(attn_mask).to(torch.device(self.device)))

    def _build_y_masks(self, ids):
        tok_sent = [self._tokenizer.convert_ids_to_tokens(i) for i in ids]
        mask = torch.zeros(len(tok_sent))

        for i, token in enumerate(tok_sent):
            if token.startswith('##'):
                mask[i] = self.IGNORE_ID
            elif token in self._tokenizer.all_special_tokens:
                mask[i] = self.IGNORE_ID
            else:
                mask[i] = 1
        return mask

    def _extend_labels(self, labels, mask):
        """
        Example:
        ______________________________________________________________

        - Sentence:
        'Nominasjonskampen i Oslo SV mellom Heikki Holmås og
        Akhtar Chaudhry i desember i fjor handlet blant annet om
        beskyldninger om juks.'

        - Token and its ID or mask, respectively:
        '[CLS]'=6, 'No'=2, '##min'=2, '##asjons'=2, '##kampen'=2, 'i'=2,
        'Oslo'=3, 'SV'=4, 'mellom'=2, 'Hei'=5, '##kk'=5, '##i'=5,
        'Holm'=1, '##ås'=1, 'og'=2, 'Ak'=5, '##htar'=5, 'Ch'=1,
        '##aud'=1, '##hr'=1, '##y'=1, 'i'=2, 'desember'=2, 'i'=2,
        'fjor'=2, 'handlet'=2, 'blant'=2, 'annet'=2, 'om'=2,
        'beskyld'=2, '##ninger'=2, 'om'=2, 'juks'=2, '##.'=2,
        '[SEP]'=6

        - returns: torch.tensor containing the token's ID or mask
        tensor([6, 2, 2, 2, 2, 2, 3, 4, 2, 5, 5, 5, 1, 1, 2, 5, 5, 1,
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6])
        """
        extended = torch.zeros(len(mask))
        t = 0

        for i, m in enumerate(mask):
            if m == 0:
                # add previous
                extended[i] = extended[i - 1].item()
                # keep t indexer the same
            elif m == 1:
                # add current
                extended[i] = int(labels[t])
                t += 1
            else:
                # add special tokens
                extended[i] = int(self.label_indexer['[MASK]'])
                # keep t indexer the same

        return extended

    def __len__(self):
        return len(self.text)
