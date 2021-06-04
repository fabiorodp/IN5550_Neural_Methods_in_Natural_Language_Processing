# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


def pad(batch, IGNORE_ID):
    longest_sentence = max([X.size(0) for X, y, z, i in batch])
    new_X, new_y, new_z, new_i = [], [], [], []

    for X, y, z, i in batch:
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
        new_i.append(i)

    new_X = torch.stack(new_X).long()
    new_y = torch.stack(new_y).long()
    new_z = torch.stack(new_z).long()

    return new_X, new_y, new_z, new_i


def pad_b(batch, IGNORE_ID):
    longest_sentence = max([X.size(0) for X, y, z, k, h, i in batch])
    new_X, new_y, new_z, new_k, new_h, new_i = [], [], [], [], [], []

    for X, y, z, k, h, i in batch:
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
        new_k.append(torch.nn.functional.pad(
            k,
            (0, longest_sentence - k.size(0)),
            value=IGNORE_ID)
        )
        new_h.append(torch.nn.functional.pad(
            h,
            (0, longest_sentence - h.size(0)),
            value=IGNORE_ID)
        )
        new_i.append(i)

    new_X = torch.stack(new_X).long()
    new_y = torch.stack(new_y).long()
    new_z = torch.stack(new_z).long()
    new_k = torch.stack(new_k).long()
    new_h = torch.stack(new_h).long()

    return new_X, new_y, new_z, new_k, new_h, new_i


class OurDataset(Dataset):

    @staticmethod
    def load_raw_data(data_file):
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)

        # sents = [' '.join(s) for s in sents]
        return sents, all_labels

    @staticmethod
    def getting_y(all_labels):
        BIO_labels, polarity_labels = [], []
        for row in all_labels:
            bio, polarity = [], []
            for label in row:
                if label == 'O':
                    polarity.append(label)
                else:
                    polarity.append(label.split('-')[2])
                bio.append(label.split('-')[0])
            BIO_labels.append(bio)
            polarity_labels.append(polarity)

        return BIO_labels, polarity_labels

    def __init__(self, data_file, specify_y=None, NORBERT_path=None,
                 tokenizer=None):
        """
        NORBERT_path = '/cluster/shared/nlpl/data/vectors/latest/216'
        data_file = 'exam/data/train.conll'
        specify_y = None, 'BIO' or 'polarity' or 'both'
        """
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(NORBERT_path)

        self.sents, self.all_labels = self.load_raw_data(data_file)
        self.specify_y = specify_y

        if self.specify_y is not None:
            self.BIO_labels, self.polarity_labels = self.getting_y(
                all_labels=self.all_labels
            )

        self.indexer = {
            "O": 0,
            "B-targ-Positive": 1,
            "I-targ-Positive": 2,
            "B-targ-Negative": 3,
            "I-targ-Negative": 4
        }

        self.IGNORE_ID = len(self.indexer)
        self.indexer['[MASK]'] = self.IGNORE_ID

        self.BIO_indexer = {
            "O": 0,
            "I": 1,
            "B": 2,
            '[MASK]': self.IGNORE_ID
        }

        self.polarity_indexer = {
            "O": 0,  # self.IGNORE_ID,
            "Positive": 1,
            "Negative": 2,
            '[MASK]': self.IGNORE_ID
        }

    def __getitem__(self, index):
        current_sentence = self.sents[index]
        current_label = self.all_labels[index]

        # #################### filters
        new_current_sentence = []
        for idx, word in enumerate(current_sentence):
            if len(word) > 1:
                new_word = word.replace("-", "")
                new_word = new_word.replace('—', "")
                new_word = new_word.replace("&", "")
                new_word = new_word.replace("*", "")
                new_word = new_word.replace("@", "")
                new_word = new_word.replace("+", "")
                new_word = new_word.replace("(", "")
                new_word = new_word.replace(")", "")
                new_word = new_word.replace("_", "")
                new_word = new_word.replace("'", "")
                new_word = new_word.replace(".", "")
                new_word = new_word.replace(":", "")
                new_word = new_word.replace("...", "")
                new_word = new_word.replace("/", "")
                new_word = new_word.replace(",", "")
                new_word = new_word.replace("|", "")
                new_current_sentence.append(new_word)
            else:
                new_current_sentence.append(word)

        current_sentence = new_current_sentence
        del new_current_sentence

        # ####################

        input_ids = self.tokenizer(
            current_sentence,
            is_split_into_words=True
        )['input_ids']

        attention_mask = self.tokenizer(
            current_sentence,
            is_split_into_words=True
        )['attention_mask']

        # y for full
        y = [self.indexer[bt] for bt in current_label]
        y_masks = self._build_y_masks(input_ids)
        y_extended = self._extend_labels(y, y_masks)
        del y, y_masks

        if self.specify_y is not None:
            # y for bio
            y_BIO = [self.BIO_indexer[bt] for bt in self.BIO_labels[index]]
            y_masks = self._build_y_masks(input_ids)
            y_extended_bio = self._extend_labels(y_BIO, y_masks)
            del y_BIO, y_masks

            # y for polarity
            y_polarity = [self.polarity_indexer[bt] for bt in
                          self.polarity_labels[index]]
            y_masks = self._build_y_masks(input_ids)
            y_extended_polarity = self._extend_labels(y_polarity, y_masks)

            if self.specify_y == 'BIO':
                return (
                    torch.LongTensor(input_ids),
                    torch.LongTensor(y_extended_bio),
                    torch.LongTensor(attention_mask),
                    torch.LongTensor(y_extended),
                    torch.LongTensor(y_extended_polarity),
                    current_sentence
                )
            else:
                return (
                    torch.LongTensor(input_ids),
                    torch.LongTensor(y_extended_polarity),
                    torch.LongTensor(attention_mask),
                    torch.LongTensor(y_extended),
                    torch.LongTensor(y_extended_bio),
                    current_sentence
                )

        else:
            return (
                torch.LongTensor(input_ids),
                torch.LongTensor(y_extended),
                torch.LongTensor(attention_mask),
                current_sentence
            )

    def __len__(self):
        return len(self.sents)

    def _build_y_masks(self, ids):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~
        train_dataset.current_label >> size >> 17
        ['B-targ-Negative', 'I-targ-Negative', 'I-targ-Negative', 'O',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        train_dataset.current_sentence >> size >> 17
        ['Hiphop-acts', 'med', 'liveband', 'feiler', 'desverre',
         'altfor', 'ofte', '-', 'og', 'dette', 'er', 'et', 'godt',
         'eksempel', 'akkurat', 'dét', '.']

        token and y_mask
        ['[CLS]' = 0, 'Hiphop-acts' = 1, 'med' = 1, 'live' = 1, '##band' = 0,
        'feil' = 1, '##er' = 0, 'des' = 1, '##ver' = 0, '##re' = 0,
        'altfor' = 1, 'ofte' = 1, '-' = 1, 'og' = 1, 'dette' = 1,
        'er' = 1, 'et' = 1, 'godt' = 1, 'eksempel' = 1, 'akkurat' = 1,
        'd' = 1, '##ét' = 0, '.' = 1, '[SEP]=0']
        """
        tok_sent = [self.tokenizer.convert_ids_to_tokens(i) for i in ids]
        mask = torch.empty(len(tok_sent), dtype=torch.long)

        for i, token in enumerate(tok_sent):
            if token.startswith('##'):
                mask[i] = 0
            elif token in self.tokenizer.all_special_tokens + ['[MASK]']:
                mask[i] = 0
            else:
                mask[i] = 1

        return mask

    def _extend_labels(self, labels, mask):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~

        - Sentence:
        'Nominasjonskampen i Oslo SV mellom Heikki Holmås og
        Akhtar Chaudhry i desember i fjor handlet blant annet om
        beskyldninger om juks.'

        - Token and its ID or mask, respectively:
        '[CLS]'=6, 'No'=2, '##min'=0, '##asjons'=0, '##kampen'=0, 'i'=2,
        'Oslo'=3, 'SV'=4, 'mellom'=2, 'Hei'=5, '##kk'=0, '##i'=0,
        'Holm'=1, '##ås'=0, 'og'=2, 'Ak'=5, '##htar'=0, 'Ch'=1,
        '##aud'=0, '##hr'=0, '##y'=0, 'i'=2, 'desember'=2, 'i'=2,
        'fjor'=2, 'handlet'=2, 'blant'=2, 'annet'=2, 'om'=2,
        'beskyld'=2, '##ninger'=0, 'om'=2, 'juks'=2, '##.'=6,
        '[SEP]'=6, '[PAD]'=6 ...

        - returns: torch.tensor containing the token's ID or mask
        tensor([0, 2, 0, 0, 0, 2, 3, 4, 2, 5, 0, 0, 1, 0, 2, 5, 0, 1,
        0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0])
        """
        extended = torch.empty(mask.size(0), dtype=torch.long)

        label_idx = 0
        for i, m in enumerate(mask.tolist()):

            if m == 1:
                extended[i] = labels[label_idx]
                label_idx += 1

            else:
                extended[i] = self.IGNORE_ID

        return extended



def pad_both(batch, IGNORE_ID, both=False):
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
            (0, longest_sentence - y.size(0)) if not both else(0, longest_sentence*2 - y.size(0)),
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

class MTLDataset(Dataset):

    @staticmethod
    def load_raw_data(data_file):
        sents, all_labels = [], []
        sent, labels = [], []
        for line in open(data_file, encoding="ISO-8859-1"):
            if line.strip() == "":
                sents.append(sent)
                all_labels.append(labels)
                sent, labels = [], []
            else:
                token, label = line.strip().split("\t")
                sent.append(token)
                labels.append(label)

        # sents = [' '.join(s) for s in sents]
        return sents, all_labels

    @staticmethod
    def getting_y(specify_y, all_labels):

        if specify_y == 'BIO':
            BIO_labels = []
            for row in all_labels:
                bio = []
                for label in row:
                    bio.append(label.split('-')[0])
                BIO_labels.append(bio)

            return BIO_labels, []

        elif specify_y == 'polarity':
            polarity_labels = []
            for row in all_labels:
                polarity = []
                for label in row:
                    if label == 'O':
                        polarity.append(label)
                    else:
                        polarity.append(label.split('-')[2])
                polarity_labels.append(polarity)

            return [], polarity_labels

        elif specify_y == 'both':
            BIO_labels, polarity_labels = [], []
            for row in all_labels:
                bio, polarity = [], []
                for label in row:
                    if label == 'O':
                        polarity.append(label)
                    else:
                        polarity.append(label.split('-')[2])
                    bio.append(label.split('-')[0])
                BIO_labels.append(bio)
                polarity_labels.append(polarity)

            return BIO_labels, polarity_labels
        
        return all_labels

    def __init__(self, data_file, specify_y=None, NORBERT_path=None,
                 tokenizer=None):
        """
        NORBERT_path = '/cluster/shared/nlpl/data/vectors/latest/216'
        data_file = 'exam/data/train.conll'
        specify_y = None, 'BIO' or 'polarity' or 'both'
        """
        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(NORBERT_path)

        self.sents, self.all_labels = self.load_raw_data(data_file)
        self.specify_y = specify_y

        if self.specify_y:
            self.BIO_labels, self.polarity_labels = self.getting_y(
                specify_y=self.specify_y,
                all_labels=self.all_labels
            )

        self.indexer = {
            "O": 0,
            "B-targ-Positive": 1,
            "I-targ-Positive": 2,
            "B-targ-Negative": 3,
            "I-targ-Negative": 4
        }

        self.BIO_indexer = {
            "O": 0,
            "I": 1,
            "B": 2,
        }

        self.polarity_indexer = {
            "O": 0,
            "Positive": 1,
            "Negative": 2,
        }

        if self.specify_y == 'BIO':
            self.IGNORE_ID = len(self.BIO_indexer)
            self.BIO_indexer['[MASK]'] = self.IGNORE_ID
        elif self.specify_y == 'polarity':
            self.IGNORE_ID = len(self.polarity_indexer)
            self.polarity_indexer['[MASK]'] = self.IGNORE_ID
        elif self.specify_y == 'both':
            self.IGNORE_ID = len(self.BIO_indexer)
            self.BIO_indexer['[MASK]'] = self.IGNORE_ID
            self.polarity_indexer['[MASK]'] = self.IGNORE_ID
        else: #if self.specify_y is None:
            self.IGNORE_ID = len(self.indexer)
            self.indexer['[MASK]'] = self.IGNORE_ID

    def __getitem__(self, index):
        self.index = index

        self.current_sentence1 = self.sents[index]
        self.current_label = self.all_labels[index]

        # #################### filters
        new_current_sentence = []
        for idx, word in enumerate(self.current_sentence1):
            if len(word) > 1:
                new_word = word.replace("-", "")
                new_word = new_word.replace('—', "")
                new_word = new_word.replace("&", "")
                new_word = new_word.replace("*", "")
                new_word = new_word.replace("@", "")
                new_word = new_word.replace("+", "")
                new_word = new_word.replace("(", "")
                new_word = new_word.replace(")", "")
                new_word = new_word.replace("_", "")
                new_word = new_word.replace("'", "")
                new_word = new_word.replace(".", "")
                new_word = new_word.replace(":", "")
                new_word = new_word.replace("...", "")
                new_word = new_word.replace("/", "")
                new_word = new_word.replace(",", "")
                new_word = new_word.replace("|", "")
                new_current_sentence.append(new_word)
            else:
                new_current_sentence.append(word)

        self.current_sentence = new_current_sentence
        # ####################

        self.input_ids = self.tokenizer(
            self.sents[index],
            is_split_into_words=True,
            #return_tensors='pt'
        )['input_ids']#.squeeze(0) #NOTE: needed squeeze here

        self.attention_mask = self.tokenizer(
            self.sents[index],
            is_split_into_words=True,
            #return_tensors='pt'
        )['attention_mask']#.squeeze(0) #NOTE: needed squeeze here

        if self.specify_y == 'BIO':

            y_BIO = [
                self.BIO_indexer[bt]
                for bt in self.BIO_labels[index]
            ]

            self.y_masks = self._build_y_masks(self.input_ids)
            self.y_extended = self._extend_labels(y_BIO,
                                                  self.y_masks)

        elif self.specify_y == 'polarity':

            y_polarity = [
                self.polarity_indexer[bt]
                for bt in self.polarity_labels[index]
            ]

            return (
                torch.LongTensor(self.input_ids),
                torch.LongTensor(y_polarity),
                torch.LongTensor(self.attention_mask)
            )
            
        elif self.specify_y == 'both':
            # print(self.current_label)
            self.y_BIO = [
                self.BIO_indexer[bt.split('-')[0]]
                for bt in self.current_label
            ]

            self.y_polarity = [
                self.polarity_indexer[pt.split('-')[-1]]
                for pt in self.current_label
            ]
            # print('-'*15)
            # print(self.y_BIO)
            # print(self.y_polarity)
            # print('-'*15)

            # ymask will have size input_ids.shape
            self.y_masks = self._build_y_masks(self.input_ids)
            # y_ext will have size self.current_sentence.split(' ').shape
            self.y_BIO_extended = self._extend_labels(self.y_BIO, self.y_masks)
            self.y_polarity_extended = self._extend_labels(self.y_polarity, self.y_masks)
            # print(
            #     self.y_BIO_extended,
            #     self.y_polarity_extended
            # )
            
            self.y_extended = torch.cat((self.y_BIO_extended, self.y_polarity_extended))
            # self.y_extended = self.y_BIO_extended + self.y_polarity_extended
            # print(len(self.y_BIO_extended))
            # print(len(self.y_polarity_extended))
            # print(len(self.y_extended))

        elif self.specify_y is None:
            self.y = [
                self.indexer[bt]
                for bt in self.current_label
            ]

            self.y_masks = self._build_y_masks(self.input_ids)
            self.y_extended = self._extend_labels(self.y, self.y_masks)

        return (
            torch.LongTensor(self.input_ids),
            torch.LongTensor(self.y_extended),
            torch.LongTensor(self.attention_mask)
        )

    def __len__(self):
        return len(self.sents)

    def _build_y_masks(self, ids):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~
        train_dataset.current_label >> size >> 17
        ['B-targ-Negative', 'I-targ-Negative', 'I-targ-Negative', 'O',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        train_dataset.current_sentence >> size >> 17
        ['Hiphop-acts', 'med', 'liveband', 'feiler', 'desverre',
         'altfor', 'ofte', '-', 'og', 'dette', 'er', 'et', 'godt',
         'eksempel', 'akkurat', 'dét', '.']
        token and y_mask
        ['[CLS]' = 0, 'Hiphop-acts' = 1, 'med' = 1, 'live' = 1, '##band' = 0,
        'feil' = 1, '##er' = 0, 'des' = 1, '##ver' = 0, '##re' = 0,
        'altfor' = 1, 'ofte' = 1, '-' = 1, 'og' = 1, 'dette' = 1,
        'er' = 1, 'et' = 1, 'godt' = 1, 'eksempel' = 1, 'akkurat' = 1,
        'd' = 1, '##ét' = 0, '.' = 1, '[SEP]=0']
        """
        self.tok_sent = [self.tokenizer.convert_ids_to_tokens(i) for i in ids]
        mask = torch.empty(len(self.tok_sent), dtype=torch.long)

        for i, token in enumerate(self.tok_sent):
            if token.startswith('##'):
                mask[i] = 0
            elif token in self.tokenizer.all_special_tokens + ['[MASK]']:
                mask[i] = 0
            else:
                mask[i] = 1

        return mask

    def _extend_labels(self, labels, mask):
        """
        Example 1:
        ~~~~~~~~~~~~~~~~~~~~
        - Sentence:
        'Nominasjonskampen i Oslo SV mellom Heikki Holmås og
        Akhtar Chaudhry i desember i fjor handlet blant annet om
        beskyldninger om juks.'
        - Token and its ID or mask, respectively:
        '[CLS]'=6, 'No'=2, '##min'=0, '##asjons'=0, '##kampen'=0, 'i'=2,
        'Oslo'=3, 'SV'=4, 'mellom'=2, 'Hei'=5, '##kk'=0, '##i'=0,
        'Holm'=1, '##ås'=0, 'og'=2, 'Ak'=5, '##htar'=0, 'Ch'=1,
        '##aud'=0, '##hr'=0, '##y'=0, 'i'=2, 'desember'=2, 'i'=2,
        'fjor'=2, 'handlet'=2, 'blant'=2, 'annet'=2, 'om'=2,
        'beskyld'=2, '##ninger'=0, 'om'=2, 'juks'=2, '##.'=6,
        '[SEP]'=6, '[PAD]'=6 ...
        - returns: torch.tensor containing the token's ID or mask
        tensor([0, 2, 0, 0, 0, 2, 3, 4, 2, 5, 0, 0, 1, 0, 2, 5, 0, 1,
        0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0])
        """
        extended = torch.empty(mask.size(0), dtype=torch.long)

        label_idx = 0
        for i, m in enumerate(mask.tolist()):

            if m == 1:
                extended[i] = labels[label_idx]
                label_idx += 1

            else:
                extended[i] = self.IGNORE_ID

        return extended

