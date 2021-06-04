from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import BertTokenizer, BertForTokenClassification
from metrics import binary_analysis, proportional_analysis, get_analysis


class Vocab(defaultdict):
    """
    This function creates the vocabulary dynamically. 
    As you call ws2ids, it updates the vocabulary with any new tokens.
    """

    def __init__(self, train=True):
        super().__init__(lambda : len(self))
        self.train = train
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        # set UNK token to 1 index
        self[self.PAD] # why not specify =1?
        self[self.UNK]
    #
    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws] # returns token's id (creates new if not present)
        else:
            return [self[w] if w in self else 1 for w in ws] # NOTE ignore index=1
    #
    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if int(i) in idx2w else self.UNK for i in ids]


class Split(object):
    def __init__(self, data):
        '''
        Iterable object that takes care of batch padding.
        ______________________________________________________________
        Parameters:
            data: list(tuples) containing each datum
        '''
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    # NOTE don't know if above is used somewhere else, so made new in case    
    def pack(self, ws):
        return pack_sequence(ws, enforce_sorted=False)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item : len(item[0]), reverse=True)

        # NOTE are 4 for-loops really fast than 1 w/ 2 appends?
        raws = [raw for raw, word, target, idx in batch]
        words = self.pack([word for raw, word, target, idx in batch])
        targets = self.pack([target for raw, word, target, idx in batch])
        idxs = [idx for raw, word, target, idx in batch]

        return raws, words, targets, idxs

    def collate_fn_transformer(self, batch):
        '''
        parameters
            batch: 
                raw: ...
        '''
        longest_sentence = max([X.size(0) for raw, X, y, z, idx in batch])

        raws, words, targets, masks, idxs = [], [], [], [], []

        for raw, word, target, mask, idx in batch:
            raws += [raw]
            words += [torch.nn.functional.pad(
                word,
                (0, longest_sentence - word.size(0)),
                value=1 # both pad index and unk index are 1
            ) ]
            targets += [torch.nn.functional.pad(
                target,
                (0, longest_sentence - target.size(0)),
                value=1 # both pad index and unk index are 1
            ) ]
            masks += [torch.nn.functional.pad(
                mask,
                (0, longest_sentence - mask.size(0)),
                value=1 # both pad index and unk index are 1
            ) ]
            idxs += [idx]
        
        words = torch.stack(words).long()
        targets = torch.stack(targets).long()
        masks = torch.stack(masks).long()


        return raws, words, targets, masks, idxs


class ConllDataset(object):
    def __init__(self, vocab=None, tokenizer=None):
        '''
        Creates iterable dataset object with list(tokens), tensor(token ids),
        and tensor(label ids).
        ______________________________________________________________
        Parameters:
            vocab: embedder object, can be Vocab (local) or BertTokenizer (imported) 
        ______________________________________________________________
        Returns:
            Split() object, where each index has 3-tuple with values:
                0: text - list(tokens) representing sentence
                1: token ids - torch.LongTensor of embedding ids
                2: label ids - torch.LongTensor of label ids

        '''

        self.vocab = vocab
        self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                          "B-targ-Negative": 3, "I-targ-Negative": 4}

        # Specifies whether to setup for tokenizer use
        self.tokenizer = tokenizer

    def load_conll(self, data_file):
        '''
        Parses conllu file.
        Returns:
            sents - list(list(tokens))      # [['sentence', 'one'],...]
            all_labels - list(list(tags))   # [['B-targ-Positive', 'O'], ...]
        '''
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
        return sents, all_labels

    def get_split(self, data_file):
        '''
        Reads sentences from conll file and maps tokens to ids.
        ______________________________________________________________
        Parameters:
            data_file: str specifying path to conll file
        ______________________________________________________________
        Returns:
            Split() object, where each index has 3-tuple with values:
                0: text - list(tokens) representing sentence
                1: token ids - torch.LongTensor of embedding ids
                2: label ids - torch.LongTensor of label ids

        '''
        if self.tokenizer:
            return self.get_tokenizer_split(data_file)
        
        sents, labels = self.load_conll(data_file)
        data_split = [(text,
                       torch.LongTensor(self.vocab.ws2ids(text)),
                       torch.LongTensor([self.label2idx[l] for l in label]),
                       idx) for idx, (text, label) in enumerate(zip(sents, labels))]
        return Split(data_split)

    def get_tokenizer_split(self, data_file):
        '''
        Reads sentences from conll file and maps tokens to ids.
        ______________________________________________________________
        Parameters:
            data_file: str specifying path to conll file
        ______________________________________________________________
        Returns:
            Split() object, where each index has 3-tuple with values:
                0: text - list(tokens) representing sentence
                1: token ids - torch.LongTensor of embedding ids
                2: label ids - torch.LongTensor of label ids

        '''
        
        sents, labels = self.load_conll(data_file)
        data_split = [
            (
                text,           # raw
                self.tokenizer(text,
                    is_split_into_words=True,
                    return_tensors='pt'
                )['input_ids'].squeeze(0), # token ids
                torch.LongTensor(
                    [self.label2idx[l] for l in label]
                ),              # label ids
                self.tokenizer(text,
                    is_split_into_words=True,
                    return_tensors='pt'
                )['attention_mask'].squeeze(0), # attention mask
                idx             # sentence index in batch
            ) 
            for idx, (text, label) in enumerate(zip(sents, labels))
        ]
        return Split(data_split)

