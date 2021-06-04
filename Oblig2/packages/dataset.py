import torch
import pandas as pd
import zipfile
from torch.utils.data import Dataset
import gzip
import lzma
from tqdm import tqdm


class CoNLLDataset(Dataset):
    def __init__(self, partition, embedder, upos_vocab=None):
        entries = []
        with open(f'data/en_ewt-ud-{partition}.conllu', 'r', encoding='utf-8') as f:
            current = []
            for line in f:
                if line.startswith("#"):
                    continue

                if not line.rstrip():
                    entries.append(current)
                    current = []
                    continue

                res = line.strip().split("\t")
                current.append(res)

        self.forms = [[current[1] for current in entry] for entry in entries]
        self.upos = [[current[3] for current in entry] for entry in entries]

        if upos_vocab:
            self.upos_vocab = upos_vocab
        else:
            self.upos_vocab = list(set([item for sublist in self.upos for item in sublist]))
            self.upos_vocab.extend(['@UNK'])

        self.upos_indexer = {i: n for n, i in enumerate(self.upos_vocab)}
        self.embedder = embedder

    def __getitem__(self, index):
        forms = self.forms[index]
        upos = self.upos[index]

        X = torch.LongTensor([self.embedder.vocab[i].index if i in self.embedder.vocab else self.embedder.vocab['<unk>'].index
                              for i in forms])

        y = torch.LongTensor([self.upos_indexer[i] if i in self.upos_vocab else self.upos_indexer['@UNK']
                              for i in upos])
        return X, y

    def __len__(self):
        return len(self.forms)


class CoNLLDataset_per(Dataset):
    def __init__(self, partition='train', embedder=None, upos_vocab=None):
        entries = []
        with open(f'data/en_ewt-ud-{partition}.conllu') as f:
            current = []
            for line in f:
                if line.startswith("#"):
                    continue

                if not line.rstrip():
                    entries.append(current)
                    current = []
                    continue

                res = line.strip().split("\t")
                current.append(res)

        
        # every sentence, every word, list in list
        self.text = [[current[1] for current in entry] for entry in entries]

    def _read_data(self):
        dirname = '/cluster/shared/nlpl/data/corpora/conll17/udpipe/English/'
        filename= 'en-common_crawl-187.conllu.xz'
        
        c = 0
        with lzma.open(dirname+filename, 'rt', ) as f:
            entries = []
            current = []
            for line in tqdm(f):
                if line.startswith("#"):
                    continue

                if not line.rstrip():
                    entries.append(current)
                    current = []
                    continue

                res = line.strip().split("\t")
                current.append(res)
                c+=1
                if c>1000000:
                    break
        



if __name__=='__main__':

    d = CoNLLDataset()
    d._read_data()