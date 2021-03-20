import torch
import pandas as pd
from torch.utils.data import Dataset


class CoNLLDataset(Dataset):
    def __init__(self, partition, embedder=None, upos_vocab=None):
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

    