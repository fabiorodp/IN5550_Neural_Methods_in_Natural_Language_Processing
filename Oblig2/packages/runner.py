import sys
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from dataset import CoNLLDataset
from model import RNNModel

from play_with_gensim import load_embedding

def collate_fn(batch, pad_X):
    longest_sentence = max([X.size(0) for X, y in batch])
    new_X = torch.stack([F.pad(X, (0, longest_sentence - X.size(0)), value=pad_X) for X, y in batch])
    new_y = torch.stack([F.pad(y, (0, longest_sentence - y.size(0)), value=-1) for X, y in batch])
    lengths = torch.LongTensor([X.size(0) for X, y in batch])
    return (new_X, lengths), new_y


def evaluate(y_pred, y_gold):
    y_pred = y_pred.argmax(dim=-1)
    correct = (y_pred == y_gold).nonzero(as_tuple=False).size(0)
    total = (y_gold != -1).nonzero(as_tuple=False).size(0)
    return correct / total


def main():
    # load your own model
    word2vec = load_embedding(sys.argv[1])
    word2vec.add('<unk>', weights=np.random.rand(word2vec.vector_size))
    word2vec.add('<pad>', weights=np.random.rand(word2vec.vector_size))

    # build datasets
    train_data = CoNLLDataset(embedder=word2vec, partition='train')
    dev_data = CoNLLDataset(embedder=word2vec, partition='dev', upos_vocab=train_data.upos_vocab)
    # train_data = TSVDataset(embedder=word2vec, partition='train')
    # dev_data = TSVDataset(embedder=word2vec, partition='dev', upos_vocab=train_data.upos_vocab)

    pad_token = train_data.embedder.vocab['<pad>'].index

    # build and pad with loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token))
    dev_loader = DataLoader(dev_data, batch_size=len(dev_data), shuffle=False, collate_fn=lambda x: collate_fn(x, pad_token))


    
    # ignore_index -> padding for the labels
    model = RNNModel(word2vec, train_data.upos_vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    # train/eval loop
    for epoch in range(5):
        train_iter = tqdm.tqdm(train_loader)
        model.train()
        for X, y in train_iter:
            optimiser.zero_grad()
            print(X[0].shape)
            print(X[1].shape)
            print(y.shape)
            y_pred = model(X[0]).permute(0, 2, 1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            train_iter.set_postfix_str(f"loss: {loss.item()}")

        dev_iter = tqdm.tqdm(dev_loader)
        model.eval()
        for X, y in dev_iter:
            y_pred = model(X)
            print(f"dev. accuracy: {evaluate(y_pred, y)}")


if __name__ == '__main__':
    main()
