from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
try:
    from .packages.preprocessing import make_embedding, load_embedding, process_raw_data 
    from .packages.preprocessing import RNNDataset, TSVDataset, pad_batches
    from .packages.ann_models import RNNtoMLPModel
except:
    from packages.preprocessing import make_embedding, load_embedding, process_raw_data  
    from packages.preprocessing import RNNDataset, TSVDataset, pad_batches
    from packages.ann_models import RNNtoMLPModel  # type: ignore
import pandas as pd
import numpy as np
import torch
import nltk
import os
import logging
import json
import random
import re
import pickle
import lzma


store = True
RANDOM_STATE = 1



# load pickled items, or genertate new
try:
    # raise NotImplementedError('break pickle load on purpose')
    pickle_items = {}
    pickle_item_names = ['embedding', 'train_data', 'test_data']
    for pname in pickle_item_names:
        with open(pname+'.pkl', 'rb') as f:
            pickle_items[pname]= pickle.load(f)

    print('All pickled file read. Loading to objects')
    embedding = pickle_items['embedding']
    train_data = pickle_items['train_data']
    test_data = pickle_items['test_data']
    pad_idx = embedding.vocab['<pad>'].index
    print('Pickled items loaded')

# get data from rnndataset not tsv
except:
    print('Local pickle loads failed.\n Generating new...')
    # print('creating embedding...')
    # embedding = make_embedding(N=2) # only load first 2 coprus files
    # embedding = load_embedding('/cluster/shared/nlpl/data/vectors/latest/40.zip')
    print('loading embedding...')
    embedding = load_embedding('data/40.zip')
    embedding.add(
        '<pad>',
        weights=torch.zeros(embedding.vector_size)
    )
    pad_idx = embedding.vocab['<pad>'].index

    embedding.add(
        '<unk>',
        weights=torch.rand(embedding.vector_size)
    )
    unk_idx = embedding.vocab['<unk>'].index

    

    print('\n\n\nloading dataset...')
    df_train, df_test = process_raw_data(
        data_url='data/stanford_sentiment_binary.tsv.gz',
        train_prop=0.75,
        verbose=True,
        pos_tagged=False,
        random_state=1
    )

    # TODO: make sure the slicing is removed for full data set analysis
    train_data = RNNDataset(            # NOTE Check preprocessing for new dataset type
        embedder=embedding,
        df=df_train.iloc[:500],     # NOTE: Make smaller data set for quicker testing
        device="cpu",
        random_state=1,
        label_vocab=None,
        verbose=True
    )

    test_data = RNNDataset(
        embedder=embedding,
        df=df_test.iloc[:200],      # NOTE: Make smaller data set for quicker testing
        device="cpu",
        random_state=1,
        label_vocab=train_data.label_vocab, # NOTE: genius!
        verbose=True
    )

    del df_test, df_train

    # pickle all these items: embedding and data
    if store:
        pickle_items = [embedding, train_data, test_data]
        pickle_item_names = ['embedding', 'train_data', 'test_data']
        for pitem, pname in zip(pickle_items, pickle_item_names):
            with open(pname+'.pkl', 'wb+') as f:
                pickle.dump(pitem, f)


print('\n\n\n')
print('loading data loader...')
train_loader = DataLoader(
    train_data, 
    batch_size=32,
    collate_fn = lambda x: pad_batches(x, pad_idx, True) # True: add lengths functionality
)

test_loader = DataLoader(
    test_data, 
    batch_size=len(test_data), # test entire test set as one batch
    collate_fn = lambda x: pad_batches(x, pad_idx, True) # True: add lengths functionality
)


model_params = {
    'n_hl': 1,
    'n_classes':2,
    'dropout': 0.2,
    'epochs': 10,
    'units': 25,
    'lr': 1.,  # TODO change back to default 0.01
    'momentum': 0.1, # TODO change back to default 0.9
    'device': 'cpu',
    'weights_init': "xavier_normal",
    'hl_actfunct': "tanh",      
    'out_actfunct': "relu",         # TODO make sure these are updated
    'loss_funct': "cross-entropy",  
    'random_state':RANDOM_STATE,
    'verbose': True,
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    # specific for rnn  #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
    'rnn_type':'gru',       # gru, lstm, or elman. decided internally using static method
    'rnn_n_hl':3,           # how many hidden rnn layers should model have
    'blank_emb':False,      # whether to load embeddings from pretrained or not.
    'nonlinearity': 'tanh', # tanh, relu
    'bidirectional':False,  # messes with shape of output. needs fixing before using TODO
    'freeze':True,         # update embeddings every epoch or not
    'pool_type': None,
    'optim': 'adam',         # sgd, adam, adagrad, adadelta. decided internally using s.m. 
    'stack_size': 3,        # values>1 create multiple RNN objects input sequentially passes through
}
print('\n\n\n')
print('loading model with params:')
print(json.dumps(model_params, indent=2))
model = RNNtoMLPModel(
    emb=embedding,
    num_features=embedding.vector_size,
    **model_params,
)


print('\n\n\n')
print('fitting model...')
model.fit(loader=train_loader)


print('\n\n\n')
print('predicting test data...')
# add test data through data loader 
# why would you bother writing this over again?
# TODO fix the serving of X and y to repdict and accuracy
# test_batch = iter(next(test_loader))
# X, y, lengths = test_batch
# print(X.shape)
# print(y.shape)
# print(len(lengths))
# raise NotImplementedError('break before predicting in single')
y_pred = model.predict_classes(test_loader) 

# TODO make sure predict classes is up to date with this call
print('\n\n\n')
print('scoring...')
y_gold = [torch.LongTensor([y]) for y in data.label_test]
y_gold = torch.stack(y_gold)

print('y_gold: ', y_gold.size())
print('y_pred: ', y_pred.size())
##########################################################

accur = accuracy_score(
    y_gold, 
    y_pred
)
print('Accuracy: ', accur)
f1score = f1_score(
    y_gold, 
    y_pred,
    average="macro",
    zero_division=0
)
print('F1: ', f1score)
