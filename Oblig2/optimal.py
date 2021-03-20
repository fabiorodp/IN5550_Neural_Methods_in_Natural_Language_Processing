# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


# imports
try:
    from .packages.preprocessing import pad_batches, process_raw_data
    from .packages.preprocessing import load_embedding, TSVDataset
    from .packages.preprocessing import RNNDataset, collate_fn
    from .packages.ann_models import MLPModel, RNNModel
except:
    from packages.preprocessing import pad_batches, process_raw_data
    from packages.preprocessing import load_embedding, TSVDataset
    from packages.preprocessing import RNNDataset, collate_fn
    from packages.ann_models import MLPModel, RNNModel

from sklearn.metrics import accuracy_score, f1_score
from torch._C import device
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
import time
import pickle


# configure optimal parameters
## data specific
batch_size      = 20 # TODO check if this is desired batch size with fab and eiv         
device_         = 'cpu' 
pos_tagged      = False
random_state    = 21 # TODO change 3 times
store           = True
train_data_url  = 'data/stanford_sentiment_binary.tsv.gz' 
train_prop      = 0.75 # use full train set on optimal model
verbose         = False
vocab_path      = '/cluster/shared/nlpl/data/vectors/latest/22.zip'  # other good ones were 40, 82, ...
# vocab_path      = 'data/22.zip'  # other good ones were 40, 82, ...
## model specific
model_params = {
    'n_hl': 3,                  # rnn param
    'dropout': 0.2,             # not needed, bc singular output. left here for ref
    'epochs': 50,               # anywhere between 30 and 60 seemed reasonable
    'units': 50,                # between 25 and 50, w/ 50 giving optimal score
    'lr': 0.01,                 # dependent on momentum
    'momentum': 0.9,            # dependent on learning rate
    'device': "cpu",            # take as str not torch.device -> easier json print
    'loss_funct': "cross-entropy", # most stable throughout experiments
    'random_state': 1,          # to test different splits
    'verbose': verbose,     
    'rnn_type': 'gru',          # rnn, lstm, gru
    'bidirectional': True,      # checks in first study
    'freeze': True,             # only difference in training time, True -> faster
    'lr_scheduler': True,       # major help for score improvement
    'factor': 0.01,             # not too much affect, but still optimal 
    'patience': 2,              # same as above
    'pool_type': 'first',       # first assuming bidirectional, last otherwise
}

def run(random_state=random_state):
    # load embeding
    embedder = load_embedding(vocab_path)
    embedder.add('<unk>', weights=torch.rand(embedder.vector_size))
    embedder.add('<pad>', weights=torch.zeros(embedder.vector_size))
    pad_token = embedder.vocab['<pad>'].index

    # load data set/loaders
    df_train, df_test = process_raw_data(
            data_url=train_data_url,
            train_prop=train_prop,
            verbose=verbose,
            pos_tagged=pos_tagged,
            random_state=random_state
        )

    train_data = RNNDataset(
        embedder=embedder,
        df=df_train,
        device=device_,
        random_state=random_state,
        label_vocab=None,
        verbose=verbose
    )

    test_data = RNNDataset(
        embedder=embedder,
        df=df_test,
        device=device_,
        random_state=random_state,
        label_vocab=train_data.label_vocab,
        verbose=verbose
    )

    del df_test, df_train

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_token, device=device_)
    )

    test_loader = DataLoader(
        test_data,
        batch_size=len(test_data),
        collate_fn=lambda x: collate_fn(x, pad_token, device=device_)
    )

    # build/fit the model
    model = RNNModel(
        emb=embedder,
        num_features=embedder.vector_size,
        **model_params
    )
    model.fit(
        loader=train_loader, 
        verbose=verbose,
        test=None # no need to evaluate along the way
    )

    # test the model
    y_test, y_pred = model.predict_classes(test_loader)

    print(f'Random state: {random_state}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

    return model, train_data

if __name__=='__main__':
    # model_1337 = run(1337)
    # model_42 = run(42)
    # model_13032021 = run(13032021)

    # # will need to store the model 
    # if store:
    #     with open('model_1337.pkl', 'wb+') as f:
    #         pickle.dump(model_1337, f)
    #     with open('model_42.pkl', 'wb+') as f:
    #         pickle.dump(model_42, f)
    #     with open('model_13032021.pkl', 'wb+') as f:
    #         pickle.dump(model_13032021, f)

    train_prop = 0.999 # use all the data as training data for optimal model
    model, data = run()
    torch.save(model.state_dict(), "model.pt")
    with open('train_data.pkl', 'wb+') as f:
        pickle.dump(data, f)
    

