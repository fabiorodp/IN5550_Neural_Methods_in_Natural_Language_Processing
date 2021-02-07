# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from packages.ann_models import MLPModel_wl
from packages.preprocessing import BOW
import pandas as pd
import torch
import time
import sys


MDL_PATH = 'final_model.pt' # where the optimal torch model is saved
VERBOSE  = True

# optimal model & preprocessing parameters
model_params = {
    'n_hl' : 1, 
    'num_features' : 20000,
    'n_classes' : 20, 
    'dropout' : 0.2,
    'epochs' : 5, 
    'units' : 250, 
    'bias' : 0.1,
    'lr' : 0.01, 
    'momentum' : 0.7,
    'device' : torch.device("cpu"),
    'weights_init' : "xavier_normal",
    'hl_actfunct' : "tanh",
    'out_actfunct' : "relu",
    'loss_funct' : "cross-entropy",
    'random_state' : 1,
    'verbose' : True
}
BOW_params = {
    'bow_type' : 'counter', 
    'vocab_size' : model_params['num_features'],
    'verbose' : False, 
    'random_state' : 1
}

# load model from saved state
def load_model(MDL_PATH:str='final_model.pt') -> MLPModel_wl:
    print('Loading model..') if VERBOSE else None
    model = MLPModel_wl(**model_params)
    model.load_state_dict(torch.load(MDL_PATH))
    model.eval()
    
    print('Model loaded \n') if VERBOSE else None
    
    
    return model


# create new preprocessor 
def create_preprocessor() -> BOW:
    print('Creating preprocessor..') if VERBOSE else None
    loader = BOW(**BOW_params)
    loader.fit_transform(
        url="data/signal_20_obligatory1_train.tsv.gz",
        train_prop=0.7
    )
    
    print('Preprocessor loaded. \n') if VERBOSE else None
    
    return loader


# load new data
def load_new_data(new_data_path) -> pd.DataFrame:
    print('Loading data...') if VERBOSE else None
    data = pd.read_csv(new_data_path, sep='\t', header=0)
    
    print(f'Data loaded. data.shape={data.shape}\n') if VERBOSE else None
    

    return data


# preprocess new data
def preprocess_new_data(loader, data) -> tuple:
    print('Extract counter and encoder from loader...') if VERBOSE else None
    counter = loader.vectorizer_features
    encoder = loader.vectorizer_classes

    print('Transform text, source to vectors X, y...') if VERBOSE else None
    X_sparse = counter.transform(data.text)
    y_sparse = encoder.transform(data.source.values.reshape(-1, 1))

    print('Convert X, y to torch tensors...') if VERBOSE else None
    X = torch.from_numpy(X_sparse.toarray()).float()
    y = torch.from_numpy(y_sparse.toarray().argmax(axis=1)).long()

    print('Load complete.\n') if VERBOSE else None


    return (X, y)


# evaluation
def evaluation(X, y, model):
    print('Predicting classes...') if VERBOSE else None
    y_pred = model.predict_classes(X)
    
    # evaluate preformance
    print('Evaluating predictions...') if VERBOSE else None
    f1 = f1_score(y_pred, y, average='macro', zero_division=0)
    print(f'Final F1-score:  {f1}') if VERBOSE else None

    accuracy = accuracy_score(y_pred, y)
    print(f'Final accuracy:  {accuracy}') if VERBOSE else None

    precision = precision_score(y_pred, y, average='macro', zero_division=0)
    print(f'Final precision: {precision}') if VERBOSE else None
    
    recall = recall_score(y_pred, y, average='macro', zero_division=0)
    print(f'Final recall:    {recall}') if VERBOSE else None


# everything
def run(new_data_path):
    print(f'Running evaluation on {new_data_path}')
    # load model
    model = load_model()

    # create preprocessor
    loader = create_preprocessor()

    # load new data
    data = load_new_data(new_data_path)

    # preprocess new data
    X, y = preprocess_new_data(loader, data)
  
    # evaluate model
    evaluation(X, y, model)


if __name__=='__main__':
    # default to main data set
    TEST_URL = 'data/valid.tsv'

    # allow for path as command line arg 
    if len(sys.argv)>1:
        TEST_URL = sys.argv[1]
    else:
        print('\n\nPlease provide path to test data for evaluation')
        print('(Leave blank to use validation set)')
        arg = input('path: ')
        if len(arg)>0:
            TEST_URL = arg

    # execute evaluation
    run(TEST_URL)

