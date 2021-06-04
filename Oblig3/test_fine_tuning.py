# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.studies import FreezerStudy
    from Oblig3.packages.model import Transformer

except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.studies import FreezerStudy
    from packages.model import Transformer

import torch


# first step
datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
NORBERT = '/cluster/shared/nlpl/data/vectors/latest/216'
# datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
# NORBERT = 'Oblig3/saga/216/'

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# dict with current best params (updated after every study)
params = {
    'vocab_dir': NORBERT,
    'train_data_url': datapath,
    'verbose': True,
    'random_state': 1,
    'epochs': 100,
    'device': device,
    'loss_funct': 'cross-entropy',
    'batch_size': 32,
    'freeze': True,
    'lr_scheduler': False,
    'factor': 0.01,
    'patience': 4,
    'min_entities': 5,
    'lr': 0.01,
    'momentum': 0.9,
    'epoch_patience': 1
}

# parameter space for each study
space = {
    'AttentionFreezerStudy': {
        'par_1': ['attention', ['query', 'key'], ['key', 'value'], ['query', 'value']],
        'par_2': [5, 10, 15, 20] # max number of frozen parameters (5 attentions per layer)
    },
    'LayerFreezerStudy': {
        'par_1': ['layer', ['layer.9.', 'layer.10.', 'layer.11.'],['layer.4.','layer.5.','layer.6.']],
        'par_2': [8, 16, 32] # max number of frozen parameters (8 parameters per layer)
    }
}

# ################# 1st study: AttentionFreezerStudy
print("AttentionFreezerStudy")
FreezerStudy(
    par_1=space['AttentionFreezerStudy']['par_1'],
    par_2=space['AttentionFreezerStudy']['par_2'],
    out_path_filename="outputs/AttentionFreezerStudy",
    **params
).run()
# #####################################

# ################# 2nd study: LayerFreezerStudy
print("LayerFreezerStudy")
FreezerStudy(
    par_1=space['LayerFreezerStudy']['par_1'],
    par_2=space['LayerFreezerStudy']['par_2'],
    out_path_filename="outputs/LayerFreezerStudy",
    **params
).run()
# #####################################
