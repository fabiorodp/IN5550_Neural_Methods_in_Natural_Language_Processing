
try:
    from Oblig3.packages.studies import RNNTypeFineTuneBert, RNNHLUnits, RNNLrMmt, \
        MLPFilterSizeFineTuneBert, MLPActivationFunction, MLPHlUnits, MLPLrOpt

except:
    from packages.studies import RNNTypeFineTuneBert, RNNHLUnits, RNNLrMmt, \
        MLPFilterSizeFineTuneBert, MLPActivationFunction, MLPHlUnits, MLPLrOpt

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
    'vocab_dir':NORBERT,
    'train_data_url':datapath,
    'verbose':True,
    'random_state':1,
    'epochs':5,
    'device':device,
    'loss_funct':'cross-entropy',
    'batch_size':32,
    'lr_scheduler':False,
    'lr_factor':0.1,
    'lr_patience':2,
    'min_entities': 5,
    'lr':0.01,
    'momentum':0.9,
    'epoch_patience':1,
    'fine_tune_bert':True,

    ########### FOR BertRNN MODEL #########################
    'n_hl':3,            # rnn: hidden layers of rnn
    'units':50,          # rnn: units per layer of rnn (default BERT val)
    'dropout':0.1,       # rnn: dropout
    'input_size':768,    # rnn: make sure model is BertModel not ForToken 
    'rnn_type':'rnn',    # rnn: rnn, gru # currently cannot use lstm
    'nonlinearity': 'tanh',     # rnn: tanh, relu, ..?
    'pool_type': 'first',       # rnn: first, last cat
    'bidirectional': False,     # rnn:
    ########################################################
}

# parameter space for each study
space = {
    'RNNTypeFineTuneBert': {
        'par_1': ['rnn', 'gru', 'lstm'],
        'par_2': [True, False] 
    },
    'RNNHLUnits': {
        'par_1': [1, 3, 5, 10],
        'par_2': [10, 50, 100, 500] 
    },
    'RNNLrMmt': {
        'par_1': [0.5, 0.1, 0.01, 0.001, 0.0001],
        'par_2': [0.9, 0.7, 0.5] 
    },
    'MLPFilterSizeFineTuneBert': {
        'par_1': [6,7], # 4,5,
        'par_2': [True, False] 
    },
    'MLPActivationFunction': {
        'par_1': ['softmax', 'relu', 'tanh'], # very bad results: 'sigmoid', 
        'par_2': ['relu', 'tanh'] # 'softmax', 'sigmoid', 
    },
    'MLPHlUnits': {
        'par_1': [1, 3, 5],
        'par_2': [50, 100, 500] 
    },
    'MLPLrOpt': {
        'par_1': [0.1, 0.01, 0.001, 0.0001],
        'par_2': ['sgd']# , 'adamw']# , 'adam'] # adam and adamw take forever!
    },
}


# ###########    RNN   #############################
# # ################# 1st study: RNNTypeFineTuneBert
# print('RNNTypeFineTuneBert')
# _ = RNNTypeFineTuneBert(
#     par_1=space['RNNTypeFineTuneBert']['par_1'],
#     par_2=space['RNNTypeFineTuneBert']['par_2'],
#     out_path_filename="outputs/RNNTypeFineTuneBert",
#     **params
# ).run()
# del _ # easy clean
# # #####################################

# # ################# 2nd study: HLUnits
# print("RNNHLUnits")
# _ = RNNHLUnits(
#     par_1=space['RNNHLUnits']['par_1'],
#     par_2=space['RNNHLUnits']['par_2'],
#     out_path_filename="outputs/RNNHLUnits",
#     **params
# ).run()
# del _
# # #####################################

# # ################# 3rd study: LrMmt
# print("RNNLrMmt")
# _ = RNNLrMmt(
#     par_1=space['RNNLrMmt']['par_1'],
#     par_2=space['RNNLrMmt']['par_2'],
#     out_path_filename="outputs/RNNLrMmt",
#     **params
# ).run()
# del _
# # #####################################

rnn_specific = ['rnn_type', 'nonlinearity', 'pool_type',\
    'bidirectional']
for key in rnn_specific:
    params.pop(key)

mlp_params = {
    'bias':0.1,
    'hl_actfunct':'tanh',
    'out_actfunct':'relu',
    'weights_init':'xavier_normal',
    'small_data': None, 
}

params.update(mlp_params)

# ############    MLP   #####################
# # ################# 1st study: label filter size versus fine tune bert
print('MLPFilterSizeFineTuneBert')
_ = MLPFilterSizeFineTuneBert(
    par_1=space['MLPFilterSizeFineTuneBert']['par_1'],
    par_2=space['MLPFilterSizeFineTuneBert']['par_2'],
    out_path_filename="outputs/MLPFilterSizeFineTuneBert",
    **params
).run()
del _ # easy clean
# # #####################################

# # ################# 2nd study: hidden activation funciton versus output
# print("MLPActivationFunction")
# _ = MLPActivationFunction(
#     par_1=space['MLPActivationFunction']['par_1'],
#     par_2=space['MLPActivationFunction']['par_2'],
#     out_path_filename="outputs/MLPActivationFunction",
#     **params
# ).run()
# del _
#####################################

# # # # ################# 3rd study: hidden layers vs units per layer
# print("MLPHlUnits")
# _ = MLPHlUnits(
#     par_1=space['MLPHlUnits']['par_1'],
#     par_2=space['MLPHlUnits']['par_2'],
#     out_path_filename="outputs/MLPHlUnits",
#     **params
# ).run()
# del _
# # # # #####################################

# # ################# 4th study: learning rate vs optimizer
# print("MLPLrOpt", params)
# _ = MLPLrOpt(
#     par_1=space['MLPLrOpt']['par_1'],
#     par_2=space['MLPLrOpt']['par_2'],
#     out_path_filename="outputs/MLPLrOpt",
#     **params
# ).run()
# del _
# #####################################

