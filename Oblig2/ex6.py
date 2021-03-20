# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig2.packages.rnn_studies import FactorPatience, BestEpochRNNType
    from Oblig2.packages.rnn_studies import VocabRnnType, LrMmt, BatchEpoch
    from Oblig2.packages.rnn_studies import RnnTypeBiDirectional, HLUnits
    from Oblig2.packages.rnn_studies import PoolTypeRNNType
except:
    from packages.rnn_studies import FactorPatience, BestEpochRNNType
    from packages.rnn_studies import VocabRnnType, LrMmt, BatchEpoch
    from packages.rnn_studies import RnnTypeBiDirectional, HLUnits
    from packages.rnn_studies import PoolTypeRNNType


DIR = '/cluster/shared/nlpl/data/vectors/latest/'
# DIR = 'saga/'
URL = 'data/stanford_sentiment_binary.tsv.gz'
verbose = True


# dict with current best params (updated after every study)
params = {
    'n_hl': 1,
    'dropout': 0.2,
    'epochs': 50,               # eivind:25, fabio:10, per:20
    'units': 75,                # eivind:80, fabio:10, per:30
    'lr': 0.01,
    'momentum': 0.9,            # eivind:.9, fabio:.5, per:.3
    'device': "cpu",
    'loss_funct': "cross-entropy",
    'random_state': 1,          # TODO try different radnom states
    'verbose': verbose,
    'batch_size': 32,
    'vocab': '22',               # other good ones were 22, 16, 82
    'rnn_type': 'rnn',
    'bidirectional': True,     # checks in first study
    'freeze': True,            # really no difference, only training time
    'lr_scheduler': True,       # eivind: True, fabio: False, per: True
    'factor': 0.1,
    'patience': 2,
    'pool_type': 'last',       # eivind: last, fabio: first, per: cat
}

# parameter space for each study
space = {
    'RnnTypeBiDirectional': {
        'par_1': ['rnn', 'lstm', 'gru'],
        'par_2': [False, True]
    },
    'VocabRnnType': {
        'par_1': ['1', '6', '8', '10', '12', '14', '16', '18', '20', '22',
                  '40', '82', '0', '3', '4', '5', '7', '9', '11', '13',
                  '15', '17', '19', '21', '23', '24', '25', '26', '27',
                  '28', '29', '75', '200'],
        'par_2': ['rnn', 'lstm', 'gru'],
    },
    'PoolTypeRnnType': {
        'par_1': ['last', 'first', 'cat'],
        'par_2': ['rnn', 'lstm', 'gru'],
    },
    'BestEpochRnnType': {
        'par_1': [200],   # this should just be an int...
        'par_2': ['rnn', 'lstm', 'gru'],
    },
    'HLUnits': {
        'par_1': [1, 2, 3, 4],                      # hidden layers
        'par_2': [5, 10, 25, 50, 100, 250, 500],    # units
    },
    'LrMmt': {
        'par_1': [10.0, 1.0, 0.9, 0.1, 0.01, 0.001, 0.0001],
        'par_2': [0.9, 0.7, 0.5, 0.3, 0.1, 0],      # momentum
    },
    'FactorPatience': {
        'par_1': [0.01, 0.05, 0.1, 0.5],
        'par_2': [1, 2, 3, 4, 5],
    },
    'BatchEpoch': {
        'par_1': [10, 32, 36, 40, 44, 48, 200],
        'par_2': [5, 10, 30, 50, 100],
    }
}

# ################# 1st study: (RnnType Vs Bidirectional) with freeze=False
print("RnnType Vs Bidirectional with freeze=False")
params.pop('rnn_type')
params.pop('bidirectional')
params.pop('freeze')
params['freeze'] = False
print(f"freeze = {params['freeze']}")
params['rnn_type'], params['bidirectional'] = RnnTypeBiDirectional(
    par_1=space['RnnTypeBiDirectional']['par_1'],
    par_2=space['RnnTypeBiDirectional']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    out_path_filename="outputs/RnnTypeBiDirectional_freeze_false",
    need_valid=False,
    **params
).run()._best_params()
# #####################################

# ################# 2nd study: (RnnType Vs Bidirectional) with freeze=True
print("RnnType Vs Bidirectional with freeze=True")
params.pop('rnn_type')
params.pop('bidirectional')
params.pop('freeze')
params['freeze'] = True
print(f"freeze = {params['freeze']}")
params['rnn_type'], params['bidirectional'] = RnnTypeBiDirectional(
    par_1=space['RnnTypeBiDirectional']['par_1'],
    par_2=space['RnnTypeBiDirectional']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/RnnTypeBiDirectional_freeze_true",
    **params
).run()._best_params()
# #####################################

# ##################################### 3rd study: (Vocabs Vs RNNTypes)
print("Vocabs Vs RNNTypes...")
params.pop('vocab')
params.pop('rnn_type')
print(f"bidirectional = {params['bidirectional']}")
print(f"freeze = {params['freeze']}")
params['vocab'], params['rnn_type'] = VocabRnnType(
    par_1=space['VocabRnnType']['par_1'],
    par_2=space['VocabRnnType']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/VocabRnnType",
    **params
).run()._best_params()
# #####################################

# ##################################### 4th study: Pool types
print("Pool type Vs RNNTypes...")
params.pop('pool_type')
params.pop('rnn_type')
# show previous best params
print(f"bidirectional = {params['bidirectional']}")
print(f"freeze = {params['freeze']}")
print(f"vocab = {params['vocab']}")
params['pool_type'], params['rnn_type'] = PoolTypeRNNType(
    par_1=space['PoolTypeRnnType']['par_1'],
    par_2=space['PoolTypeRnnType']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/PoolTypeRnnType",
    **params
).run()._best_params()
# #####################################

# ##################################### 5th Study: Best Epoch vs RNN type
print("Best Epoch Vs RNNTypes...")
params.pop('epochs')
params.pop('rnn_type')

# show previous best params
print(f'Current params {params}')

params['epochs'], params['rnn_type'] = BestEpochRNNType(
    par_1=space['BestEpochRnnType']['par_1'],
    par_2=space['BestEpochRnnType']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=True,
    out_path_filename="outputs/BestEpochRnnType",
    **params
).run()._best_params()

# appending the best number of epochs to be tested in the 'BatchEpoch' study
space['BatchEpoch']['par_2'].append(params['epochs'])
# #####################################

# ##################################### 6th study: (HL Vs Units)
print("HL Vs Units...")
params.pop('n_hl')
params.pop('units')

# show previous best params
print(f'Current params {params}')

params['n_hl'], params['units'] = HLUnits(
    par_1=space['HLUnits']['par_1'],
    par_2=space['HLUnits']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/HLUnits",
    **params
).run()._best_params()
# #####################################

# ###########################  7th study: (Lr Vs Mmt) with lr_scheduler=False
print("Lr Vs Mmt with lr_scheduler=False...")
params.pop('lr')
params.pop('momentum')

# show previous best params
print(f'Current params {params}')

params['lr'], params['momentum'] = LrMmt(
    par_1=space['LrMmt']['par_1'],
    par_2=space['LrMmt']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/LrMmt_scheduler_false",
    **params
).run()._best_params()
# #####################################

# ################# 8th study: (Factor Vs Patience) with lr_scheduler=True
print("Lr Vs Mmt with lr_scheduler=True...")
params.pop('factor')
params.pop('patience')
params.pop('lr_scheduler')
params['lr_scheduler'] = True

# show previous best params
print(f'Current params {params}')

params['factor'], params['patience'] = FactorPatience(
    par_1=space['FactorPatience']['par_1'],
    par_2=space['FactorPatience']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/FactorPatience",
    **params
).run()._best_params()
# #####################################

# #####################################  9th study: (Batches Vs Epochs)
print("Batches Vs Epochs...")
params.pop('batch_size')
params.pop('epochs')

# show previous best params
print(f'Current params {params}')

params['batch_size'], params['epochs'] = BatchEpoch(
    par_1=space['BatchEpoch']['par_1'],
    par_2=space['BatchEpoch']['par_2'],
    vocab_dir=DIR,
    train_data_url=URL,
    need_valid=False,
    out_path_filename="outputs/BatchEpoch",
    **params
).run()._best_params()

print(f"batch_size = {params['batch_size']}")
print(f"epochs = {params['epochs']}")
print("Studies done!")
# #####################################
