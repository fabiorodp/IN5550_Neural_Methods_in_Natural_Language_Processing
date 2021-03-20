# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

try:
    from packages.studies import HlUnitsStudy, LrMmtStudy, EbdTypeNewVocab
    from packages.studies import ActivationFunctionStudy, VocabLossFunct
    from packages.studies import EpochsBatchesStudy
except:
    from Oblig2.packages.studies import VocabLossFunct, EbdTypeNewVocab
    from Oblig2.packages.studies import ActivationFunctionStudy
    from Oblig2.packages.studies import HlUnitsStudy, LrMmtStudy
    from Oblig2.packages.studies import EpochsBatchesStudy

import torch


def run_studies(DEVICE, verbose, save_to_path):
    # dict with current best params (updated after every study)
    params = {
        'n_hl': 1,
        'dropout': 0.2,
        'epochs': 10,
        'units': 25,
        'lr': 0.01,
        'momentum': 0.9,
        'device': torch.device(DEVICE),
        'weights_init': "xavier_normal",
        'hl_actfunct': "tanh",
        'out_actfunct': "relu",
        'loss_funct': "cross-entropy",
        'random_state': 1,
        'verbose': verbose,
        'embedding_type': "mean",
        'pre_trained_vocab': "40",
        'batch_size': 32,
    }

    # parameter space for each study
    space = {
        'vocabloss': {
            'par_1': ['1', '6', '8', '10', '12', '14', '16', '18', '20', '22',
                      '40', '82', '0', '3', '4', '5', '7', '9', '11', '13',
                      '15', '17', '19', '21', '23', '24', '25', '26', '27',
                      '28', '29', '75', '200'],
            'par_2': [
                "cross-entropy",
                "hinge-embedding",
                "bce-logit",
                "soft-margin"
            ]
        },
        'EbdTypeNewVocab': {
            'par_1': [],
            'par_2': ['mean', 'sum'],
        },
        'activation': {
            'par': ["sigmoid", "tanh", "relu", "softmax"]
            # NOTE: softmax on output (again) bc testing diff. loss func.s
        },
        'epochbatch': {
            'par_1': [5, 10, 15, 20, 30, 40, 50],
            'par_2': [32, 35, 40, 45, 50, 500],
        },
        'hlunits': {
            'par_1': [1, 2, 3, 4],
            'par_2': [5, 10, 25, 50, 100, 250, 500, 1000, 2000],
        },
        'lrmmt': {
            'par_1': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
            'par_2': [0.9, 0.7, 0.5, 0.3, 0.1, 0],
        },
    }

    # ## 1st study: preTrainedVocab Vs loss function
    params.pop('pre_trained_vocab')
    params.pop('loss_funct')
    params['pre_trained_vocab'], params['loss_funct'] = VocabLossFunct(
        par_1=space['vocabloss']['par_1'],
        par_2=space['vocabloss']['par_2'],
        out_path_filename=save_to_path+"VocabLossFunct",
        **params
    ).run()._best_params()

    # ## 2nd study: Vocab Vs embeddingTypes
    # appending the best pre-trained-vocab string in EbdTypeNewVocab -> par_1
    space['EbdTypeNewVocab']['par_1'].append(params['pre_trained_vocab'])
    params.pop('pre_trained_vocab')
    params.pop('embedding_type')
    params['pre_trained_vocab'], params['embedding_type'] = EbdTypeNewVocab(
        par_1=space['EbdTypeNewVocab']['par_1'],
        par_2=space['EbdTypeNewVocab']['par_2'],
        out_path_filename=save_to_path+"EbdTypeNewVocab",
        **params
    ).run()._best_params()

    # ## 3rd study: Activation function in hidden layer and output layer
    params.pop('hl_actfunct')
    params.pop('out_actfunct')
    params['hl_actfunct'], params['out_actfunct'] = ActivationFunctionStudy(
        par_1=space['activation']['par'],
        par_2=space['activation']['par'],
        out_path_filename=save_to_path+"ActFunct",
        **params
    ).run()._best_params()

    # ## 4th study: Bacth size and epoch number
    params.pop('epochs')
    params.pop('batch_size')
    params['epochs'], params['batch_size'] = EpochsBatchesStudy(
        par_1=space['epochbatch']['par_1'],
        par_2=space['epochbatch']['par_2'],
        out_path_filename=save_to_path+"EpochsBatches",
        **params
    ).run()._best_params()

    # ## 5th study: Number of hidden layers and units per layer
    params.pop('n_hl')
    params.pop('units')
    params['n_hl'], params['units'] = HlUnitsStudy(
        par_1=space['hlunits']['par_1'],
        par_2=space['hlunits']['par_2'],
        out_path_filename=save_to_path+"HlUnits",
        **params
    ).run()._best_params()

    # ## 6th study: Learning rate and momentum
    params.pop('lr')
    params.pop('momentum')
    params['lr'], params['momentum'] = LrMmtStudy(
        par_1=space['lrmmt']['par_1'],
        par_2=space['lrmmt']['par_2'],
        out_path_filename=save_to_path+"LrMmt",
        **params
    ).run()._best_params()


if __name__ == '__main__':
    run_studies(DEVICE='cpu', verbose=True, save_to_path='outputs/')
