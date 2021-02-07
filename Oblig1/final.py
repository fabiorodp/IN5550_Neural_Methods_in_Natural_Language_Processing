from packages.ann_models import MLPModel_wl
from packages.preprocessing import BOW
import torch


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
    'verbose' : model_params['verbose'], 
    'random_state' : 1
}


tensors = BOW(**BOW_params)

tensors.fit_transform(
    url="data/signal_20_obligatory1_train.tsv.gz",
    train_prop=0.7
)

final_model = MLPModel_wl(**model_params, loader=tensors)

final_model.fit(loader=(tensors.X_train, tensors.y_train))

torch.save(final_model.state_dict(), "final_model.pt")