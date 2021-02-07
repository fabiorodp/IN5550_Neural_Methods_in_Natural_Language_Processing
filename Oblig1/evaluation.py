# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from packages.preprocessing import Signal20Dataset
from packages.ann_models import MLPModel_wl
from packages.ann_models import MLPModel
from packages.preprocessing import BOW
from packages.studies import ActivationFunctionStudy, BoWStudy
from packages.studies import HlUnitsStudy, LrMmtStudy
from packages.studies import EpochsBatchesStudy
import torch


actFunct = ActivationFunctionStudy(
    par_1=["sigmoid", "tanh", "relu", "softmax"],
    par_2=["sigmoid", "tanh", "relu", "softmax"],
    random_state=1,
    verbose=True,
    device=torch.device("cpu"),
    out_path_filename="output/ActFunct"
).run(
    url='data/signal_20_obligatory1_train.tsv.gz',
    pos=None,
    bow_type="counter",
    vocab_size=5000,
    batch_size=32,
    train_size=0.7,
    n_hl=1,
    dropout=0.2,
    epochs=20,
    units=25,
    bias=0.1,
    lr=0.01,
    momentum=0.9,
    weights_init="xavier_normal",
    loss_funct="cross-entropy"
)


BoW = BoWStudy(
    par_1=[30000, 25000, 20000, 15000, 10000, 5000, 1000,
           750, 500, 250, 100, 50],
    par_2=["counter", "binary", "tfidf"],
    random_state=1,
    verbose=True,
    device=torch.device("cpu"),
    out_path_filename="output/BoW"
).run(
    url='data/signal_20_obligatory1_train.tsv.gz',
    pos=None,
    batch_size=32,
    train_size=0.7,
    n_hl=1,
    dropout=0.2,
    epochs=20,
    units=25,
    bias=0.1,
    lr=0.01,
    momentum=0.9,
    weights_init="xavier_normal",
    loss_funct="cross-entropy",
    hl_actfunct="tanh",
    out_actfunct="relu",
)

# PoS Study
name = ["output/verb_BoW", "output/adj_BoW", "output/propn_BoW",
        "output/noun_BoW", "output/adv_BoW", "output/noun_propn_BoW",
        "output/noun_propn_verb_BoW", "output/noun_propn_verb_adj_BoW",
        "output/noun_propn_verb_num_BoW"]

comb = [["_VERB"], ["_ADJ"], ["_PROPN"], ["_NOUN"], ["_ADV"],
        ["_NOUN", "_PROPN"], ["_NOUN", "_PROPN", "_VERB"],
        ["_NOUN", "_PROPN", "_VERB", "_ADJ"],
        ["_NOUN", "_PROPN", "_VERB", "_NUM"]]

for n, c in zip(name, comb):
    BoW = BoWStudy(
        par_1=[20000],
        par_2=["binary"],
        random_state=1,
        verbose=True,
        device=torch.device("cpu"),
        out_path_filename=n
    ).run(
        url='data/signal_20_obligatory1_train.tsv.gz',
        pos=c,
        batch_size=32,
        train_size=0.7,
        n_hl=1,
        dropout=0.2,
        epochs=20,
        units=25,
        bias=0.1,
        lr=0.01,
        momentum=0.9,
        weights_init="xavier_normal",
        loss_funct="cross-entropy",
        hl_actfunct="tanh",
        out_actfunct="relu",
    )

EpochsBatches = EpochsBatchesStudy(
    par_1=[3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20],
    par_2=[32, 36, 40, 45, 50, 55, 60, 500],
    random_state=1,
    verbose=True,
    device=torch.device("cpu"),
    out_path_filename="output/EpochsBatches"
).run(
    url='data/signal_20_obligatory1_train.tsv.gz',
    pos=None,
    train_size=0.7,
    dropout=0.2,
    bias=0.1,
    weights_init="xavier_normal",
    loss_funct="cross-entropy",
    hl_actfunct="tanh",
    out_actfunct="relu",
    vocab_size=20000,
    bow_type="binary",
    n_hl=1,
    units=25,
    lr=0.01,
    momentum=0.9,
)

HlUnits = HlUnitsStudy(
    par_1=[1, 2, 3, 4, 5, 8, 11, 14],
    par_2=[10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000],
    random_state=1,
    verbose=True,
    device=torch.device("cpu"),
    out_path_filename="output/HlUnits"
).run(
    url='data/signal_20_obligatory1_train.tsv.gz',
    pos=None,
    batch_size=32,
    train_size=0.7,
    dropout=0.2,
    epochs=5,
    bias=0.1,
    lr=0.01,
    momentum=0.9,
    weights_init="xavier_normal",
    loss_funct="cross-entropy",
    hl_actfunct="tanh",
    out_actfunct="relu",
    vocab_size=20000,
    bow_type="binary"
)

LrMmt = LrMmtStudy(
    par_1=[0.9, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
    par_2=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0],
    random_state=1,
    verbose=True,
    device=torch.device("cpu"),
    out_path_filename="output/LrMmt"
).run(
    url='data/signal_20_obligatory1_train.tsv.gz',
    pos=None,
    batch_size=32,
    train_size=0.7,
    dropout=0.2,
    epochs=5,
    bias=0.1,
    weights_init="xavier_normal",
    loss_funct="cross-entropy",
    hl_actfunct="tanh",
    out_actfunct="relu",
    vocab_size=20000,
    bow_type="binary",
    n_hl=1,
    units=250
)


