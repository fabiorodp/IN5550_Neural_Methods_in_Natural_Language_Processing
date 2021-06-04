# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.model import Transformer, BertRNN
    from Oblig3.packages.ner_eval import Evaluator

except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.model import Transformer, BertRNN
    from packages.ner_eval import Evaluator

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch


# first step
datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
NORBERT = '/cluster/shared/nlpl/data/vectors/latest/216'
# datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
# NORBERT = 'Oblig3/saga/216/'

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# loading raw data
con_df = load_raw_data(datapath=datapath)
con_df = filter_raw_data(df=con_df, min_entities=2, max_entities=5)

# #  reduce size even more
# con_df, _ = train_test_split(con_df, train_size=0.20)
# del _

# for testing Evaluate
con_df, eval_df = train_test_split(con_df, train_size=0.5)


# splitting data
train_df, val_df = train_test_split(
    con_df,
    test_size=0.25,
    random_state=1,
    shuffle=True,
)

tokenizer = BertTokenizer.from_pretrained(NORBERT)

# creating data sets
train_dataset = OurCONLLUDataset(
    df=train_df,
    tokenizer=tokenizer,
    device=device
)

val_dataset = OurCONLLUDataset(
    df=val_df,
    tokenizer=tokenizer,
    label_vocab=train_dataset.label_vocab,
    device=device
)

# creating data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
    collate_fn=lambda batch: pad(batch, train_dataset.IGNORE_ID)
)

params = {
    'NORBERT':NORBERT,
    'num_labels':len(train_dataset.label_indexer),
    'NOT_ENTITY_ID':train_dataset.label_indexer['O'],
    'device':device,
    'epochs':15,
    'epoch_patience':1,
    'fine_tune_bert':True,
    'loss_funct':'cross-entropy',
    'lr':0.1,
    'lr_scheduler':False,
    'lr_factor':0.1,
    'lr_patience':2,
    'momentum':0.9,
    'random_state':1,
    'verbose':True,
    'label_indexer':train_dataset.label_indexer,

    ########### FOR BertRNN MODEL #########################
    'bidirectional': True,   # rnn:
    'dropout':0.1,       # rnn: dropout
    'input_size':768,    # rnn: make sure model is BertModel not ForToken 
    'nonlinearity': 'tanh',   # rnn: tanh, relu, ..?
    'n_hl':3,            # rnn: hidden layers of rnn
    'pool_type': 'first',      # rnn: first, last cat
    'rnn_type':'rnn',    # rnn: rnn, lstm, gru
    'units':100,           # rnn: units per layer of rnn (default BERT val)
    ########################################################
}

# calling transformer model
model = BertRNN(
    **params
)

model.fit(
    loader=train_loader,
    # test=val_loader,
    verbose=True
)

model._eval(val_loader)

# save if better than saved¨
modelpath = '/cluster/projects/nn9851k/IN5550/pmhalvor/bert_rnn.pt'
try:
    saved_model = torch.load(modelpath)
    if model.eval > saved_model.eval:
        torch.save(model, modelpath)
except:
    torch.save(model, modelpath)

