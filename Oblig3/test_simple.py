# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.model import Transformer, BertMLP, BertSimple, BertRNN

except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.model import Transformer, BertMLP, BertSimple, BertRNN

from sklearn.model_selection import train_test_split
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
con_df = filter_raw_data(df=con_df, min_entities=5)
# shrink size for quick testing
# con_df, _ = train_test_split(con_df, test_size=0.80)
# del _

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

simple_params = {
    'NORBERT': NORBERT,
    'num_labels': len(train_dataset.label_indexer), # NOTE get from model when loading saved models 
    'NOT_ENTITY_ID': train_dataset.label_indexer['O'], 
    'device': device,
    'epochs': 3,
    'lr_scheduler': False,
    'lr_factor': 0.1,
    'lr_patience': 4,
    'loss_funct': 'cross-entropy',
    'random_state': 1,
    'verbose': True,
    'lr': 0.01,
    'momentum': 0.9,
    'epoch_patience': 1,
    'label_indexer': train_dataset.label_indexer,
}

print('attempt to load model, else build new')
try:
    raise NotImplementedError('break here on purpose to avoid loading')
    print('loading state dict')
    # load instance of saved model
    loaded_model = BertSimple(**simple_params)
    # update current instance with saved values
    loaded_model.load_state_dict(
        torch.load('./saga/model_state_dict.pt')
    )
    # TODO load saved simple params
    print('local load successful!')
except:
    print('load failed. genereating new BertSimple')
    # calling transformer model
    model = BertSimple(
        **simple_params
    )

    model.fit(
        loader=train_loader,
        test=val_loader,
        verbose=True
    )

    print('finished simlpe fit. saving...')
    torch.save(model.state_dict(), './saga/simple_state_dict.pt')
    torch.save(simple_params, './saga/simple_params.pkl')
    print('finished. \n')


print('Attempt loading saved model form state dict. \n\
    loading state dict')
# load instance of saved model
loaded_params = torch.load('./saga/simple_params.pkl')
loaded_model = BertSimple(**loaded_params)
# update current instance with saved values
loaded_model.load_state_dict(
    torch.load('./saga/simple_state_dict.pt')
)


