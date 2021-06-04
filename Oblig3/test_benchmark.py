# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.model import Transformer

except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.model import Transformer

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch


# first step
# datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
# NORBERT = '/cluster/shared/nlpl/data/vectors/latest/216'
datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
NORBERT = 'Oblig3/saga/216/'

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# loading raw data
con_df = load_raw_data(datapath=datapath)
con_df = filter_raw_data(df=con_df, min_entities=5)

# splitting data
train_df, val_df = train_test_split(
    con_df,
    # train_size=0.50,
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

# calling transformer model
transformer = Transformer(
    NORBERT=NORBERT,
    num_labels=len(train_dataset.label_indexer),
    NOT_ENTITY_ID=train_dataset.label_indexer['O'],
    device=device,
    epochs=100,  # 12 for the optimal
    lr_scheduler=False,
    factor=0.1,
    patience=2,
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr=0.01,
    momentum=0.9,
    epoch_patience=1,  # 0 for the optimal
    label_indexer=train_dataset.label_indexer
)

transformer.fit(
    loader=train_loader,
    test=val_loader,
    verbose=True
)

torch.save(transformer, "transformer_benchmark_12ep.pt")
