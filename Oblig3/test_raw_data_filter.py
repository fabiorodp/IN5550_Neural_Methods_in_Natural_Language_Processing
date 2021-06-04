# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
except:
    from packages.preprocess import load_raw_data, pad
    from packages.preprocess import OurCONLLUDataset

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.utils import resample
import pandas as pd
import nltk


# first step
# datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
# NORBERT = 'cluster/shared/nlpl/data/vectors/lastest/216'
datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
NORBERT = 'Oblig3/saga/216/'


def train_test_split(df, train_prop, random_state=1):
    n = len(df) * train_prop

    # split train/valid
    df_train = resample(
        df,
        replace=False,
        # stratify=df.label,
        n_samples=n,
        random_state=random_state
    )

    df_test = df[~df.index.isin(df_train.index)]

    return df_train, df_test


# loading raw data
con_df = load_raw_data(datapath)

# removing sentences with only 'O' labels
row_idxs_to_drop = []
for row_idx, row in enumerate(con_df.iloc[:, 2]):
    labels = []

    for e in row:
        if e != 'O':
            labels.append(e)

    if len(labels) < 3:
        row_idxs_to_drop.append(row_idx)

train_df = con_df.drop(row_idxs_to_drop, axis=0)
