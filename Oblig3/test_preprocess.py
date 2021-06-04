# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from Oblig3.packages.preprocess import load_raw_data, pad, OurCONLLUDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd


# first step
# datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
# NORBERT = 'cluster/shared/nlpl/data/vectors/lastest/216'
datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
NORBERT = 'Oblig3/saga/216/'

# # loading raw data
# con_df = load_raw_data(datapath)
#
# # splitting data
# train_df, val_df = train_test_split(
#     con_df,
#     test_size=0.25,
#     random_state=1,
#     shuffle=True,
# )

sentences_ie = [
    'Nominasjonskampen i Oslo SV mellom Heikki Holmås og Akhtar Chaudhry i desember i fjor handlet blant annet om beskyldninger om juks.',
    'I Malaysia har det nettopp vært vaktskifte:',
    'Kapitalbevegelsene over landegrensene ble så enorme at IMFs pengebinge viste seg altfor liten da organisasjonen skulle hjelpe de landene som ble rammet.',
    'Foto:'
]

labels_ie = [
    ['O', 'O', 'B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-GPE_LOC', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'O']
]

df_short = pd.DataFrame(zip(sentences_ie, labels_ie),
                        columns=['sentence', 'labels'])

tokenizer = BertTokenizer.from_pretrained(NORBERT)


train_ie = OurCONLLUDataset(df=df_short, tokenizer=tokenizer)

train_ie_loader = DataLoader(
    train_ie,
    batch_size=2,
    collate_fn=lambda batch: pad(batch, train_ie.IGNORE_ID)
)

print(next(iter(train_ie_loader)))

# >>
# (tensor([
# [   102,  4467,  3060, 25518,  8681,  2925,  6481,  3529, 14003,  5608,
#     11619,  2932,  7899, 28393,  2857,  9817, 32409,  1682, 26411, 11598,
#     4126,  2925, 31697,  2925, 27514,  9347, 17461, 18950,  6083, 16803,
#     10351,  6083, 20579,  1113,   103],
#
# [   102, 18396, 23473,  3721,  4226,  3493,  4525, 12944,  6103, 20058,
#     1272,   103,     0,     0,     0,     0,     0,     0,     0,     0,
#     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#     0,     0,     0,     0,     0]]),
#
#  tensor([
# [
#
# [   '[CLS]'=6, 'No'=2, '##min'=2, '##asjons'=2, '##kampen'=2, 'i'=2,
# 'Oslo'=3, 'SV'=4, 'mellom'=2, 'Hei'=5, '##kk'=5, '##i'=5, 'Holm'=1,
# '##ås'=1, 'og'=2, 'Ak'=5, '##htar'=5, 'Ch'=1, '##aud'=1, '##hr'=1,
# '##y'=1, 'i'=2, 'desember'=2, 'i'=2, 'fjor'=2, 'handlet'=2, 'blant'=2,
# 'annet'=2, 'om'=2, 'beskyld'=2, '##ninger'=2, 'om'=2, 'juks'=2, '##.'=2,
# '[SEP]'=6    ],
#
# [   6, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#     6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]]),
#
#  tensor([
# [   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#
# [   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
# <<

