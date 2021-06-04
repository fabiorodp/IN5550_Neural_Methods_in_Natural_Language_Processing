# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from Oblig3.packages.preprocess import OurCONLLUDataset, load_raw_data, Dataset
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.model_selection import train_test_split
from Oblig3.packages.model import Transformer
from torch.utils.data import DataLoader
import torch
from keras.preprocessing.sequence import pad_sequences


# first step
# datapath = '/cluster/projects/nn9851k/IN5550/norne-nb-in5550-train.conllu'
# NORBERT = 'cluster/shared/nlpl/data/vectors/lastest/216'
datapath = 'Oblig3/saga/norne-nb-in5550-train.conllu'
NORBERT = 'Oblig3/saga/216/'

# loading raw data
data = load_raw_data(datapath)

# tokenizer = BertTokenizer.from_pretrained(NORBERT, do_lower_case=False)
# tokenizer('jeg bor i brasil .')
# tokenizer.tokenize('jeg bor i Brasil')
# tokenizer.encode(['jeg', 'bor', 'i', 'Brasil'])
# tokenizer.decode([102, 9764, 2279, 2925, 27438, 5191, 1113, 103])

dataset = Dataset(NORBERT, data)
dataset1 = OurCONLLUDataset(data)

b_input_ids, b_labels, b_input_mask = dataset[0]

# loading NORBERT
model = BertForTokenClassification.from_pretrained(
    NORBERT,
    num_labels=18,
    output_attentions=False,
    output_hidden_states=False
)

configuration = model.config

y_pred_dist = model(
    b_input_ids,
    token_type_ids=None,
    attention_mask=b_input_mask,
    labels=b_labels
)

y_pred = y_pred_dist.logits.max(dim=1)[1]  # subscript 0 return prob_dist
