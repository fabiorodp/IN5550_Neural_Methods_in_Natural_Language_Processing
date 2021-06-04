import pandas as pd
from transformers import BertForTokenClassification, BertTokenizer


data_file = 'IN5550/exam/data/train.conll'


sents, all_labels = [], []
sent, labels = [], []
for line in open(data_file):
    if line.strip() == "":
        sents.append(sent)
        all_labels.append(labels)
        sent, labels = [], []
    else:
        token, label = line.strip().split("\t")
        sent.append(token)
        labels.append(label)

BIO_labels, polarity_labels = [], []
for row in all_labels:
    bio, polarity = [], []
    for label in row:
        if label == 'O':
            polarity.append(label)
        else:
            polarity.append(label.split('-')[2])
        
        bio.append(label.split('-')[0])
    BIO_labels.append(bio)
    polarity_labels.append(polarity)


NORBERT = '/cluster/shared/nlpl/data/vectors/latest/216'
tokenizer = BertTokenizer.from_pretrained(NORBERT)

idx = 0
input_ids = tokenizer(sents[idx], is_split_into_words=True)['input_ids']
attention_mask = tokenizer(sents[idx], is_split_into_words=True)['attention_mask']
encoded_sentence = tokenizer.tokenize(' '.join(sents[idx]))
decoded_sentence = tokenizer.decode(input_ids)

indexer = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                          "B-targ-Negative": 3, "I-targ-Negative": 4}
BIO_indexer = {
    "O":0,
    "I":1,
    "B":2,
}
polarity_indexer = {
    "O":0,
    "Positive":1,
    "Negative":2,
}

y_BIO = [
    BIO_indexer[bt]
    for bt in BIO_labels[idx]
]
y_polarity = [
    polarity_indexer[bt]
    for bt in polarity_labels[idx]
]
y = [
    indexer[bt]
    for bt in all_labels[idx]
]
### build y mask
IGNORE_ID = indexer['O']
tok_sent = [tokenizer.convert_ids_to_tokens(i) for i in input_ids]
ignore_mask = []
for i, token in enumerate(tok_sent[idx]):
    if token.startswith('##'):
        ignore_mask.append(IGNORE_ID)
    elif token in tokenizer.all_special_tokens:
        ignore_mask.append(IGNORE_ID)
    else:
        ignore_mask.append(1)



print('the end')