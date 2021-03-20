# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig2.packages.preprocessing import load_embedding, TSVDataset
    from Oblig2.packages.preprocessing import pad_batches
    from Oblig2.packages.ann_models import MLPModel
except:
    from packages.preprocessing import load_embedding, TSVDataset
    from packages.preprocessing import pad_batches
    from packages.ann_models import MLPModel

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch


# DIR = "/cluster/shared/nlpl/data/vectors/latest/"
DIR = "Oblig2/saga/"
VOCAB = "40"
TAGGED = False
DEVICE = "cpu"


print('\n\n\nload embeddings...')
embedding = load_embedding(DIR + VOCAB + '.zip')

embedding.add(
    '<pad>',
    weights=torch.zeros(embedding.vector_size)
)

pad_idx = embedding.vocab['<pad>'].index

print('\n\n\nloading data set...')
data = TSVDataset(
    embedder=embedding,
    url='Oblig2/data/stanford_sentiment_binary.tsv.gz',
    pos_tagged=TAGGED,
    random_state=1,
    device=DEVICE
)

# import pandas as pd
# df = pd.read_csv('Oblig2/data/stanford_sentiment_binary.tsv.gz', sep='\t')

print('\n\n\nloading data loader...')
loader = DataLoader(
    data,
    batch_size=32,
    collate_fn=lambda x: pad_batches(x, pad_idx)
)

# X, y = next(iter(loader))
# input:(32, 43 -> **300) -> HL:(43 -> **300, 25) -> output:(25, 2)
# ** mean of word_embedding inside forward method

print('\n\n\nloading model...')
model = MLPModel(
    emb=embedding,
    num_features=embedding.vector_size,
    loss_funct="cross-entropy",
    random_state=1,
    verbose=True,
    epochs=10,
    device=torch.device(DEVICE)
)

print('\n\n\nfitting model...')
model.fit(loader=loader, verbose=True)

y_pred = model.predict_classes(data.get_embedded_test_tensor())
y_pred = y_pred.to(torch.device("cpu"))

print('\n\n\nscoring model...')
gold = [torch.LongTensor([y]) for y in data.label_test]
gold = torch.stack(gold)
gold = gold.to(torch.device("cpu"))

f1score = f1_score(
    gold,
    y_pred,
    average="macro",
    zero_division=0
)

print(f'F1-score: {f1score}')

# X, y = next(iter(loader))
#
# m = torch.nn.Sigmoid()
# loss = torch.nn.BCELoss()
# input = torch.randn(3, requires_grad=True)
# target = torch.empty(3).random_(2)
# N = m(input)
# output = loss(N, target)
# output.backward()
#

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder(sparse=False)
# z1 = encoder.fit_transform(y)
# z2 = torch.from_numpy(z1)

