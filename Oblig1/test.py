# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from Oblig1.packages.ann_models import MLPModel
import pandas as pd
import torch

# determine what device to use
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

# loading  df
df = pd.read_csv('Oblig1/data/sample.tsv', sep='\t',
                 header=0, compression='infer')

# Extracting train
# text = list(df.text.str.split('\t'))
samples = df.text.values
sources = df.source.values

# x = np.array([i[0] for i in text], dtype=str)
x = samples
y = sources.reshape(-1, 1)

# #######
vocab_size = 5
vectorizer_features = CountVectorizer(max_features=vocab_size)
X_train = vectorizer_features.fit_transform(x)
X_train_array = X_train.toarray()

# ##### checking
keys = vectorizer_features.get_feature_names()
values = X_train_array.sum(axis=0).tolist()
d = {k: [v] for k, v in zip(keys, values)}
df_counts = pd.DataFrame(d)

# ######## collections.Counter
from collections import Counter

flat_text = []
for sample in samples:
    flat_text += sample.split()

tokens = Counter(flat_text)

if vocab_size is None:
    vocab_size = len(tokens.keys())

# get unique words in set of training data
text_vocab = [i[0] for i in tokens.most_common(vocab_size)]

# ########

vectorizer_classes = OneHotEncoder()
Y_train = vectorizer_classes.fit_transform(y)
Y_train_array = Y_train.toarray()
Y_tranformed = Y_train_array.argmax(axis=1)

input_tensor = torch.from_numpy(X_train_array).float()
target = torch.from_numpy(Y_tranformed).long()

model = MLPModel(num_features=934, n_classes=7,
                 dropout=0.25, epochs=50, units=5,
                 bias=0.1, weight_init_method="xavier_normal",
                 device=torch.device("cpu"),
                 random_state=10)

model.fit(input_tensor, target)
