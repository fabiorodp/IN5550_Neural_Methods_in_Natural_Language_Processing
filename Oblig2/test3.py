# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig2.packages.preprocessing import load_embedding, collate_fn
    from Oblig2.packages.preprocessing import RNNDataset, process_raw_data
    from Oblig2.packages.ann_models import RNNModel
except:
    from packages.preprocessing import load_embedding, collate_fn
    from packages.preprocessing import RNNDataset, process_raw_data
    from packages.ann_models import RNNModel


from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
import pickle
import torch
import json
import pickle


DIR = '/cluster/shared/nlpl/data/vectors/latest/'
# DIR = 'saga/'
VOCAB = '16'
data_url = "data/stanford_sentiment_binary.tsv.gz"


try:
    with open('embedder.pkl', 'rb') as f:
        embedder = pickle.load(f)

    with open('train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open('test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    print('\npickle worked. moving along\n')
except:
    print('\npickle did not work. regenerating\n')
    store = True

        
    embedder = load_embedding(DIR + VOCAB + '.zip')
    embedder.add('<unk>', weights=torch.rand(embedder.vector_size))
    embedder.add('<pad>', weights=torch.zeros(embedder.vector_size))

    df_train, df_test = process_raw_data(
        data_url=data_url,
        train_prop=0.75,
        verbose=True,
        pos_tagged=False,
        random_state=1
    )

    # NOTE: new way to load dataset is to just feed it the dataframe preloaded
    train_data = RNNDataset(            # NOTE Check preprocessing for new dataset type
        embedder=embedder,
        df=df_train.iloc[:500],     # NOTE: Make smaller data set for quicker testing
        device="cpu",
        random_state=1,
        label_vocab=None,
        verbose=True
    )

    test_data = RNNDataset(
        embedder=embedder,
        df=df_test.iloc[:200],      # NOTE: Make smaller data set for quicker testing
        device="cpu",
        random_state=1,
        label_vocab=train_data.label_vocab, # NOTE: genius!
        verbose=True
    )

    del df_test, df_train

    # -> X[0] : tensor.size([num_samples, longest_sentence]) := words indices,
    # -> X[1] : tensor.size([num_samples]) := words_lengths,
    # -> y : tensor.size([num_samples, longest_sentence]) := labels.

    if store:
        with open('embedder.pkl', 'wb+') as f:
            pickle.dump(embedder, f)
        with open('train_data.pkl', 'wb+') as f:
            pickle.dump(train_data, f)
        with open('test_data.pkl', 'wb+') as f:
            pickle.dump(test_data, f)
        # with open('train_loader.pkl', 'wb+') as f:
        #     pickle.dump(train_loader, f)
        # with open('test_loader.pkl', 'wb+') as f:
        #     pickle.dump(test_loader, f)

pad_token = embedder.vocab['<pad>'].index

# build and pad with loaders
# -> X[0] : tensor.size([num_samples, longest_sentence]) := words indices,
# -> X[1] : tensor.size([num_samples]) := words_lengths,
# -> y : tensor.size([num_samples, longest_sentence]) := labels.
train_loader = DataLoader(
    train_data,
    batch_size=20,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, pad_token, device='cpu')
)

test_loader = DataLoader(
    test_data,
    batch_size=len(test_data),
    collate_fn=lambda x: collate_fn(x, pad_token, device='cpu')
)

model = RNNModel(
    emb=embedder,
    n_hl=3,
    num_features=embedder.vector_size,
    n_classes=2,
    dropout=0.2,
    epochs=30,
    units=50,
    lr=0.1,
    momentum=0.3,
    device="cpu",
    loss_funct="cross-entropy",
    random_state=1,
    verbose=True,
    rnn_type="gru",
    bidirectional=True,
    freeze=True,
    lr_scheduler=True,
    factor=0.01,
    patience=2,
    pool_type='first',
)

model.fit(loader=train_loader, verbose=True, test=test_loader)
y_test, y_pred = model.predict_classes(test_loader)
y_test = y_test.to(torch.device("cpu"))
y_pred = y_pred.to(torch.device("cpu"))

print("Acc. score: ", accuracy_score(y_pred, y_test))
print("F1 score: ", f1_score(y_pred, y_test, average='macro'))

# torch.save(model.state_dict(), "model.pt")
#
# with open('train_data.pkl', 'wb+') as f:
#     pickle.dump(train_data, f)
#
# with open('test_data.pkl', 'wb+') as f:
#     pickle.dump(test_data, f)
