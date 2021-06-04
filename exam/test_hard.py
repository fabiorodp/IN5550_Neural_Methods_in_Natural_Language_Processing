# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.preprocessing import MTLDataset, MTLDataset, pad_both
    from exam.utils.models import HardShareTransformer
except:
    from utils.preprocessing import MTLDataset, MTLDataset, pad_both
    from utils.models import HardShareTransformer

from torch.utils.data import DataLoader
import torch

NORBERT = "/cluster/shared/nlpl/data/vectors/latest/216"
data_dir = "data/"
# NORBERT = "data/216"
# data_dir = "exam/data/"


print('loading data...')
train_dataset = MTLDataset(
    NORBERT_path=NORBERT,
    data_file=data_dir+'train.conll',
    specify_y='both',
    tokenizer=None
)

dev_dataset = MTLDataset(
    NORBERT_path=NORBERT,
    data_file=data_dir+'dev.conll',
    specify_y='both',
    tokenizer=train_dataset.tokenizer
)

test_dataset = MTLDataset(
    NORBERT_path=NORBERT,
    data_file=data_dir+'test.conll',
    specify_y='both',
    tokenizer=train_dataset.tokenizer
)
# print(train_dataset[0][0])
# print(train_dataset[0][1])
# print(train_dataset[0][2])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda batch: pad_both(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID,
                                 both=True)
)

dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    shuffle=False,    
    collate_fn=lambda batch: pad_both(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID,
                                 both=True)
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_both(batch=batch,
                                 IGNORE_ID=train_dataset.IGNORE_ID,
                                 both=True)
)

# X, y, attn = next(iter(train_loader))

# print('X')
# print(X.shape)
# print(X)
# print('y')
# print(y.shape)
# print(y)
# print('attn')
# print(attn.shape)
# print(attn)

model = HardShareTransformer(
    NORBERT=NORBERT,
    IGNORE_ID=train_dataset.IGNORE_ID,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    epochs=10, #9383
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr_BIO=1e-7,#9382
    optim_hyper2_BIO=0.9,
    lr_polarity=1e-7,
    optim_hyper2_polarity=0.9,
    epoch_patience=1,
    optimizer='SGD'
)

print('fitting model...')
model.fit(train_loader=train_loader, verbose=True, dev_loader=dev_loader)
binary_f1, propor_f1 = model.evaluate(test_loader)
