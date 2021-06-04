# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.preprocessing import OurDataset, pad_b
    from exam.utils.models import Transformer, TransformerMTL
except:
    from utils.preprocessing import OurDataset, pad_b
    from utils.models import Transformer, TransformerMTL

from torch.utils.data import DataLoader
import torch


NORBERT = "/cluster/shared/nlpl/data/vectors/latest/216"
# NORBERT = 'exam/saga/216'

train_file = '/cluster/projects/nn9851k/IN5550/fabior/train.conll'
dev_file = '/cluster/projects/nn9851k/IN5550/fabior/dev.conll'
test_file = '/cluster/projects/nn9851k/IN5550/fabior/test.conll'

# train_file = 'exam/data/train.conll'
# dev_file = 'exam/data/dev.conll'
# test_file = 'exam/data/test.conll'


train_dataset = OurDataset(
    data_file=train_file,
    specify_y='BIO',
    NORBERT_path=NORBERT,
    tokenizer=None
)

# x_ds, y_ds, att_ds = next(iter(train_dataset))
# sentence_tk_ds = train_dataset.tokenizer.convert_ids_to_tokens(x_ds)
# sentence_ds = train_dataset.tokenizer.decode(x_ds)

dev_dataset = OurDataset(
    data_file=dev_file,
    specify_y='BIO',
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

test_dataset = OurDataset(
    data_file=test_file,
    specify_y='BIO',
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x, y, att = next(iter(train_loader))

dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x1, y1, att1 = next(iter(dev_loader))

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x2, y2, att2 = next(iter(test_loader))

model_bio = torch.load("/cluster/projects/nn9851k/IN5550/fabior/"
                       "transformer_bio.pt")

model_polarity = TransformerMTL(
    NORBERT=NORBERT,
    tokenizer=train_dataset.tokenizer,
    num_labels=3,
    IGNORE_ID=train_dataset.IGNORE_ID,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,  # best is 2
    lr_scheduler=False,
    factor=0.1,
    lrs_patience=2,
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr=0.00001,
    momentum=0.9,
    epoch_patience=1,
    label_indexer=None,
    optmizer='AdamW',
    previous_model=model_bio,
    hs_type='mean'
)

model_polarity.fit(
    train_loader=train_loader,
    verbose=True,
    dev_loader=dev_loader
)

binary_f1, propor_f1 = model_polarity.evaluate(test_loader)
# torch.save(model_polarity, "exam/transformer_polarity_last.pt")
