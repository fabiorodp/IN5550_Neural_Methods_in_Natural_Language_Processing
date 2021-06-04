# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.preprocessing import OurDataset, pad_b
    from exam.utils.models import Transformer, TransformerMTL, pipeline
except:
    from utils.preprocessing import OurDataset, pad_b
    from utils.models import Transformer, TransformerMTL, pipeline

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

model_bio = torch.load("/cluster/projects/nn9851k/IN5550/fabior/"
                       "transformer_bio.pt")
model_polarity = torch.load("/cluster/projects/nn9851k/IN5550/fabior/"
                            "transformer_polarity_2.pt")
model_polarity_last = torch.load("/cluster/projects/nn9851k/IN5550/fabior/"
                                 "transformer_polarity_last.pt")
model_polarity_mean = torch.load("/cluster/projects/nn9851k/IN5550/fabior/"
                                 "transformer_polarity_mean.pt")

# model_bio = torch.load("exam/transformer_bio.pt")
# model_polarity = torch.load("exam/transformer_polarity_2.pt")
# model_polarity_last = torch.load("exam/transformer_polarity_last.pt")
# model_polarity_mean = torch.load("exam/transformer_polarity_mean.pt")


test_dataset = OurDataset(
    data_file=test_file,
    specify_y='POLARITY',
    NORBERT_path=None,
    tokenizer=model_bio.tokenizer
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_b(
        batch=batch,
        IGNORE_ID=model_bio.IGNORE_ID
    )
)

pipeline(
    test_loader=test_loader,
    model1=model_bio,
    model2=model_polarity
)

pipeline(
    test_loader=test_loader,
    model1=model_bio,
    model2=model_polarity_last
)

pipeline(
    test_loader=test_loader,
    model1=model_bio,
    model2=model_polarity_mean
)
