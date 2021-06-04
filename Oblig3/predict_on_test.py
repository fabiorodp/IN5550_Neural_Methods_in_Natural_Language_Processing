# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from Oblig3.packages.preprocess import load_raw_data, filter_raw_data, pad
    from Oblig3.packages.preprocess import OurCONLLUDataset
    from Oblig3.packages.model import Transformer, TransformerRNN
    from Oblig3.packages.metrics import evaluate


except:
    from packages.preprocess import load_raw_data, filter_raw_data, pad
    from packages.preprocess import OurCONLLUDataset
    from packages.model import Transformer, TransformerRNN
    from packages.metrics import evaluate

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from argparse import ArgumentParser
from conllu import parse_incr
import pandas as pd
import numpy as np
import torch
import conllu


# datapath = 'Oblig3/saga/test.conllu'
# NORBERT = 'Oblig3/saga/216'
# modelpath = 'Oblig3/saga/model_benchmark_12ep.pt'

datapath = '/cluster/projects/nn9851k/IN5550/fabior/test.conllu'
# datapath = '/cluster/projects/nn9851k/IN5550/fabior/test2000.conllu'
# modelpath = '/cluster/projects/nn9851k/IN5550/fabior/transformer_rnn_benchmark.pt'
# modelpath = '/cluster/projects/nn9851k/IN5550/fabior/trnn_test.pt'
# modelpath = '/cluster/projects/nn9851k/IN5550/pmhalvor/bert_rnn.pt'
modelpath = '/cluster/projects/nn9851k/IN5550/pmhalvor/model_benchmark_12ep.pt'
NORBERT = '/cluster/shared/nlpl/data/vectors/latest/216'


def load_raw_test_data(path):
    with open(path, "r", encoding="utf-8") as data_file:
        slow_parse = [{'TokenList': tokenlist,
                       'sentence': tokenlist.metadata['text'],
                       'labels': [t['misc']['name'] for t in tokenlist]}
                      for tokenlist in parse_incr(data_file)]
    return pd.DataFrame(slow_parse)


def pred_to_conllu(y_pred, mask, token_list, indexer):
    pred = []
    print('mask', len(mask))
    print('len token list', len(token_list))
    for si, sent in enumerate(token_list):
        print('len sent', len(sent))
        for ti, token in enumerate(sent):
            if mask[si][ti]==1:
                label = val2key(indexer, y_pred[si][ti].item())
                token['misc']['name'] = label if label != '[MASK]' else 'O'
        pred.append(sent)

    with open('outputs/predictions.conllu', 'w') as f:
        for item in pred:
            f.write(item.serialize())

    print("saving file in outputs/predictions.conllu...")
    print("outputs/predictions.conllu saved.")


def val2key(d, v):
    return list(d.keys())[
        list(d.values()).index(v)
    ]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        required=False,
        help="Path to a test dataset.",
        default=datapath,
    )
    
    parser.add_argument(
        "--model",
        "-m",
        required=False,
        help="Path to the saved model.",
        default=modelpath,
    )
    
    parser.add_argument(
        "--bert",
        "-b",
        required=False,
        help="Path to the NORBERT file.",
        default=NORBERT,
    )
    
    parser.add_argument(
        "--device",
        "-d",
        required=False,
        help="Device to be used in the model.",
        default='cpu',
    )
    args = parser.parse_args()

    print('loading data...')
    df = load_raw_test_data(path=args.test)
    tokenizer = BertTokenizer.from_pretrained(args.bert)
    model = torch.load(f=args.model)

    # creating data sets
    dataset = OurCONLLUDataset(
        df=df,
        tokenizer=tokenizer,
        device=args.device,
        label_indexer=model.label_indexer
    )

    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        collate_fn=lambda batch: pad(batch, dataset.IGNORE_ID)
    )

    print('predicting...')
    input_ids, y_true, y_pred, mask = model.predict_classes(
        loader=loader, get_batch=True)

    print("saving file in outputs/predictions.conllu...")
    pred = []
    for i in range(input_ids.size(0)):
        count = 0
        for idx, id in enumerate(input_ids[i]):

            if dataset._tokenizer.convert_ids_to_tokens(id.item()) in \
                    dataset._tokenizer.all_special_tokens:
                continue

            elif dataset._tokenizer.convert_ids_to_tokens(id.item())\
                    .startswith('##'):
                continue

            label = val2key(dataset.label_indexer, y_pred[i][idx].item())
            dataset.TokenList[i][count]['misc']['name'] = label
            count += 1

        pred.append(dataset.TokenList[i])

    with open('outputs/predictions.conllu', 'w') as f:
        for item in pred:
            f.write(item.serialize())
    print("outputs/predictions.conllu saved.")
