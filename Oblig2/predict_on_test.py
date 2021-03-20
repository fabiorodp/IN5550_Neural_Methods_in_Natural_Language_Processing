# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from packages.preprocessing import collate_fn
from packages.preprocessing import RNNDataset
from packages.ann_models import RNNModel
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pandas as pd
import pickle
import torch
import nltk


def load_model(model_path, embedder, model_params):
    print("Loading RNN model...")
    model = RNNModel(
        emb=embedder,
        num_features=embedder.vector_size,
        **model_params
        )
    model.load_state_dict(torch.load(model_path), strict=True)
    print('Model loaded')
    return model


def load_data(train_data_path, test_data_path):
    print("Loading train data...")
    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)
    print('Train data loaded')

    print("Loading test data...")
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)
    print('Test data loaded')

    return train_data, test_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--test",
        "-t",
        required=True,
        help="Path to a test dataset.",
        default="/cluster/projects/nn9851k/IN5550/pmhalvor/"
                "stanford_sentiment_binary.tsv.gz",
    )

    args = parser.parse_args()
    # print(args.test)

    with open("/cluster/projects/nn9851k/IN5550/pmhalvor/"
              "embedder.pkl", "rb") as f:
        embedder = pickle.load(f)

    pad_token = embedder.vocab['<pad>'].index

    with open("/cluster/projects/nn9851k/IN5550/pmhalvor/"
              "model_params.pkl", "rb") as f:
        model_params = pickle.load(f)

    model = load_model(
        model_path="/cluster/projects/nn9851k/IN5550/pmhalvor/model.pt",
        embedder=embedder,
        model_params=model_params
    )

    df = pd.read_csv(args.test, sep='\t')
    df.tokens = df.tokens.apply(lambda x: nltk.word_tokenize(x))

    test_data = RNNDataset(
        embedder=embedder,
        df=df,
        device="cpu",
        random_state=1,
        label_vocab=None,
        verbose=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=len(df),
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_token, device='cpu')
    )

    y_test, y_pred = model.predict_classes(test_loader)

    score = accuracy_score(y_test, y_pred)
    print(f"The score between y_true and y_predict is {score}")

    print("saving file "
          "in data/stanford_sentiment_binary_test_predictions.tsv.gz to "
          "be used in evaluation.py...")
    y_pred = y_pred.numpy()
    y_pred = list(y_pred)
    y_pred = ["negative" if x == 0 else "positive" for x in y_pred]
    df.label = y_pred
    df.to_csv("data/stanford_sentiment_binary_test_predictions.tsv.gz")
    print("data/stanford_sentiment_binary_test_predictions.tsv.gz"
          "saved.")
