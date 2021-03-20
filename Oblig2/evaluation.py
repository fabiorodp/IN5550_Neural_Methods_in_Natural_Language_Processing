#!/bin/env python3
# coding: utf-8

# To run this on Saga, load this module:
# NLPL-SciPy-bundle/2021.01-gomkl-2019b-Python-3.7.4

from argparse import ArgumentParser
import pandas as pd
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="path to a file with system predictions",
        default="stanford_sentiment_binary_test_predictions.tsv.gz",
    )
    parser.add_argument(
        "--gold",
        "-g",
        help="path to  file with gold scores",
        required=True,
        default="stanford_sentiment_binary_test_gold.tsv.gz",
    )
    args = parser.parse_args()

    pred_df = pd.read_csv(args.predictions, sep="\t", header=0)
    gold_df = pd.read_csv(args.gold, sep="\t", header=0)

    pred_labels = pred_df.label.values
    gold_labels = gold_df.label.values

    print(f"Test accuracy: {accuracy_score(pred_labels, gold_labels)}")
