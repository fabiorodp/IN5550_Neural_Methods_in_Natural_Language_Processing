#!/bin/env python3
# coding: utf-8

# To run this on Saga, load this module:
# NLPL-nlptools/2021.01-gomkl-2019b-Python-3.7.4

import sys
import gensim
import logging
import zipfile
import json
import random
import pickle
from argparse import ArgumentParser
import os

# Simple toy script to get an idea of what one can do with word embedding models using Gensim
# Models can be found at http://vectors.nlpl.eu/explore/embeddings/models/
# or in the /cluster/shared/nlpl/data/vectors/latest/  directory on Saga
# (for example, /cluster/shared/nlpl/data/vectors/latest/200.zip)


def load_embedding(modelfile):
    # Detect the model format by its extension:
    # Binary word2vec format:
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True, unicode_errors="replace"
        )
    # Text word2vec format:
    elif (
        modelfile.endswith(".txt.gz")
        or modelfile.endswith(".txt")
        or modelfile.endswith(".vec.gz")
        or modelfile.endswith(".vec")
    ):
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False, unicode_errors="replace"
        )
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith(".zip"):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open("meta.json")
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print("============")
            # Loading the model itself:
            stream = archive.open(
                "model.bin"  # or model.txt, if you want to look at the model
            )
            emb_model = gensim.models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors="replace"
            )
    else:  # Native Gensim format?
        emb_model = gensim.models.KeyedVectors.load(modelfile)
        #  If you intend to train the model further:
        # emb_model = gensim.models.Word2Vec.load(args.embeddings)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


if __name__ == "__main__":
    parser = ArgumentParser()
    # add these as default arguments
    parser.add_argument("--model", default='')
    parser.add_argument(
        "--embeddings_id", 
        default="200.zip"
    )
    parser.add_argument(
        "--embeddings_dir",
        default = "/cluster/shared/nlpl/data/vectors/latest/"
    )
    args = parser.parse_args()

    # make sure model matches embedding id
    if "wiki" in args.model:
        args.embedding_id = "200.zip"
    elif "giga" in args.model:
        args.embedding_id = "29.zip"
    elif "200" in args.embeddings_id:
        args.model = "wiki.pkl"
    elif "29" in args.embeddings_id:
        args.model = "giga.pkl"
    else:
        logging.info("User input outside assignment requirements")

    # generate logger object
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)


    # load pickled model
    try:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Loaded model{args.model}")
    except:
        logger.info(f"Pickled model {args.model} not found. \n\
        Generating new...")
        model = None
    
    if model is None:
        embeddings = os.path.join(args.embeddings_dir, args.embeddings_id)
        logger.info(f"Loading embeddings from {embeddings}")
        model = load_embedding(embeddings)
        logger.info("Finished loading the embedding model...")


    # check model
    logger.info(f"Model vocabulary size: {len(model.vocab)}")
    logger.info(f"Random example of a word in the model: {random.choice(model.index2word)}")


    # always repickle the current model
    logger.info("Pickle model for faster loading...")
    if args.embeddings_id == '200.zip':
        model_name='wiki'
    elif args.embeddings_id == '29.zip':
        model_name='giga'
    else:
        model_name='other'
    with open(model_name+'.pkl', 'wb+') as f:
        pickle.dump(model, f)


    logger.info("Step 1. Provide text file/path")
    filename = input("File name: ")
    print(filename) # for storing in .output file
    with open(filename, 'r') as f:
        lines = f.read()
    logger.info(lines)
    logger.info("Step 2. Make sure file is lemmatized and POS tagged")

    words = lines.strip().split()
    logger.info(words)

    logger.info("Step 3. Output top 5 semantic associations for each word")
    for word in words:
        if word in model:
            print("=====")
            print(f"Word in model:\t{word}")
            print("Associate\tCosine")
            for i in model.most_similar(positive=[word], topn=5):
                print(f"{i[0]}\t{i[1]:.3f}")
            print("=====")


    logger.info("FINISHED")

    while False:
        query = input("Enter your word (type 'exit' to quit):")
        if query == "exit":
            exit()
        words = query.strip().split()
        # If there's only one query word, produce nearest associates
        if len(words) == 1:
            word = words[0]
            print(word)
            if word in model:
                print("=====")
                print("Associate\tCosine")
                for i in model.most_similar(positive=[word], topn=10):
                    print(f"{i[0]}\t{i[1]:.3f}")
                print("=====")
            else:
                print(f"{word} is not present in the model")

        # Else, find the word which doesn't belong here
        else:
            print("=====")
            print("This word looks strange among others:", model.doesnt_match(words))
            print("=====")
