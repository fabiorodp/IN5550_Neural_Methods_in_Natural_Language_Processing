# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


import nltk
import gensim
import logging
import zipfile
import json
import random

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


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
        # emb_model = gensim.models.Word2Vec.load(embeddings_file)
    # Unit-normalizing the vectors (if they aren't already):
    emb_model.init_sims(
        replace=True
    )
    return emb_model


def load_embedded_model(url):
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    embeddings_file = url

    logger.info("Loading the embedding model...")
    model = load_embedding(embeddings_file)
    logger.info("Finished loading the embedding model...")

    logger.info(f"Model vocabulary size: {len(model.vocab)}")

    logger.info(f"Random example of a word in the model: "
                f"{random.choice(model.index2word)}")

    return model


def write_file_with_tokens_and_pos(sentence):
    tokens = [i[0] + "_" + i[1] for i in nltk.pos_tag(
        nltk.word_tokenize(sentence), tagset='universal')]

    with open('outputs/ex4/tokens.txt', 'w') as file:
        for token in tokens:
            file.write(token + "\n")


if __name__ == '__main__':
    s = "Almost all current dependency parsers classify based on " \
        "millions of sparse indicator features"
    write_file_with_tokens_and_pos(sentence=s)

    url_200 = input("Enter the url for embedded model 200 or ENTER for "
                    "(saga/200.zip):")
    model_200 = load_embedded_model(
        url="saga/200.zip" if url_200 == "" else url_200)

    url_29 = input("Enter the url for embedded model 29 or ENTER for "
                   "(saga/29.zip):")
    model_29 = load_embedded_model(
        url="saga/29.zip" if url_29 == "" else url_29)

    with open('outputs/ex4/tokens.txt', 'r') as file:
        input_tokens = [token[:-1] for token in file.readlines()]

    for model, name in zip([model_200, model_29], ["200", "29"]):
        with open(f'outputs/ex4/results_for_{name}.txt', 'w') as file:

            for query in input_tokens:
                words = query.strip().split()

                # If there's only one query word, produce nearest associates
                if len(words) == 1:
                    word = words[0]
                    print(word)
                    file.write("\n" + word + "\n")

                    if word in model:
                        print("=====\n" + "Associate\tCosine")
                        file.write("=====\n" + "Associate\tCosine\n")
                        for i in model.most_similar(positive=[word], topn=5):
                            print(f"{i[0]}\t{i[1]:.3f}")
                            file.write(f"{i[0]}\t{i[1]:.3f}\n")
                        print("=====")
                        file.write(f"=====\n")
                    else:
                        print(f"{word} is not present in the model")
                        file.write(f"{word} is not present in the model\n")

                # Else, find the word which doesn't belong here
                else:
                    print("=====")
                    print("This word looks strange among others:",
                          model.doesnt_match(words))
                    print("=====")
