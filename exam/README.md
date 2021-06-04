## Targeted Sentiment Analysis for Norwegian

This repository provides the data, baseline code, and extras necessary to begin work on the targeted sentiment track for IN5550. Cloning this repo is meant as a quick way of getting something working, but there are many ways of improving these results, ranging from small technical changes (including a hyperparameter search, more/less regularization, small architecture modifications) to larger and more theoretical changes (Comparing model architectures, adding character-level information, or using transfer learning models). Feel free to change anything necessary in the code.

## Usage

```
python test_{model_type}.py 
```
The possible `model_type`s include:
- `MTL_pipeline` : our best model
- `bert`: out-of-the_box Bert model
- `bert_bio`: predict BIO tags only
- `bert_polarity`: predict polarity only
- `bert_biopolarity`: predict BIO, thne polarity
- `hard`: hard sharing MTL setup
- `soft`: soft sharing MTL setup
- `noshare`: no share MTL setup



## Requirements

1. Python 3
2. sklearn  ```pip install -U scikit-learn```
3. Pytorch ```pip install torch torchvision torchtext```
4. tqdm ```pip install tqdm```
5. torchtext ```pip install torchtext```

