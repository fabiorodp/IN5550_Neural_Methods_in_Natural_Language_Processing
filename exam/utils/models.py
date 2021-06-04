# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


try:
    from exam.utils.metrics import binary_analysis, proportional_analysis
    from exam.utils.metrics import get_analysis
except:
    from utils.metrics import binary_analysis, proportional_analysis
    from utils.metrics import get_analysis

from itertools import chain
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from transformers import BertForTokenClassification, BertTokenizer, BertModel
from tqdm import tqdm


import numpy as np
import torch.nn as nn
import torch


def pipeline(test_loader, model1, model2):
    preds1, golds1, sents1, gf1 = model1.predict(test_loader)
    preds2, golds2, sents2, gf2 = model2.predict(test_loader)

    indexer = {
        "O": 0,
        "B-targ-Positive": 1,
        "I-targ-Positive": 2,
        "B-targ-Negative": 3,
        "I-targ-Negative": 4
    }

    BIO_indexer = {
        0: "O",
        1: "I-targ-",
        2: "B-targ-",
    }

    polarity_indexer = {
        0: "O",
        1: "Positive",
        2: "Negative",
    }

    idxs_bio, idxs_polarity = [], []
    for idx_b, e_bio in enumerate(sents1):
        for idx_p, e_polarity in enumerate(sents2):
            if e_bio == e_polarity:
                idxs_bio.append(idx_b)
                idxs_polarity.append(idx_p)
                break

    collapsed_sents, collapsed_preds = [], []
    for i1, i2 in zip(idxs_bio, idxs_polarity):
        temp_sents, temp_preds, temp_golds = [], [], []

        for e in sents2[i2]:
            temp_sents.append(e)

        collapsed_sents.append(temp_sents)

        for e1, e2 in zip(preds1[i1], preds2[i2]):
            if BIO_indexer[e1] == polarity_indexer[e2]:
                temp_preds.append(BIO_indexer[e1])
            elif BIO_indexer[e1] == 'O':
                temp_preds.append(BIO_indexer[e1])
            elif polarity_indexer[e2] == 'O':
                temp_preds.append(polarity_indexer[e2])
            else:
                temp_preds.append(BIO_indexer[e1] + polarity_indexer[e2])

        collapsed_preds.append(temp_preds)

    preds = []
    for e_ in collapsed_preds:
        temp = []
        for e in e_:
            temp.append(indexer[e])
        preds.append(temp)

    flat_preds = [int(i) for l in preds for i in l]
    flat_golds = [int(i) for l in gf1 for i in l]

    analysis = get_analysis(
        sents=collapsed_sents,
        y_pred=preds,
        y_test=gf1
    )

    binary_f1 = binary_analysis(analysis)
    propor_f1 = proportional_analysis(flat_golds, flat_preds)
    return binary_f1, propor_f1


class BiLSTM(nn.Module):

    def __init__(self, word2idx,
                 embedding_matrix,  # pretrained emb model 
                 embedding_dim,     # emb model dim
                 hidden_dim,        # tunable node count per hidden layer
                 device,            # {torch.device('cpu'), torch.device('cuda')}
                 output_dim=5,      # fixed bc possible labels: {BP, BN, IP, IN, O}
                 num_layers=2,      # tunable size of our BiLSTM
                 lstm_dropout=0.2,  # tunable model dropout
                 word_dropout=0.5,  # tunable embedding dropout
                 learning_rate=0.01,# 
                 train_embeddings=False
                 ):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.vocab_size = len(word2idx)
        self.lstm_dropout = lstm_dropout
        self.word_dropout = word_dropout
        self.learning_rate = learning_rate
        self.sentiment_criterion = nn.CrossEntropyLoss()

        # set up pretrained embeddings
        weight = torch.FloatTensor(embedding_matrix)
        self.word_embeds = nn.Embedding.from_pretrained(weight,
                                                        freeze=False)
        # for optimizer simplicity, just turn off gradients for vocab params
        self.word_embeds.requires_grad = train_embeddings  

        self.word_dropout = nn.Dropout(word_dropout)

        # set up BiLSTM and linear layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True
        ).to(self.device)

        self.linear = nn.Linear(hidden_dim * 2, self.output_dim) # "flattens" hidden state of lstm

        # We include the optimizer here to enable sklearn style fit() method
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

    def init_hidden(self, batch_size=1):
        """
        :param batch_size: batch size for the training/dev/test batch
        """
        h0 = torch.zeros(
            (self.lstm.num_layers * (1 + self.lstm.bidirectional),
             batch_size, self.lstm.hidden_size),
            device=self.device
        )

        c0 = torch.zeros_like(h0, device=self.device)
        return h0, c0

    def forward(self, x):
        """
        :param x: a packed padded sequence
        """
        # get the batch sizes, which will be used for packing embeddings
        batch_size = x.batch_sizes[0]

        # move data to device (CPU, GPU)
        data = x.data.to(self.device)

        # Embed and add dropout
        emb = self.word_embeds(data)
        emb = self.word_dropout(emb)

        # Pack and pass to LSTM layer
        packed_emb = PackedSequence(emb, x.batch_sizes)
        self.hidden = self.init_hidden(batch_size)
        output, (hn, cn) = self.lstm(packed_emb, self.hidden)

        # Unpack and send to linear layer
        o, _ = pad_packed_sequence(output, batch_first=True)
        o = self.linear(o)
        return o

    def fit(self, train_loader, dev_loader, epochs=10):
        """
        Trains a model in an Sklearn style manner.
        :param dev_loader: torch.utils.data.DataLoader object of the train data
        :param dev_loader: torch.utils.data.DataLoader object of the dev data
                           with batch_size=1
        :param epochs: number of epochs to train the model
        """

        for epoch in range(epochs):
            # Iterate over training data
            self.train()
            epoch_loss = 0
            num_batches = 0

            # Data reader requires each example to have (raw_text, x, y, idx)
            for raw_text, x, y, idx in tqdm(train_loader):
                self.zero_grad()

                # Get the batch_sizes, batches_len, and seq_length for future
                # changes to data view
                original_data, batch_sizes = pad_packed_sequence(
                    x,
                    batch_first=True
                )
                batches_len, seq_length = original_data.size()

                # Get the predictions and the gold labels (y)
                preds = self.forward(x)
                y, _ = pad_packed_sequence(y, batch_first=True)

                # Reshape predictions
                # (batch_size * max_seq_length, num_labels)
                preds = preds.reshape(batches_len * seq_length, 5)
                y = y.reshape(batches_len * seq_length).to(self.device)

                # Get loss and update epoch_loss
                loss = self.sentiment_criterion(preds, y)
                epoch_loss += loss.data
                num_batches += 1

                # Update parameters
                loss.backward()
                self.optimizer.step()

            print()
            print("Epoch {0} loss: {1:.3f}".format(
                epoch + 1,
                epoch_loss / num_batches)
            )

            print("Dev")
            self.evaluate(dev_loader)

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions = []
        golds = []
        sents = []
        for raw_text, x, y, idx in tqdm(test_loader):
            self.preds1 = self.forward(x).argmax(2)
            predictions.append(self.preds1[0])
            golds.append(y.data)
            sents.append(raw_text[0])
        return predictions, golds, sents

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the examples passed via test_loader.
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.preds, self.golds, self.sents = self.predict(test_loader)
        flat_preds = [int(i) for l in self.preds for i in l]
        flat_golds = [int(i) for l in self.golds for i in l]

        analysis = get_analysis(
            sents=self.sents,
            y_pred=self.preds,
            y_test=self.golds
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    def print_predictions(self, test_loader, outfile, idx2label):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        :param outfile: the file name to print the predictions to
        :param idx2label: a python dictionary which maps label indices to
                          the actual label
        """
        preds, golds, sents = self.predict(test_loader)
        with open(outfile, "w") as out:
            for sent, gold, pred in zip(sents, golds, preds):
                for token, gl, pl in zip(sent, gold, pred):
                    glabel = idx2label[int(gl)]
                    plabel = idx2label[int(pl)]
                    out.write(("{0}\t{1}\t{2}\n".format(token,
                                                        glabel,
                                                        plabel)))
                out.write("\n")


class HardShareTransformer(nn.Module):
    """
    Mulitask architecture with hard parameter sharing (bottom layers fully shared)
    """

    def __init__(self, NORBERT, IGNORE_ID, device='cpu',
                 epochs=10, loss_funct='cross-entropy', random_state=None,
                 verbose=False, lr_BIO=2e-5, optim_hyper2_BIO=0.9, lr_polarity=2e-5, 
                 optim_hyper2_polarity=0.9, epoch_patience=1, optimizer='SGD'):

        super().__init__()

        torch.autograd.set_detect_anomaly(True)

        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)


        # global parameters
        self.NORBERT = NORBERT
        self.IGNORE_ID = IGNORE_ID
        self.device = device
        self.epochs = epochs
        self.epoch_patience = epoch_patience
        self.last_epoch = None # for integrated early stopping


        # creating embedder and models from NORBERT
        self.tokenizer = BertTokenizer.from_pretrained(self.NORBERT) # for evaluation decoding
        self.bert = BertModel.from_pretrained(
            self.NORBERT,
            num_labels=3,   # NOTE: fixed BIO label count (0:O, 1:I, 2:B)
        ).to(torch.device(self.device))
        self.linear_BIO = nn.Linear(768, 3).to(torch.device(self.device))
        self.linear_polarity = nn.Linear(768, 3).to(torch.device(self.device))


        # loss criterions for each task
        self.criterion_BIO = self._init_loss_function(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )
        self.criterion_polarity = self._init_loss_function(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )


        # setting each task's optimizer
        self.optimizer = optimizer # for simplcity both get same 

        ## BIO 
        self.lr_BIO = lr_BIO
        self.optim_hyper2_BIO = optim_hyper2_BIO
        self.optimizer_BIO = self._init_optimizer(self.optimizer,
            chain(self.bert.parameters(), self.linear_BIO.parameters()), 
            self.lr_BIO, hyper2=self.optim_hyper2_BIO
        )
        
        ## polarity
        self.lr_polarity = lr_polarity
        self.optim_hyper2_polarity = optim_hyper2_polarity
        self.optimizer_polarity = self._init_optimizer(self.optimizer,
            chain(self.bert.parameters(), self.linear_polarity.parameters()), 
            self.lr_polarity, hyper2=self.optim_hyper2_polarity
        )


        # storing scores
        self.losses = []
        self.binary_f1 = []
        self.propor_f1 = []


        # early stop
        self.early_stop_epoch = None


        # label indexers
        self.indexer_BIO = {"O":0, "I":1, "B":2}
        self.indexer_polarity = {"O":0, "Positive":1, "Negative":2}


        # storing outputs
        self.outputs = None

    def fit(self, train_loader=None, verbose=False, dev_loader=None):
        """
        Fits the model to the training data using the models
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        loader: torch.nn.Dataloader=None
            Dataloader object to load the batches, defaults to None
        verbose: bool=False
            If True: prints out progressive output, defaults to False
        ______________________________________________________________
        Returns:
        None
        """
        iterator = tqdm(range(self.epochs)) if self.verbose else range(self.epochs)

        for epoch in iterator:
            _loss = []

            for b, batch in enumerate(train_loader):
                output_BIO, output_polarity= self.forward(batch=batch)
                loss = self.backward(
                    output_BIO=output_BIO,
                    output_polarity=output_polarity,
                    batch=batch
                )
                _loss.append(loss.item())

                if self.verbose:
                    print(f"Batch: {b}  |"
                          f"  Train Loss: {loss}  |")

            # if self._early_stop(epoch_idx=epoch,
            #                     patience=self.epoch_patience):
            #     print('Early stopped!')

            #     self.losses.append(np.mean(_loss))

            #     if self.verbose:
            #         print(f"Epoch: {epoch}  |"
            #               f"  Train Loss: {self.losses[epoch]}")

            #     if dev_loader is not None:
            #         binary_f1, propor_f1 = \
            #             self.evaluate(test_loader=dev_loader)
            #         self.binary_f1.append(binary_f1)
            #         self.propor_f1.append(propor_f1)
            #     break

            else:
                self.losses.append(np.mean(_loss))

                if self.verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")


                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)

        self.last_epoch = epoch
        return self

    def forward(self, batch):
        """
        Performs a feed forward step.
            ______________________________________________________________
            Parameters:
            batch: tuple containing (input_ids, targets, att_maks)

            ______________________________________________________________
            Returns:
            outputs: torch.Tensor
        """
        
        input_ids=batch[0].to(torch.device(self.device))
        attention_mask=batch[2].to(torch.device(self.device))
        output_hidden_states=True

        output = self.bert(input_ids, attention_mask, 
            output_hidden_states=output_hidden_states
        ).last_hidden_state
        # print('output', output.shape)

        # output = output[:,batch[3]].diagonal().permute(2,0,1)
        # print('output', output.shape)

        # print('y_mask', batch[3].shape, batch[3])
        return self.linear_BIO(output), self.linear_polarity(output)

    def backward(self, output_BIO, output_polarity, batch):
        """
        Performs a backpropogation step computing the loss.
            ______________________________________________________________
            Parameters:
            output:
                The output after forward with shape (batch_size, num_classes).
            target:
                The real targets.
            ______________________________________________________________
            Returns:
            loss: float
                How close the estimate was to the gold standard.
        """
        # batch = [x, y, z] = [[32, 49], [32, 98], [32, 49]]
        targets_BIO = batch[1][:,:batch[0].size(1)]# [(BIO, polarity),...]
        targets_polarity = batch[1][:,batch[0].size(1):]# [(BIO, polarity),...]
        # output_BIO = output_BIO.permute(0, 2, 1)
        # output_polarity = output_polarity.permute(0, 2, 1)

        # print('before transform')
        # print('output_BIO',output_BIO.shape)
        # print('target_BIO', targets_BIO.shape) # this should now be double as long...
        # print('output_polarity',output_polarity.shape)
        # print('target_polarity', targets_polarity.shape)
        output_BIO = output_BIO.reshape(output_BIO.shape[0]*output_BIO.shape[1], 3)
        output_polarity = output_polarity.reshape(
            output_polarity.shape[0]*output_polarity.shape[1], 3)

        targets_BIO = targets_BIO.reshape(targets_BIO.shape[0]*targets_BIO.shape[1])
        targets_polarity = targets_polarity.reshape(
            targets_polarity.shape[0]*targets_polarity.shape[1])

        # print('after transform')
        # print('output_BIO',output_BIO.shape)
        # print('target_BIO', targets_BIO.shape)
        # print('output_polarity',output_polarity.shape)
        # print('target_polarity', targets_polarity.shape)
        
        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self.optimizer_BIO.zero_grad()
        self.optimizer_polarity.zero_grad()

        # find loss
        loss_BIO = self.criterion_BIO(
            input=output_BIO,
            target=targets_BIO.to(torch.device(self.device))
        ) 
        loss_polarity = self.criterion_polarity(
            input=output_polarity,
            target=targets_polarity.to(torch.device(self.device))
        )

        total_loss = loss_BIO + loss_polarity


        # calculating gradients
        total_loss.backward()

        # updating weights from the model by calling optimizer.step()
        self.optimizer_BIO.step()
        self.optimizer_polarity.step()

        return total_loss

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        self.predictions, self.golds, self.sents = [], [], []

        for batch in tqdm(test_loader):
            output_BIO, output_polarity = self.forward(batch)
            y_pred_BIO = output_BIO.argmax(2).squeeze(0)
            y_pred_polarity = output_polarity.argmax(2).squeeze(0)
            y_pred = self.join_outputs(y_pred_BIO, y_pred_polarity)
            self.predictions.append(y_pred.tolist())
            self.golds.append(batch[1].squeeze(0).tolist())

            for i in batch[0]:
                self.decoded_sentence = \
                    self.tokenizer.convert_ids_to_tokens(i)
                self.sents.append(self.decoded_sentence)

        # #################### truncating predictions, golds and sents
        self.predictions__, self.golds__, self.sents__ = [], [], []
        for l_p, l_g, l_s in zip(self.predictions, self.golds, self.sents):
            predictions_, golds_, sents_ = [], [], []

            for e_p, e_g, e_s in zip(l_p, l_g, l_s):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)
                    sents_.append(e_s)

            self.predictions__.append(predictions_)
            self.golds__.append(golds_)
            self.sents__.append(sents_)
        # ####################

        return self.predictions__, self.golds__, self.sents__

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via test_loader.

        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        preds, golds, sents = self.predict(test_loader)
        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [int(i) for l in golds for i in l]

        analysis = get_analysis(
            sents=sents,
            y_pred=preds,
            y_test=golds
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    @staticmethod
    def join_outputs(y_pred_BIO, y_pred_polarity):
        y_pred = torch.zeros_like(y_pred_BIO)
        for i, (b, p) in enumerate(zip(y_pred_BIO, y_pred_polarity)):
            # b: 0:O, 1:I, 2:B
            # p: 0:O, 1:Positive, 2:Negative
            # y: 0:O, 1:BP, 2:IP, 3:BN, 4:IN
            # if b.item()==0 and p.item()==0:#O
            #     y_pred[i] = 0
            if b.item()==2 and p.item()==1:#BP
                y_pred[i] = 1
            if b.item()==1 and p.item()==1:#IP
                y_pred[i] = 2
            if b.item()==2 and p.item()==2:#BN
                y_pred[i] = 3
            if b.item()==1 and p.item()==2:#IN
                y_pred[i] = 4
        return y_pred
            

    @staticmethod
    def _init_loss_function(lf_type, IGNORE_ID):
        """
        Returns the specified loss function from torch.nn
        ______________________________________________________________
        Parameters:
        lf_type: str
            The loss function to return
        ______________________________________________________________
        Returns:
        lf: torch.nn.function
            The specified loss function
        """
        if lf_type == "cross-entropy":  # I:(N,C) O:(N)
            return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)

        if lf_type == 'binary-cross-entropy':
            return torch.nn.BCELoss()

    @staticmethod
    def _init_optimizer(optimizer_str, parameters, lr, hyper2=None):
        if optimizer_str.lower()=='sgd':
            return torch.optim.SGD(
                parameters,
                lr,
                momentum=hyper2
            )
        elif optimizer_str.lower()=='adam':
            return torch.optim.Adam(
                parameters,
                lr,
                weight_decay=hyper2
            )
        else:
            return torch.optim.AdamW(
                parameters,
                lr,
                weight_decay=hyper2
            )




class SoftShareTransformer(nn.Module):
    """
    Mulitask architecture with soft parameter sharing via l_2 regularization
    """

    def __init__(self, NORBERT, IGNORE_ID, device='cpu',
                 epochs=10, loss_funct='cross-entropy', random_state=None,
                 verbose=False, lr_BIO=2e-5, momentum_BIO=0.9, lr_polarity=2e-5, 
                 momentum_polarity=0.9, epoch_patience=1, optimizer='SGD'):

        super().__init__()


        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)


        # global parameters
        self.NORBERT = NORBERT
        self.IGNORE_ID = IGNORE_ID
        self.device = device
        self.epochs = epochs
        self.epoch_patience = epoch_patience
        self.last_epoch = None # for integrated early stopping


        # creating embedder and models from NORBERT
        self.tokenizer = BertTokenizer.from_pretrained(self.NORBERT) # for evaluation decoding
        self.model_BIO = BertForTokenClassification.from_pretrained(
            self.NORBERT,
            num_labels=3,   # NOTE: fixed BIO label count (0:O, 1:I, 2:B)
        ).to(torch.device(self.device))
        self.model_polarity = BertForTokenClassification.from_pretrained(
            self.NORBERT,
            num_labels=3,   # NOTE: fixed polarity count (0:None, 1:Positive, 2:Negative)
        ).to(torch.device(self.device))


        # loss criterions for each task
        self.criterion_BIO = self._init_loss_function(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )
        self.criterion_polarity = self._init_loss_function(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )


        # setting each task's optimizer
        self.optimizer = optimizer # for simplcity both get same 

        ## BIO 
        self.lr_BIO = lr_BIO
        self.momentum_BIO = momentum_BIO
        self.optimizer_BIO = self._init_optimizer(self.optimizer,
            self.model_BIO.parameters(), self.lr_BIO, hyper2=self.momentum_BIO
        )
        
        ## polarity
        self.lr_polarity = lr_polarity
        self.momentum_polarity = momentum_polarity
        self.optimizer_polarity = self._init_optimizer(self.optimizer,
            self.model_polarity.parameters(), 
            self.lr_polarity, 
            hyper2=self.momentum_polarity
        )


        # storing scores
        self.losses = []
        self.binary_f1 = []
        self.propor_f1 = []


        # early stop
        self.early_stop_epoch = None


        # label indexers
        self.indexer_BIO = {"O":0, "I":1, "B":2}
        self.indexer_polarity = {"O":0, "Positive":1, "Negative":2}


        # storing outputs
        self.outputs = None


    def fit(self, train_loader=None, verbose=False, dev_loader=None):
        """
        Fits the model to the training data using the models
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        laoder: torch.nn.Dataloader=None
            Dataloader object to load the batches, defaults to None
        verbose: bool=False
            If True: prints out progressive output, defaults to False
        ______________________________________________________________
        Returns:
        None
        """
        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)

        for epoch in iterator:
            _loss = []

            for b, batch in enumerate(train_loader):
                self.train()
                output_BIO, output_polarity= self.forward(batch=batch)
                loss = self.backward(
                    output_BIO=output_BIO,
                    output_polarity=output_polarity,
                    batch=batch
                )
                _loss.append(loss.item())

                if self.verbose:
                    print(f"Batch: {b}  |"
                          f"  Train Loss: {loss}  |")

            # if self._early_stop(epoch_idx=epoch,
            #                     patience=self.epoch_patience):
            #     print('Early stopped!')

            #     self.losses.append(np.mean(_loss))

            #     if verbose:
            #         print(f"Epoch: {epoch}  |"
            #               f"  Train Loss: {self.losses[epoch]}")

            #     if dev_loader is not None:
            #         binary_f1, propor_f1 = \
            #             self.evaluate(test_loader=dev_loader)
            #         self.binary_f1.append(binary_f1)
            #         self.propor_f1.append(propor_f1)
            #     break

            else:
                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")


                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)

        self.last_epoch = epoch
        return self



    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor

        """
        
        input_ids=batch[0].to(torch.device(self.device))
        attention_mask=batch[2].to(torch.device(self.device))

        output_BIO = self.model_BIO(input_ids, attention_mask).logits
        output_polarity = self.model_polarity(input_ids, attention_mask).logits

        return output_BIO, output_polarity

    def backward(self, output_BIO, output_polarity, batch):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The real targets.
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        targets = batch[1]
        targets_BIO = targets[:,:batch[0].size(1)]      # slice first half of targets
        targets_polarity = targets[:,:batch[0].size(1)] # slice last half of targets

        # print('output_BIO:      ', output_BIO.shape)
        # print('output_polarity: ', output_polarity.shape)
        # print('output_BIO_perm: ', output_BIO.permute(0, 2, 1).shape)
        # print('output_polarity_perm:', output_polarity.permute(0, 2, 1).shape)
        # print('target_BIO:      ', targets_BIO.shape)
        # print('target_polarity: ', targets_polarity.shape)

        # find loss
        loss_BIO = self.criterion_BIO(
            input=output_BIO.permute(0, 2, 1),
            target=targets_BIO.to(torch.device(self.device))
        )
        loss_polarity = self.criterion_polarity(
            input=output_polarity.permute(0, 2, 1),
            target=targets_polarity.to(torch.device(self.device))
        )
        loss_soft = self.soft_loss()

        # magic sum loss
        loss_total = loss_BIO + loss_polarity + loss_soft
        

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self.optimizer_BIO.zero_grad()
        self.optimizer_polarity.zero_grad()

        # calculating gradients
        loss_total.backward()

        # NOTE to self
        # apply L2 regularization
        ### ok so this is apparently done using weight_decay when defining the optimizer(s)
        ### jeremy mentioned there would need 2 optimizers
        ### one for each model?
        ### i feel like the problems with hard were solved magically
        ### hopefully the same for soft?

        # updating weights from the model by calling optimizer.step()
        self.optimizer_BIO.step()

        # updating weights from the model by calling optimizer.step()
        self.optimizer_polarity.step()

        return loss_total


    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        self.predictions, self.golds, self.sents = [], [], []

        for batch in tqdm(test_loader):
            output_BIO, output_polarity = self.forward(batch)
            y_pred_BIO = output_BIO.argmax(2).squeeze(0)
            y_pred_polarity = output_polarity.argmax(2).squeeze(0)
            y_pred = self.join_outputs(y_pred_BIO, y_pred_polarity)
            self.predictions.append(y_pred.tolist())
            self.golds.append(batch[1].squeeze(0).tolist())

            for i in batch[0]:
                self.decoded_sentence = \
                    self.tokenizer.convert_ids_to_tokens(i)
                self.sents.append(self.decoded_sentence)

        # #################### truncating predictions, golds and sents
        self.predictions__, self.golds__, self.sents__ = [], [], []
        for l_p, l_g, l_s in zip(self.predictions, self.golds, self.sents):
            predictions_, golds_, sents_ = [], [], []

            for e_p, e_g, e_s in zip(l_p, l_g, l_s):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)
                    sents_.append(e_s)

            self.predictions__.append(predictions_)
            self.golds__.append(golds_)
            self.sents__.append(sents_)
        # ####################

        return self.predictions__, self.golds__, self.sents__

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via test_loader.

        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        preds, golds, sents = self.predict(test_loader)
        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [int(i) for l in golds for i in l]

        analysis = get_analysis(
            sents=sents,
            y_pred=preds,
            y_test=golds
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    @staticmethod
    def join_outputs(y_pred_BIO, y_pred_polarity):
        y_pred = torch.empty_like(y_pred_BIO)
        for i, (b, p) in enumerate(zip(y_pred_BIO, y_pred_polarity)):
            # b: 0:O, 1:I, 2:B
            # p: 0:O, 1:Positive, 2:Negative
            # y: 0:O, 1:BP, 2:IP, 3:BN, 4:IN
            if b.item()==0 and p.item()==0:#O
                y_pred[i] = 0
            if b.item()==2 and p.item()==1:#BP
                y_pred[i] = 1
            if b.item()==1 and p.item()==1:#IP
                y_pred[i] = 2
            if b.item()==2 and p.item()==2:#BN
                y_pred[i] = 3
            if b.item()==1 and p.item()==2:#IN
                y_pred[i] = 4
        return y_pred
            


    def soft_loss(self):
        soft_params_BIO = self.get_soft_parameters(self.model_BIO)
        soft_params_polarity = self.get_soft_parameters(self.model_polarity)

        soft_sharing_loss = torch.tensor(0.)
        for params in zip(soft_params_BIO, soft_params_polarity): 
            soft_sharing_loss += torch.norm(params[0] - params[1], p='fro') # what's fro?

        return soft_sharing_loss # dont understand what this loss is calculating

        # looks like just returned norm of differences between each param dim
        # correct:
        #       This is the whole idea behind soft loss
        #       the two individual models are only loosly coupled
        #       so the parameters can be tuned as they see fit,
        #       with a regularization mech. to discourage overfitting.
        # The normalization here is the regularization component.

    @staticmethod
    def get_soft_parameters(model):
        """
        :return: returns a list of lists of related params from all tasks specific networks
        """
        no_share = ['classifier']
        soft_params = [
            p for n, p in model.named_parameters() 
            if not any(nd in n for nd in no_share)
        ]
        return soft_params

    @staticmethod
    def _init_loss_function(lf_type, IGNORE_ID):
        """
        Returns the specified loss function from torch.nn
        ______________________________________________________________
        Parameters:
        lf_type: str
            The loss function to return
        ______________________________________________________________
        Returns:
        lf: torch.nn.function
            The specified loss function
        """
        if lf_type == "cross-entropy":  # I:(N,C) O:(N)
            return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)

        if lf_type == 'binary-cross-entropy':
            return torch.nn.BCELoss()

    @staticmethod
    def _init_optimizer(optimizer_str, parameters, lr, hyper2=None):
        if optimizer_str.lower()=='sgd':
            return torch.optim.SGD(
                parameters,
                lr,
                momentum=hyper2
            )
        elif optimizer_str.lower()=='adam':
            return torch.optim.Adam(
                parameters,
                lr,
                weight_decay=hyper2
            )
        else:
            return torch.optim.AdamW(
                parameters,
                lr,
                weight_decay=hyper2
            )


class Transformer(torch.nn.Module):

    @staticmethod
    def _lossFunct(lf_type, IGNORE_ID):
        """
        Returns the specified loss function from torch.nn
        ______________________________________________________________
        Parameters:
        lf_type: str
            The loss function to return
        ______________________________________________________________
        Returns:
        lf: torch.nn.function
            The specified loss function
        """
        if lf_type == "cross-entropy":  # I:(N,C) O:(N)
            return torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)

        if lf_type == 'binary-cross-entropy':
            return torch.nn.BCELoss()

    def __init__(self, NORBERT, tokenizer, num_labels, IGNORE_ID,
                 device='cpu', epochs=10, lr_scheduler=False, factor=0.1,
                 lrs_patience=2, loss_funct='cross-entropy',
                 random_state=None, verbose=False, lr=2e-5, momentum=0.9,
                 epoch_patience=1, label_indexer=None, optmizer='SGD'):

        super().__init__()

        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # global parameters
        self.NORBERT = NORBERT
        self.num_labels = num_labels
        self.IGNORE_ID = IGNORE_ID
        self.device = device
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = lrs_patience
        self.epoch_patience = epoch_patience
        self.loss_funct_str = loss_funct
        self._loss_funct = self._lossFunct(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )
        self.lr = lr
        self.momentum = momentum
        self.last_epoch = None

        # setting model
        self.tokenizer = tokenizer
        self.model = BertForTokenClassification.from_pretrained(
            self.NORBERT,
            num_labels=self.num_labels,
        ).to(torch.device(self.device))

        # setting model's optimizer
        self.optmizer = optmizer
        if self.optmizer == 'SGD':
            self._opt = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum
            )
        elif self.optmizer == 'AdamW':
            self._opt = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.lr,
            )

        # setting learning rate's scheduler
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self._opt,
            mode='min',
            factor=self.factor,
            patience=self.patience
        )

        # storing scores
        self.losses = []
        self.binary_f1 = []
        self.propor_f1 = []

        # early stop
        self.early_stop_epoch = None

        # label indexer
        self.label_indexer = label_indexer

        # storing outputs
        self.outputs = None

    def forward(self, batch, inputs_embeds=False):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor

        """
        if inputs_embeds is False:
            return self.model(
                input_ids=batch[0].to(self.device),
                attention_mask=batch[2].to(self.device),
                output_hidden_states=True
            )
        else:
            return self.model(
                inputs_embeds=batch[0].to(self.device),
                attention_mask=batch[2].to(self.device),
                output_hidden_states=True
            )

    def backward(self, outputs, targets):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The real targets.
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        computed_loss = self._loss_funct(
            input=outputs,
            target=targets.to(torch.device(self.device))
            )

        # calculating gradients
        computed_loss.backward()

        # updating weights from the model by calling optimizer.step()
        self._opt.step()

        return computed_loss

    def fit(self, train_loader, verbose=False, dev_loader=None,
            need_pipeline=None):
        """
        Fits the model to the training data using the models
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        laoder: torch.nn.Dataloader=None
            Dataloader object to load the batches, defaults to None
        verbose: bool=False
            If True: prints out progressive output, defaults to False
        ______________________________________________________________
        Returns:
        None
        """
        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)

        for epoch in iterator:
            _loss = []

            for b, batch in enumerate(train_loader):
                self.train()
                outputs = self.forward(batch=batch)
                loss = self.backward(
                    outputs=outputs.logits.permute(0, 2, 1),
                    targets=batch[1]
                )
                _loss.append(loss.item())

                if verbose:
                    print(f"Batch: {b}  |"
                          f"  Train Loss: {loss}  |")

            if self._early_stop(epoch_idx=epoch,
                                patience=self.epoch_patience):
                print('Early stopped!')

                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)
                break

            else:
                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)

                if need_pipeline is not None:
                    _, _ = self.pipeline(
                        test_loader=dev_loader,
                        model1=need_pipeline,
                    )

        self.last_epoch = epoch
        return self

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions, golds, sents, golds_full = [], [], [], []

        for batch in tqdm(test_loader):
            out = self.forward(batch)
            y_pred = out.logits.argmax(2)
            predictions.append(y_pred.squeeze(0).tolist())
            golds.append(batch[1].squeeze(0).tolist())
            golds_full.append(batch[3].squeeze(0).tolist())

            for i in batch[0]:
                decoded_sentence = self.tokenizer.convert_ids_to_tokens(i)
                sents.append(decoded_sentence)

        # #################### truncating predictions, golds and sents
        predictions__, golds__, sents__, golds_full__ = [], [], [], []
        for l_p, l_g, l_s, l_gf in zip(predictions, golds, sents, golds_full):
            predictions_, golds_, sents_, golds_full_ = [], [], [], []

            for e_p, e_g, e_s, e_gf in zip(l_p, l_g, l_s, l_gf):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)
                    sents_.append(e_s)
                    golds_full_.append(e_gf)

            predictions__.append(predictions_)
            golds__.append(golds_)
            sents__.append(sents_)
            golds_full__.append(golds_full_)
        # ####################

        return predictions__, golds__, sents__, golds_full__

    def predict_old(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions, golds, sents = [], [], []

        for batch in tqdm(test_loader):
            out = self.forward(batch)
            y_pred = out.logits.argmax(2)
            predictions.append(y_pred.squeeze(0).tolist())
            golds.append(batch[1].squeeze(0).tolist())

            for i in batch[-1]:
                sents.append(i)

        # #################### truncating predictions and golds
        predictions__, golds__ = [], []
        for l_p, l_g in zip(predictions, golds):
            predictions_, golds_ = [], []

            for e_p, e_g in zip(l_p, l_g):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)

            predictions__.append(predictions_)
            golds__.append(golds_)
        # ####################

        return predictions__, golds__, sents

    def evaluate(self, test_loader):
        """
        Returns the binary and proportional F1 scores of the model on the
        examples passed via test_loader.

        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        preds, golds, sents, _ = self.predict(test_loader)
        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [int(i) for l in golds for i in l]

        print(len(sents))
        print(f'preds')
        print(len(preds))
        print(len(preds[0]))
        print(len(preds[1]))
        print('golds')
        print(len(golds))
        print(len(golds[0]))
        print(len(golds[1]))

        analysis = get_analysis(
            sents=sents,
            y_pred=preds,
            y_test=golds
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        return binary_f1, propor_f1

    # changed from val_losses to losses
    # but can be binary_f1 or propor_f1
    def _early_stop(self, epoch_idx, patience):
        if epoch_idx < patience:
            return False

        start = epoch_idx - patience

        # up to this index
        for count, loss in enumerate(
                self.losses[start + 1: epoch_idx + 1]):
            if loss > self.losses[start]:
                if count + 1 == patience:
                    self.early_stop_epoch = start
                    return True
            else:
                break

        return False

    def pipeline(self, test_loader, model1):
        print("################### PIPELINE's RESULTS ###################")
        preds1, golds1, sents1, gf1 = model1.predict(test_loader)
        preds2, golds2, sents2, gf2 = self.predict(test_loader)

        indexer = {
            "O": 0,
            "B-targ-Positive": 1,
            "I-targ-Positive": 2,
            "B-targ-Negative": 3,
            "I-targ-Negative": 4
        }

        BIO_indexer = {
            0: "O",
            1: "I-targ-",
            2: "B-targ-",
        }

        polarity_indexer = {
            0: "O",
            1: "Positive",
            2: "Negative",
        }

        idxs_bio, idxs_polarity = [], []
        for idx_b, e_bio in enumerate(sents1):
            for idx_p, e_polarity in enumerate(sents2):
                if e_bio == e_polarity:
                    idxs_bio.append(idx_b)
                    idxs_polarity.append(idx_p)
                    break

        collapsed_sents, collapsed_preds = [], []
        for i1, i2 in zip(idxs_bio, idxs_polarity):
            temp_sents, temp_preds, temp_golds = [], [], []

            for e in sents2[i2]:
                temp_sents.append(e)

            collapsed_sents.append(temp_sents)

            for e1, e2 in zip(preds1[i1], preds2[i2]):
                if BIO_indexer[e1] == polarity_indexer[e2]:
                    temp_preds.append(BIO_indexer[e1])
                elif BIO_indexer[e1] == 'O':
                    temp_preds.append(BIO_indexer[e1])
                elif polarity_indexer[e2] == 'O':
                    temp_preds.append(polarity_indexer[e2])
                else:
                    temp_preds.append(BIO_indexer[e1] + polarity_indexer[e2])

            collapsed_preds.append(temp_preds)

        preds = []
        for e_ in collapsed_preds:
            temp = []
            for e in e_:
                temp.append(indexer[e])
            preds.append(temp)

        flat_preds = [int(i) for l in preds for i in l]
        flat_golds = [int(i) for l in gf1 for i in l]

        analysis = get_analysis(
            sents=collapsed_sents,
            y_pred=preds,
            y_test=gf1
        )

        binary_f1 = binary_analysis(analysis)
        propor_f1 = proportional_analysis(flat_golds, flat_preds)
        print("################# END OF PIPELINE's RESULTS #################")
        return binary_f1, propor_f1


class TransformerMTL(Transformer):

    def __init__(self, NORBERT, tokenizer, num_labels, IGNORE_ID,
                 device='cpu', epochs=10, lr_scheduler=False, factor=0.1,
                 lrs_patience=2, loss_funct='cross-entropy',
                 random_state=None, verbose=False, lr=2e-5, momentum=0.9,
                 epoch_patience=1, label_indexer=None, optmizer='SGD',
                 previous_model=None, hs_type=None):

        super().__init__(NORBERT, tokenizer, num_labels, IGNORE_ID,
                         device, epochs, lr_scheduler, factor,
                         lrs_patience, loss_funct,
                         random_state, verbose, lr, momentum,
                         epoch_patience, label_indexer, optmizer)

        self.previous_model = previous_model
        self.hs_type = hs_type

    def fit(self, train_loader, verbose=False, dev_loader=None):
        """
        Fits the model to the training data using the models
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        laoder: torch.nn.Dataloader=None
            Dataloader object to load the batches, defaults to None
        verbose: bool=False
            If True: prints out progressive output, defaults to False
        ______________________________________________________________
        Returns:
        None
        """
        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)

        for epoch in iterator:
            _loss = []

            for b, batch in enumerate(train_loader):
                self.train()

                # ########## MTL
                out = self.previous_model.forward(batch)
                hidden_states = torch.stack(out.hidden_states[1:])

                if self.hs_type == 'last':
                    hidden_states = hidden_states[-1, :, :, :]
                elif self.hs_type == 'mean':
                    hidden_states = torch.mean(hidden_states, dim=0)

                outputs = self.forward(
                    batch=(hidden_states, batch[4], batch[2]),
                    inputs_embeds=True
                )
                # ##########
                loss = self.backward(
                    outputs=outputs.logits.permute(0, 2, 1),
                    targets=batch[4]
                )

                _loss.append(loss.item())

                if verbose:
                    print(f"Batch: {b}  |"
                          f"  Train Loss: {loss}  |")

            if self._early_stop(epoch_idx=epoch,
                                patience=self.epoch_patience):
                print('Early stopped!')

                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)
                break

            else:
                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {self.losses[epoch]}")

                if dev_loader is not None:
                    binary_f1, propor_f1 = \
                        self.evaluate(test_loader=dev_loader)
                    self.binary_f1.append(binary_f1)
                    self.propor_f1.append(propor_f1)

        self.last_epoch = epoch
        return self

    def predict(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions, golds, sents, golds_full = [], [], [], []

        for batch in tqdm(test_loader):

            # ########## MTL
            out = self.previous_model.forward(batch)
            hidden_states = torch.stack(out.hidden_states[1:])

            if self.hs_type == 'last':
                hidden_states = hidden_states[-1, :, :, :]
            elif self.hs_type == 'mean':
                hidden_states = torch.mean(hidden_states, dim=0)

            out = self.forward(
                batch=(hidden_states, batch[4], batch[2]),
                inputs_embeds=True
            )
            # ##########

            y_pred = out.logits.argmax(2)
            predictions.append(y_pred.squeeze(0).tolist())
            golds.append(batch[4].squeeze(0).tolist())
            golds_full.append(batch[3].squeeze(0).tolist())

            for i in batch[0]:
                decoded_sentence = self.tokenizer.convert_ids_to_tokens(i)
                sents.append(decoded_sentence)

        # #################### truncating predictions, golds and sents
        predictions__, golds__, sents__, golds_full__ = [], [], [], []
        for l_p, l_g, l_s, l_gf in zip(predictions, golds, sents,
                                       golds_full):
            predictions_, golds_, sents_, golds_full_ = [], [], [], []

            for e_p, e_g, e_s, e_gf in zip(l_p, l_g, l_s, l_gf):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)
                    sents_.append(e_s)
                    golds_full_.append(e_gf)

            predictions__.append(predictions_)
            golds__.append(golds_)
            sents__.append(sents_)
            golds_full__.append(golds_full_)
        # ####################

        return predictions__, golds__, sents__, golds_full__

    def predict_old(self, test_loader):
        """
        :param test_loader: torch.utils.data.DataLoader object with
                            batch_size=1
        """
        self.eval()
        predictions, golds, sents = [], [], []

        for batch in tqdm(test_loader):

            # ########## MTL
            out = self.previous_model.forward(batch)
            hidden_states = torch.stack(out.hidden_states[1:])

            if self.hs_type == 'last':
                hidden_states = hidden_states[-1, :, :, :]
            elif self.hs_type == 'mean':
                hidden_states = torch.mean(hidden_states, dim=0)

            out = self.forward(
                batch=(hidden_states, batch[4], batch[2]),
                inputs_embeds=True
            )
            # ##########

            y_pred = out.logits.argmax(2)
            predictions.append(y_pred.squeeze(0).tolist())
            golds.append(batch[4].squeeze(0).tolist())

            for i in batch[-1]:
                sents.append(i)

        # #################### truncating predictions, golds and sents
        predictions__, golds__ = [], []
        for l_p, l_g in zip(predictions, golds):
            predictions_, golds_ = [], []

            for e_p, e_g in zip(l_p, l_g):
                if e_g != self.IGNORE_ID:
                    predictions_.append(e_p)
                    golds_.append(e_g)

            predictions__.append(predictions_)
            golds__.append(golds_)
        # ####################

        return predictions__, golds__, sents
