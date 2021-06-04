# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from itertools import chain
from transformers import BertForTokenClassification, BertModel
from sklearn.metrics import f1_score
from packages.metrics import evaluate
from tqdm import tqdm
import numpy as np
import torch


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

    def __init__(self, NORBERT, num_labels, NOT_ENTITY_ID, device='cpu',
                 epochs=10, lr_scheduler=False, factor=0.1, patience=2,
                 loss_funct='cross-entropy', random_state=None,
                 verbose=False, lr=2e-5, momentum=0.9, epoch_patience=1,
                 label_indexer=None):

        super().__init__()

        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # parameters
        self.NORBERT = NORBERT
        self.num_labels = num_labels
        self.IGNORE_ID = num_labels
        self.NOT_ENTITY_ID = NOT_ENTITY_ID
        self.device = device
        self.epochs = epochs
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.epoch_patience = epoch_patience
        self.loss_funct_str = loss_funct
        self._loss_funct = self._lossFunct(
            lf_type=loss_funct,
            IGNORE_ID=self.IGNORE_ID
        )
        self.lr = lr
        self.momentum = momentum
        self.last_epoch = None
        self.eval = None

        # setting model
        self.model = BertForTokenClassification.from_pretrained(
            self.NORBERT,
            num_labels=self.num_labels,
        ).to(torch.device(self.device))

        # setting model's optimizer
        self._opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
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
        self.val_losses = []
        self.val_f1_scores = []

        # early stop
        self.early_stop_epoch = None

        # storing training forward's outputs
        self.outputs = None

        # label indexer
        self.label_indexer = label_indexer

        # train, valid modes
        self.is_valid = False

    def train_mode(self):
        self.is_valid = False

    def eval_mode(self):
        self.is_valid = True

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
        outputs = self.model(
            input_ids=batch[0],
            attention_mask=batch[2],
            output_hidden_states=True
        )

        if self.is_valid is False:
            self.outputs = outputs

        return outputs.logits

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
            target=targets
        )

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self._opt.zero_grad()

        # calculating gradients
        computed_loss.backward()

        # updating weights from the model by calling optimizer.step()
        self._opt.step()

        return computed_loss

    def fit(self, loader=None, verbose=False, test=None):
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

        if test is not None:
            batch_test = next(iter(test))

        for epoch in iterator:
            _loss = []
            _val_loss = []
            _val_F1 = []

            for b, batch in enumerate(loader):
                outputs = self.forward(batch=batch)
                loss = self.backward(
                    outputs=outputs.permute(0, 2, 1),
                    targets=batch[1]
                )

                _loss.append(loss.item())

                if self.lr_scheduler is True:
                    self._scheduler.step(loss.item())

                if test is not None:
                    saved_f1_scores, saved_loss_valid = \
                        self._validate(batch_test)
                    

                    _val_F1.append(np.mean(saved_f1_scores))
                    _val_loss.append(np.mean(saved_loss_valid))

                    if verbose:
                        print(f"Batch: {b}  |"
                              f"  Train Loss: {loss}  |"
                              f"  Valid Loss: {saved_loss_valid}  |"
                              f"  F1 Valid: {np.mean(saved_f1_scores)}")

                else:
                    if verbose:
                        print(f"Batch: {b}  |"
                              f"  Train Loss: {loss}  |")


            if test is not None:
                y_true, y_pred = self.predict_classes(test)
                self.eval = evaluate(y_pred, y_true, self.label_indexer)

                self.losses.append(np.mean(_loss))  # train
                self.val_losses.append(np.mean(_val_loss))  # validation
                self.val_f1_scores.append(np.mean(_val_F1))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {np.mean(_loss)}  |"
                          f"  Valid Loss: {np.mean(_val_loss)}  |"
                          f"  F1 Valid: {np.mean(_val_F1)}  |" 
                          f"  Their F1: {self.eval}")

                if self._early_stop(epoch_idx=epoch,
                                    patience=self.epoch_patience):
                    print('Early stopped!')
                    break

            else:
                self.losses.append(np.mean(_loss))

                if verbose:
                    print(f"Epoch: {epoch}  |"
                          f"  Train Loss: {np.mean(_loss)}")

            self.last_epoch = epoch
            
        return self

    def _eval(self, test_loader):
        y_pred, y_true = self.predict_classes(test_loader)
        self.eval = evaluate(y_pred, y_true, self.label_indexer)

    def _validate(self, batch_test):
        self.eval_mode()
        y_valid = self.forward(batch_test)

        saved_f1_scores, saved_exact = [], []
        for i in range(batch_test[1].size(0)):  # for sentence in batch
            y_true, y_pred = [], []

            for true, pred in zip(  # for token in sentence
                    batch_test[1][i],  # targets_test
                    y_valid.max(dim=2)[1][i]):

                if (true != self.IGNORE_ID) or (true != self.NOT_ENTITY_ID):
                    y_true.append(true.item())
                    y_pred.append(pred.item())

            saved_f1_scores.append(f1_score(
                y_pred=y_pred,
                y_true=y_true,
                average='weighted')
            )

        saved_loss_valid = self._loss_funct(
            input=y_valid.permute(0, 2, 1),
            target=batch_test[1]
        ).item()

        self.train_mode()
        return saved_f1_scores, saved_loss_valid

    def _early_stop(self, epoch_idx, patience):
        if epoch_idx < patience:
            return False

        start = epoch_idx - patience

        # up to this index
        for count, loss in enumerate(
                self.val_losses[start + 1: epoch_idx + 1]):
            if loss > self.val_losses[start]:
                if count + 1 == patience:
                    self.early_stop_epoch = start
                    return True
            else:
                break

        return False

    def predict_classes(self, loader, batch=None, get_batch=False):
        if loader:
            batch = next(iter(loader))

        y_pred = self.forward(batch)

        if get_batch:
            return batch[0], batch[1], y_pred.max(dim=2)[1], batch[2]

        else:
            return batch[1], y_pred.max(dim=2)[1]


class TransformerRNN(Transformer):

    @staticmethod
    def model_constructor(n_hl, units, dropout, input_size, rnn_type,
                          nonlinearity, bidirectional):
        model = None
        if rnn_type == "rnn":
            model = torch.nn.RNN(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                nonlinearity=nonlinearity,  # -> 'tanh' or 'relu'
                batch_first=True,  # -> (batch, seq, feature)
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "lstm":
            model = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "gru":
            model = torch.nn.GRU(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        return model

    def __init__(self, NORBERT, num_labels, NOT_ENTITY_ID, transfered_model,
                 device='cpu', epochs=10, lr_scheduler=False, factor=0.1,
                 patience=2, loss_funct='cross-entropy', random_state=None,
                 verbose=False, lr=0.01, momentum=0.9, epoch_patience=1,
                 num_hidden_layers=1, hidden_size=50, dropout=0.2,
                 rnn_type='gru', nonlinearity='tanh', bidirectional=False,
                 freeze=False, previous_hidden_states_type='mean', label_indexer=None):

        super().__init__(NORBERT, num_labels, NOT_ENTITY_ID, device,
                         epochs, lr_scheduler, factor, patience,
                         loss_funct, random_state, verbose, lr, momentum,
                         epoch_patience, label_indexer)
        # parameters:
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.nonlinearity = nonlinearity
        self.freeze = freeze
        self.transfered_model = transfered_model
        self.bidirectional = bidirectional
        self.input_size = 768  # length of the input embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size

        self.model = self.model_constructor(
            n_hl=self.num_hidden_layers,
            units=self.hidden_size,
            dropout=self.dropout,
            input_size=self.input_size,
            rnn_type=self.rnn_type,
            nonlinearity=self.nonlinearity,
            bidirectional=self.bidirectional
        ).to(torch.device(self.device))

        # # output layer
        self._linear = self.transfered_model.model.classifier

        # self._linear = torch.nn.Linear(
        #     in_features=self.hidden_size,
        #     out_features=self.num_labels,
        #     bias=True
        # ).to(torch.device(self.device))

        # setting model's optimizer
        self._opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )

        self.previous_hidden_states_type = previous_hidden_states_type

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor (size[batch_size, seq_len])

        """
        self.lengths = torch.sum(batch[2], dim=1).detach().cpu()

        # >> (batch, seq_len, input_size)
        self.transfered_model.forward(batch)
        self.out = self.transfered_model.outputs.hidden_states[0]

        # >> (num_layers, batch, hidden_size)
        self.last_hidden_state = torch.stack(
            self.transfered_model.outputs.hidden_states[1:])

        if self.previous_hidden_states_type == 'mean_hidden_state':
            self.last_hidden_state = torch.mean(self.last_hidden_state, dim=2)

        elif self.previous_hidden_states_type == 'cls_hidden_state':
            self.last_hidden_state = self.last_hidden_state[:, :, 0, :]

        self.packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=self.out,
            lengths=self.lengths,
            batch_first=True,
            enforce_sorted=False
        )

        if self.rnn_type in ["rnn", "gru"]:
            # fitting the model
            # <- [batch:n_samples, longest_sentence:seq_len, input.size(1)]
            # -> [batch:n_samples, seq_len, hidden_size*num_directions]
            out, states = self.model(
                self.packed,
                self.last_hidden_state
            )

        elif self.rnn_type == "lstm":
            # [batch, num_layers * num_directions, hidden_size]
            c0 = torch.zeros(
                self.num_hidden_layers,
                batch[0].size(0),
                self.hidden_size
            ).to(torch.device(self.device))

            out, states = self.model(
                self.packed,
                (self.last_hidden_state, c0)
            )

        self.seq_unpacked, lens_unpacked = \
            torch.nn.utils.rnn.pad_packed_sequence(
                sequence=out,
                batch_first=True
            )

        return self._linear(self.seq_unpacked)


class TransformerMLP(Transformer):

    @staticmethod
    def actFunct(af_type: str):
        """
        Returns the specified activation type from torch.nn
        ______________________________________________________________
        Parameters:
        af_type: str
            The Activation function to return
        ______________________________________________________________
        Returns:
        af: torch.nn.function
            The specified activation function
        """
        if af_type == "relu":
            return torch.nn.ReLU()

        if af_type == "sigmoid":
            return torch.nn.Sigmoid()

        elif af_type == "tanh":
            return torch.nn.Tanh()

        elif af_type == "softmax":
            return torch.nn.Softmax(dim=1)

    @staticmethod
    def model_constructor(n_hl, units, dropout, input_size, hl_actfunct,
                          device):
        model = None
        if n_hl == 1:
            model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, units),
                hl_actfunct,
                torch.nn.Dropout(dropout),
            ).to(torch.device(device))

        elif n_hl >= 2:
            model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, units),
                hl_actfunct
            ).to(torch.device(device))

            for i in range(1, n_hl):
                model.add_module(
                    name=f"HL{i + 1}-Dropout",
                    module=torch.nn.Dropout(dropout)
                )

                model.add_module(
                    name=f"HL{i + 1}-Linear",
                    module=torch.nn.Linear(units, units),
                )

                model.add_module(
                    name=f"HL{i + 1}-ActFunction",
                    module=hl_actfunct,
                )

        return model

    def __init__(self, NORBERT, num_labels, NOT_ENTITY_ID, transfered_model,
                 device='cpu', epochs=10, lr_scheduler=False, factor=0.1,
                 patience=2, loss_funct='cross-entropy', random_state=None,
                 verbose=False, lr=0.01, momentum=0.9, epoch_patience=1,
                 hl_actfunct='tanh', out_actfunct='relu', dropout=0.2,
                 hidden_size=768, num_hidden_layers=12):

        super().__init__(NORBERT, num_labels, NOT_ENTITY_ID, device,
                         epochs, lr_scheduler, factor, patience,
                         loss_funct, random_state, verbose, lr, momentum,
                         epoch_patience)
        # parameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.transfered_model = transfered_model
        self.hl_actfunct = self.actFunct(af_type=hl_actfunct)
        self._out_actfunct = self.actFunct(af_type=out_actfunct)

        self.model = self.model_constructor(
            n_hl=self.num_hidden_layers,
            units=self.hidden_size,
            dropout=self.dropout,
            input_size=768,
            hl_actfunct=self.hl_actfunct,
            device=self.device,
        )

        # output layer
        self._linear = self.transfered_model.model.classifier

        # self._linear = torch.nn.Linear(
        #     in_features=self.hidden_size,
        #     out_features=self.num_labels,
        #     bias=True
        # ).to(torch.device(self.device))

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor (size[batch_size, seq_len])

        """
        # >> (batch, seq_len, input_size)
        self.transfered_model.forward(batch)
        out = self.model(self.transfered_model.outputs.hidden_states[0])
        return self._out_actfunct(self._linear(out))


##### Per working form here down #####
class BertSimple(Transformer):
    def __init__(
        self,
        NORBERT,
        num_labels,
        NOT_ENTITY_ID,
        device='cpu',
        epochs=10,
        lr_scheduler=False,
        lr_factor=0.1,
        lr_patience=2,
        loss_funct='cross-entropy',
        random_state=None,
        verbose=False,
        lr=0.01,
        momentum=0.9,
        epoch_patience=1,
        label_indexer=None
    ):

        super().__init__(
            NORBERT, 
            num_labels, 
            NOT_ENTITY_ID, 
            device,
            epochs, 
            lr_scheduler, 
            lr_factor, 
            lr_patience,
            loss_funct, 
            random_state,
            verbose, 
            lr, 
            momentum, 
            epoch_patience,
            label_indexer
        )

        # overwrite BertForTokenClassification _model in parent
        self._model = \
            BertModel.from_pretrained(NORBERT).to(torch.device(self.device))
        self._linear = torch.nn.Linear(
            in_features=768, # TODO get from BertModel
            out_features=self.num_labels,
            bias=True
        )

        # overwrite optimizer to include both linear layer and BertModel 
        # setting model's optimizer
        self._opt = torch.optim.SGD( # TODO tune to Adam and AdamW
            params=chain(
                self._model.parameters(),
                self._linear.parameters()
            ),
            lr=self.lr,
            momentum=self.momentum
        )

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor (size[batch_size, seq_len])

        """
        X = self._model(batch[0]).last_hidden_state # -> [batch_size, seq_len, 768]
        return self._linear(X) # -> [batch_size, seq_len, 18]


class BertRNN(Transformer):

    @staticmethod
    def model_constructor(n_hl, units, dropout, input_size, rnn_type,
                          nonlinearity, bidirectional):
        model = None
        if rnn_type == "rnn":
            model = torch.nn.RNN(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                nonlinearity=nonlinearity,  # -> 'tanh' or 'relu'
                batch_first=True,           # -> (batch, seq, feature)
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "lstm":
            model = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "gru":
            model = torch.nn.GRU(
                input_size=input_size,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        return model

    def __init__( 
        self, 
        NORBERT, 
        num_labels, 
        NOT_ENTITY_ID, 
        fine_tune_bert = False, # NEW!! TODO add to MLP and Simple
        device='cpu',
        epochs=10, 
        epoch_patience=1,
        label_indexer=None,
        lr=0.01, 
        lr_scheduler=False, # TODO test/tune
        lr_factor=0.1, 
        lr_patience=2,
        loss_funct='cross-entropy', 
        momentum=0.9, 
        random_state=None,
        verbose=False, 

        ## New for RNN  
        bidirectional = True,    
        dropout=0.1,       # rnn dropout
        input_size=768,    # make sure model is BertModel not ForTokenClassification 
        n_hl=6,            # hidden layers of rnn
        nonlinearity: str = 'tanh',    # tanh, relu, ..?
        pool_type: str = 'cat',        # TODO check if this is even used
        rnn_type='gru',    # rnn, lstm, gru
        units=512,         # units per layer of rnn (default BERT val)
        last_hidden_state_path = './saga/rnn_last_hidden_state_path.pt', 
        _bert_model = None, # preloaded model or path (str) else None
    ):

        super().__init__(
            NORBERT, num_labels, NOT_ENTITY_ID, device,
            epochs, lr_scheduler, lr_factor, lr_patience,
            loss_funct, random_state,
            verbose, lr, momentum, epoch_patience, label_indexer
        )

         # parameters:
        self.n_hl = n_hl                        # <- num_layers
        self.input_size = input_size            # <- input_size
        self.dropout = dropout
        self.input_size = input_size
        self.units = units
        self.rnn_type = rnn_type
        self.nonlinearity = nonlinearity
        self.bidirectional = bidirectional
        self.last_hidden_state_path = last_hidden_state_path

        # for saving/loading initial states of rnn
        self.rnn_last_hidden_state = self._load_rnn_hidden_state(last_hidden_state_path)
        # self.in_train_mode = True # to protect last hidden state from updated when evaluating
        #  this attr is built in called self.training

        if _bert_model:
            self._model = _bert_model.to(torch.device(self.device))
        else:
            self._model = BertModel.from_pretrained(NORBERT)
            self._model.to(torch.device(self.device))

        self._rnn = self.model_constructor(
            n_hl=self.n_hl,
            units=units,
            dropout=self.dropout,
            input_size=self.input_size,
            rnn_type=self.rnn_type,
            nonlinearity=self.nonlinearity,
            bidirectional=self.bidirectional
        ).to(torch.device(self.device))

        # output layer
        if self.bidirectional is True:
            self._linear = torch.nn.Linear(
                in_features=self.units*2,
                out_features=self.num_labels,
                bias=True
            ).to(torch.device(self.device))
        else:
            self._linear = torch.nn.Linear(
                in_features=self.units,
                out_features=self.num_labels,
                bias=True
            ).to(torch.device(self.device))

        if fine_tune_bert:
            optim_params = chain(
                self._model.parameters(),
                self._rnn.parameters(),
                self._linear.parameters()
            )
        else:
            optim_params = chain(
                self._rnn.parameters(),
                self._linear.parameters()
            )
        
        # setting model's optimizer
        self._opt = torch.optim.SGD(
            params=optim_params,
            lr=self.lr,
            momentum=self.momentum
        )

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor (size[batch_size, seq_len])

        """
        lengths = torch.sum(batch[2], dim=1)
        output = self._model(
            batch[0],
            attention_mask=batch[2],
            output_hidden_states = True # need hidden state for rnn init
        ) # direct  output is what we need

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=output.last_hidden_state, # final state of Bert embedding representation
            lengths=lengths, # attention_mask
            batch_first=True,
            enforce_sorted=False
        )


        out, states = self._rnn(packed, ) 
        '''
        Initial states:
            in order to load previously trained initial states for *this* rnn structure,
            we need to fit this model once (from scratch), save the final state of this rnn,
            then load the last hidden state (output)to this rnn.
        '''
        if self.training:
            self.rnn_last_hidden_state = states
        else:
            del states
       
        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=True
        )

        return self._linear(seq_unpacked)

    def _load_rnn_hidden_state(self, lhs_path='./saga/rnn_last_hidden_state.pt'): 
        try:
            return torch.load(lhs_path)
        except:
            print('could not find hidden state at:', lhs_path)
            return None


class BertMLP(Transformer):

    @staticmethod
    def actFunct(af_type: str):
        """
        Returns the specified activation type from torch.nn
        ______________________________________________________________
        Parameters:
        af_type: str
            The Activation function to return
        ______________________________________________________________
        Returns:
        af: torch.nn.function
            The specified activation function
        """
        if af_type == "relu":
            return torch.nn.ReLU()

        if af_type == "sigmoid":
            return torch.nn.Sigmoid()

        elif af_type == "tanh":
            return torch.nn.Tanh()

        elif af_type == "softmax":
            return torch.nn.Softmax(dim=1)

    @staticmethod
    def build_hidden_layers(
        n_hl, dropout, input_size, num_labels, units, bias, weights_init,
        out_actfunct, hl_actfunct,
    ):
        
        if n_hl == 0:
            model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, num_labels),
                out_actfunct,
            )

        elif n_hl == 1:
            model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, units),
                hl_actfunct,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(units, num_labels),
                out_actfunct,
            )

        else: # n_hl >= 2:
            model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(input_size, units),
                hl_actfunct
            )

            for i in range(1, n_hl):
                model.add_module(
                    name=f"HL{i + 1}-Dropout",
                    module=torch.nn.Dropout(dropout)
                )

                model.add_module(
                    name=f"HL{i + 1}-Linear",
                    module=torch.nn.Linear(units, units),
                )

                model.add_module(
                    name=f"HL{i + 1}-ActFunction",
                    module=hl_actfunct,
                )

            model.add_module(
                name="Output-Linear",
                module=torch.nn.Linear(units, num_labels),
            )

            model.add_module(
                name="Output-ActFunction",
                module=out_actfunct,
            )

        for m in model:

            if isinstance(m, torch.nn.Linear):

                # initializing bias
                torch.nn.init.constant_(m.bias, val=bias)

                # initializing weights
                if weights_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        return model

    def build_optimizer(self, optimizer='sgd'):
        # get parameters to optimize
        if self.fine_tune_bert:
            optim_params = chain(
                self._model.parameters(),
                self._mlp.parameters()
            )
        else:
            optim_params = self._mlp.parameters()

        if optimizer=='sgd':
            return torch.optim.SGD(
                params=optim_params,
                lr=self.lr,
                momentum=self.momentum
            )
        elif optimizer == 'adam':
            return torch.optim.Adam(
                params=optim_params,
                lr=self.lr,
            )
        elif optimizer == 'adamw':
            return torch.optim.AdamW(
                params=optim_params,
                lr=self.lr,
            )
        return torch.optim.SGD(
                params=optim_params,
                lr=self.lr,
                momentum=self.momentum
            )

    def __init__(
        self, NORBERT, num_labels, NOT_ENTITY_ID, device='cpu',
        epochs=10, lr_scheduler=False, lr_factor=0.1, lr_patience=2,
        loss_funct='cross-entropy', random_state=None,
        verbose=False, lr=0.01, momentum=0.9, epoch_patience=1,
        label_indexer=None,

        ############ New for MLP
        bias=0.1,
        dropout=0.2,
        fine_tune_bert=False,
        hl_actfunct='tanh',
        input_size=768,
        n_hl=1,
        out_actfunct='relu',
        units=50,
        weights_init='xavier_normal',
        optimizer = 'sgd',
    ):

        super().__init__(NORBERT, num_labels, NOT_ENTITY_ID, device,
                         epochs, lr_scheduler, lr_factor, lr_patience,
                         loss_funct, random_state, verbose, lr, momentum,
                         epoch_patience, label_indexer)

        # parameters
        self.bias = bias
        self.dropout = dropout
        self.fine_tune_bert=fine_tune_bert
        self.input_size = input_size
        self.hl_actfunct = self.actFunct(af_type=hl_actfunct)
        self.n_hl = n_hl
        self.out_actfunct = self.actFunct(af_type=out_actfunct)
        self.units = units
        self.weights_init = weights_init

        # load new BertModel
        self._model = \
            BertModel.from_pretrained(NORBERT).to(torch.device(self.device))

        self._mlp = self.build_hidden_layers(
            self.n_hl, self.dropout, self.input_size, self.num_labels, 
            self.units, self.bias, self.weights_init,
            self.out_actfunct, self.hl_actfunct,
        )
        self._mlp.to(torch.device(self.device))


        self._opt = self.build_optimizer(
            optimizer=optimizer
        )

        

    def forward(self, batch):
        """
        Performs a feed forward step.
        ______________________________________________________________
        Parameters:
        batch: tuple containing (input_ids, targets, att_maks)

        ______________________________________________________________
        Returns:
        outputs: torch.Tensor (size[batch_size, seq_len])

        """
        X = self._model(batch[0]).last_hidden_state
        return self._mlp(X)
        # how to cast Float type to Long type
