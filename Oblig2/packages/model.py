import torch
import numpy as np

from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class RNNModel(nn.Module):

    def __init__(
        self, 
        emb=None, 
        num_features: int = 10, # this could be deleted TODO
        n_hl: int = 1, 
        n_classes: int = 2, 
        dropout: float = 0.2,
        epochs: int = 20, 
        units: int = 25, 
        # bias: float = 0.1,
        lr: float = 0.01, 
        momentum: float = 0.9,
        device: torch.device = torch.device("cpu"), # this is different than fabios
        weights_init: str = "xavier_normal",
        hl_actfunct: str = "tanh",
        out_actfunct: str = "relu",
        loss_funct: str = "cross-entropy",
        random_state: int = None,
        verbose: bool = False,
        # NOTE: Here I started adding parameters for my experiements
        blank_emb=False, # whether to load embeddings form pretrained or not.
        freeze=False,
        rnn_size=200, # should be the same as units
        rnn_n_hl=1,
        bidirectional=False,
        rnn_type='elman',   # elman, lstm, or gru
        optim='sgd',        # sgd, adam, (nnlc?)
        stack_size=1,       # for values >1 a stacked architecture will be built
    ) -> None:
        super().__init__()

        # seeding
        if random_state is not None:
            print(f'Setting torch random_state to {random_state}...') if verbose else None
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # parameters
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.hl_actfunct = self.actFunct(af_type=hl_actfunct)
        self.out_actfunct = self.actFunct(af_type=out_actfunct)
        self.loss_funct_str = loss_funct
        self.loss_funct = self.lossFunct(lf_type=loss_funct)
        self.bidirectional = bidirectional
        self.stack_size = stack_size
        self.rnn_type = rnn_type

        # set up the rnn layers
        rnn = self.rnn_architecture(rnn_type)
        self._rnn = rnn(
            input_size=emb.vector_size, 
            hidden_size=emb.vector_size,  # rnn_size, 
            num_layers=n_hl,    # should these be seperate to the MLP configs?
            batch_first=True,    # shape of input as (batch size, sent size, vec dimension)
            bidirectional=bidirectional,
            )
        self.stack = []
        if stack_size>1:
            # generate number of new rnn stacks to feed through 
            for _ in range(1, stack_size):
                self.stack.append(
                        rnn(
                            input_size=emb.vector_size, 
                            hidden_size=emb.vector_size,  # rnn_size, 
                            num_layers=n_hl,    # should these be seperate to the MLP configs?
                            batch_first=True,    # shape of input as (batch size, sent size, vec dimension)
                            bidirectional=bidirectional,
                            )
                )
        print(f'RNN layer:\t{self._rnn}') if self.verbose else None
        print(f'Stacks of RNNs:\t{self.stack}') if self.verbose and self.stack else None

        # config linear hidden layers
        if n_hl == 0:
            self.model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(num_features, n_classes),
                self.out_actfunct,
            )
        elif n_hl == 1:
            self.model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(num_features, units),
                self.hl_actfunct,
                torch.nn.Dropout(dropout),
                torch.nn.Linear(units, n_classes),
                self.out_actfunct,
            )
        elif n_hl >= 2:
            self.model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(num_features, units), #from size num_features -> units
                self.hl_actfunct
            )

            for i in range(1, n_hl):
                self.model.add_module(
                    name=f"HL{i + 1}-Dropout",
                    module=torch.nn.Dropout(dropout)
                )

                self.model.add_module(
                    name=f"HL{i + 1}-Linear",
                    module=torch.nn.Linear(units, units), #every inside layer dims units -> units
                )

                self.model.add_module(
                    name=f"HL{i + 1}-ActFunction",
                    module=self.hl_actfunct,
                )

            self.model.add_module(
                name="Output-Linear",
                module=torch.nn.Linear(units, n_classes),
            )

            self.model.add_module(
                name="Output-ActFunction",
                module=self.out_actfunct,
            )

        # config weights and biased for linear layers
        for m in self.model:

            if isinstance(m, torch.nn.Linear):

                # initializing bias
                torch.nn.init.constant_(m.bias, val=bias)

                # initializing weights
                if weights_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        # cast model to specified device
        self.model.to(device)

        # set model optimizer
        self.opt = self.get_optimizer(
            optim_str=optim,
            params=self.model.parameters(),
            lr=lr,
            momentum=momentum
        )

        # to keep track of loss throughout epochs?
        self.losses = None

        # in case the padding idx was forgotten in data load step
        try:
            pad_idx = emb.vocab['<pad>'].index
        except:
            pad_idx = 0

        # when you want to train embeddings on model inputs
        if blank_emb:
            # this should be a blank embedding setup
            self.word_embeddings = nn.Embedding(
                len(emb.vectors), 
                emb.vector_size, 
                padding_idx=pad_idx
                )
            # make sure to disable freeze parameter
        else:
            # this is a pretrained embedding setup
            print('Mr. Freeze = ', freeze)
            self.word_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(emb.vectors), 
                freeze=freeze,
                padding_idx=pad_idx
            )
        
        self.verbose = False

    def forward(self, batch, lengths=[]):
        print(f'input: {batch.size()}') if self.verbose else None
        # size([32, 43])
        # size([batch_size, sent_size])
        # tensor([sent, indexes for words in sentence])

        # apply the embedding weights stored in this model 
        embeds = self.word_embeddings(batch)
        print(f'embed: {embeds.size()}') if self.verbose else None
        # size([32, 43, 100])
        # size([batch_size, sent_size, dim_size])
        # tensor([sent, word in sent, vector for word])

        # now we need to disregard the padding indexes
        embeds = nn.utils.rnn.pack_padded_sequence(
            embeds, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
            )
        # feed through rnn layer
        hidden, last = self._rnn(embeds)
        # filter out not needed pads prt 2
        hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)

        if self.stack:
            for layer in self.stack:
                # same as before for each new layer
                hidden, last = layer(embeds)
                # filter out not needed pads prt 2
                hidden, _ = nn.utils.rnn.pad_packed_sequence(hidden, batch_first=True)
        
        if self.bidirectional:
            print('bidirectional. manipulate rnn output')
        else:
            print('non-bidirectional') if self.verbose else None

        print(f'hidden: {hidden.size()}') if self.verbose else None
        # size([32, 51])
        # size([batch_size, sent_size])
        # tensor([sent, indexes for words in sentence])
        print(f'last: {last.size()}') if self.verbose else None
        # what is this last thing? TODO

        # need to combind sentence using pooling
        X = torch.mean(hidden, dim=1) # try swapping out max, mean, sum
        print(f'X: {X.size()}') if self.verbose else None
        # either add more than just mean here, or feed directly to output
        


        # size([32, 100])
        # size([batch_size, sent_vec])
        # tensor([sent, vector representing combined emb of words in sent])
        
        # feed current state to linear sequential model as before
        return self.model(X) # size ([32, 2]) 

        # # try new approach that only return the softmax of rnn output
        # return self.out_actfunct(hidden)

    def fit(self, loader=None, verbose=False) -> None:
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
        self.losses = np.empty(shape=self.epochs, dtype=float)
        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)

        for i in iterator:
            _loss = []
            for n, batch in enumerate(loader):
                # TODO clean try/except
                try:
                    text, source = batch
                    lengths = []
                except:
                    text, source, lengths = batch
                text = text.to(self.device)
                source = source.to(self.device)
                output = self.forward(text, lengths)
                loss = self.backward(output, source)
                _loss.append(loss.item())

            self.losses[i] = np.mean(_loss)

            print(f'Epoch: {i} loss:', np.mean(_loss))

    # not sure if this is necessary, but i think it is..
    def backward(self, output, target):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
            Parameters:
            output: MLPModel
                The output after the last epoch
            target:
                The target it is optimizing towards
            ______________________________________________________________
            Returns:
            loss: float
                How close the estimate was to the gold standard.
        """
        target = self.tranform_target_for_loss(target, self.loss_funct_str)
        # # calculating the loss
        loss_function = self.loss_funct
        print('target size:', target.size()) if self.verbose else None
        print('output size:', output.size()) if self.verbose else None

        loss = loss_function(output, target)

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self.opt.zero_grad()

        # calculating gradients
        loss.backward()

        # updating weights from the model by calling optimizer.step()
        self.opt.step()

        return loss

    def predict_classes(self, input_tensor):
        """
        Makes predictions from a test tensor using the model.
            ______________________________________________________________
            Parameters:
            input_tensor: torch.nn.tensor
                The tensor to make predictions on.
            ______________________________________________________________
            Returns:
            y_pred: np.array
                An array containing the predicted classes of the input 
                tensors.
        """
        print('predicting...')
        print('input size:', input_tensor.size())
        # x_em = self.word_embeddings(input_tensor)
        # x = torch.max(x_em, dim=1)
        # print('x emb size:', x.size())

        # need to get the lengths of these new inputs
        lengths = torch.LongTensor([x.size(0) for x in input_tensor])


        y_pred = self.forward(input_tensor, lengths).detach()
        print('y_pred size:', y_pred.size())
        return torch.argmax(y_pred, dim=1) # shold this be argmax or max?

    @staticmethod
    def rnn_architecture(rnn_type: str):
        if rnn_type.lower()=='elman':
            return nn.RNN
        elif rnn_type.lower()=='lstm':
            return nn.LSTM
        elif rnn_type.lower()=='gru':
            return nn.GRU
        return nn.RNN

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
    def lossFunct(lf_type: str):
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
        if lf_type == "cross-entropy":
            # needs squeezed target
            return torch.nn.CrossEntropyLoss()

        elif lf_type == "accuracy":
            return accuracy_score() # might be problem bc not torch

        elif lf_type == "hinge-embedding":
            # needs plain target (no squeeze)
            return torch.nn.HingeEmbeddingLoss()

        elif lf_type == "bce" or lf_type == 'binary-cross-entropy':
            # needs squeezed target
            return torch.nn.BCELoss()

        elif lf_type == "bce-logit":
            # needs squeezed target
            return torch.nn.BCEWithLogitsLoss()

        elif lf_type == "soft-margin":
            # check target structure
            return torch.nn.SoftMarginLoss()
            
        elif lf_type == "cosine-embedding":
            # check target structure
            return torch.nn.CosineEmbeddingLoss()
    
    # is this really needed if the data is loaded correctly?
    @staticmethod
    def tranform_target_for_loss(target, loss_funct_str):
        '''
        fix target to match what loss needs

        '''
        # ### MEAN AFTER 
        # need_squeeze = [
        #     "binary-cross-entropy",
        #     'bce-logit',
        # ]

        # need_encode = [
        #     'cross-entropy',
        # ]

        need_squeeze = [
            'binary-cross-entropy',
            'bce-logit',
            'cross-entropy',
        ]

        need_encode = [
            "binary-cross-entropy",
            'bce-logit',
        ]

        need_float = [
            'binary-cross-entropy',
            'bce-logit',
        ]

        if loss_funct_str in need_squeeze:
            target = target.squeeze(1)
        if loss_funct_str in need_encode:
            target = F.one_hot(target)
        if loss_funct_str in need_float:
            target = target.float()
        return target

    @staticmethod
    def get_lengths(X, pad_token=0):
        lengths = []
        longest = X.size(-1)
        
        X[0:10, -5:] = pad_token
        lengths.extend([longest - 5] * 10)
                
        X[10:18, -4:] = pad_token
        lengths.extend([longest - 4] * 8)

        X[18:-1, -2:] = pad_token
        lengths.extend([longest - 2] * 13)
        
        X[-1, -2:] = 140
        lengths.extend([longest])
        
        return X, torch.LongTensor(lengths)
    
    @staticmethod
    def get_optimizer(optim_str, params, lr, momentum):

        if optim_str.lower()=='sgd':
            return torch.optim.SGD(
                params=params,
                lr=lr,
                momentum=momentum
            )
        elif optim_str.lower()=='adam':
            return torch.optim.Adam(
                params=params,
                lr=lr,
                weight_decay= (1. - momentum)
            )
        elif optim_str.lower()=='adagrad':
            return torch.optim.Adagrad(
                params=params,
                lr=lr,
                weight_decay=(1.-momentum)
            )
        if optim_str.lower()=='adadelta':
            return torch.optim.Adadelta(
                params=params,
                lr=lr,
                weight_decay=(1.-momentum)
            )
