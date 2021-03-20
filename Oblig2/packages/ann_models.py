# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import torch


class MLPModel(torch.nn.Module):

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
            return torch.nn.CrossEntropyLoss()  # I:(N,C) O:(N)

        elif lf_type == "hinge-embedding":
            # needs plain target (no squeeze)
            return torch.nn.HingeEmbeddingLoss()

        elif lf_type == "bce" or \
                lf_type == 'binary-cross-entropy':
            # needs squeezed target
            return torch.nn.BCELoss()  # I:(N,C) O:(N, C)

        elif lf_type == "bce-logit":
            # needs squeezed target
            return torch.nn.BCEWithLogitsLoss()  # I:(N,C) O:(N, C)

        elif lf_type == "soft-margin":  # target .1 and 1
            # check target structure
            return torch.nn.SoftMarginLoss()

    @staticmethod
    def transform_target_for_loss(target, loss_funct_str):
        """fix target to match what loss needs."""

        need_encode = ['cross-entropy']

        if loss_funct_str in need_encode:
            target = target.squeeze(1)

        return target

    def __init__(self, emb, n_hl: int = 1, num_features: int = 10,
                 n_classes: int = 2, dropout: float = 0.2,
                 epochs: int = 5, units: int = 25, bias: float = 0.1,
                 lr: float = 0.01, momentum: float = 0.9,
                 device: torch.device = torch.device("cpu"),
                 weights_init: str = "xavier_normal",
                 hl_actfunct: str = "tanh",
                 out_actfunct: str = "relu",
                 loss_funct: str = "cross-entropy",
                 random_state: int = None,
                 verbose: bool = False,
                 embedding_type: str = "mean",
                 freeze: bool = False) -> None:
        """
        Creates a multilayer perceptron object with the specified
        parameters using the pytorch framework.
            ______________________________________________________________
            Parameters:
            n_hl: int = 1
                Number of hidden layers, defaults to 1
            num_features: int = 10
                Number of features, defaults to 10
            n_classes: int = 10
                Number of classes, defaults to 10
            dropout: float = 0.2
                Dropout value, defaults to 0.2
            epochs: int = 50
                Number of epochs to run, defaults to 20
            units: int = 25
                Number of units per hidden layer, defaults to 25
            bias: float = 0.1
                Bias value, defaults to 0.1
            lr: float = 0.01
                Learning rate, defaults to 0.01
            momentum: float = 0.9
                Momentum value, defaults to 0.9
            device: torch.device = torch.device("cpu")
                Specifies how the model is run, defaults to "cpu"
            weights_init: str = "xavier_normal"
                Specifies how the weights are initialized, defaults to
                "xavier_normal"
            hl_actfunct: str = "tanh"
                Hidden layer activation function, defaults to "tahn"
            out_actfunct: str = "relu"
                Output activation function, defaults to "relu"
            loss_funct: str = "cross-entropy"
                Loss function, defaults to "cross-entropy"
            random_state: int = None
                The seed for the random state
            verbose: bool = False
                If True: prints out progressive output, defaults to False
            ______________________________________________________________
            Returns:
            None
        """
        super().__init__()

        # seeding
        if random_state is not None:
            print(f'Setting torch random_state to {random_state}...') \
                if verbose else None
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # parameters
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.hl_actfunct = self.actFunct(af_type=hl_actfunct)
        self.out_actfunct = self.actFunct(af_type=out_actfunct)
        self.out_actfunct_str = out_actfunct
        self.loss_funct_str = loss_funct
        self.loss_funct = self.lossFunct(lf_type=loss_funct)
        self.freeze = freeze

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
                torch.nn.Linear(num_features, units),
                self.hl_actfunct
            )

            for i in range(1, n_hl):
                self.model.add_module(
                    name=f"HL{i + 1}-Dropout",
                    module=torch.nn.Dropout(dropout)
                )

                self.model.add_module(
                    name=f"HL{i + 1}-Linear",
                    module=torch.nn.Linear(units, units),
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

        for m in self.model:

            if isinstance(m, torch.nn.Linear):

                # initializing bias
                torch.nn.init.constant_(m.bias, val=bias)

                # initializing weights
                if weights_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        self.model.to(device)

        self.opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=lr,
            momentum=momentum
        )

        self.losses = None

        vectors = torch.FloatTensor(emb.vectors)

        if device == torch.device("cuda"):
            vectors = vectors.to(device)

        self.word_embeddings = torch.nn.Embedding.from_pretrained(
            vectors, freeze=self.freeze)

        self.embedding_type = embedding_type

    def forward(self, batch):
        """
        Performs a forward step on the model.
        ______________________________________________________________
        Parameters:
        batch: torch.nn.tensor
            The mini-batch input tensor to update
        ______________________________________________________________
        Returns:
        self.model: MLPModel
            The updated model
        """
        x = None
        if self.embedding_type == "mean":
            x = torch.mean(self.word_embeddings(batch), dim=1)

        elif self.embedding_type == "sum":
            x = torch.sum(self.word_embeddings(batch), dim=1)

        elif self.embedding_type == "max":
            x = torch.max(self.word_embeddings(batch), dim=1)

        return self.model(x)

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
        x = None
        if self.embedding_type == "mean":
            x = torch.mean(self.word_embeddings(input_tensor), dim=1)

        elif self.embedding_type == "sum":
            x = torch.sum(self.word_embeddings(input_tensor), dim=1)

        elif self.embedding_type == "max":
            x = torch.max(self.word_embeddings(input_tensor), dim=1)

        y_pred = self.model(x)
        return y_pred.max(dim=1)[1]

    def backward(self, output, target):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The target it is optimizing towards
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        target = self.transform_target_for_loss(target, self.loss_funct_str)

        if self.loss_funct_str == "bce" or \
                self.loss_funct_str == 'binary-cross-entropy':
            encoder = OneHotEncoder(sparse=False)
            target = encoder.fit_transform(target)
            target = torch.FloatTensor(target)

            # normalizing between 0 and 1
            min_ = output.min(dim=1, keepdim=True)[0]
            max_ = output.max(dim=1, keepdim=True)[0]
            output = (output - min_) / (max_ - min_)
            # output = sigmoid(output) # -> the same as bce-logit

        elif self.loss_funct_str == "bce-logit":
            encoder = OneHotEncoder(sparse=False)
            target = encoder.fit_transform(target)
            target = torch.FloatTensor(target)

        # calculating the loss
        loss = self.loss_funct(output, target)

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self.opt.zero_grad()

        # calculating gradients
        loss.backward()

        # updating weights from the model by calling optimizer.step()
        self.opt.step()

        return loss

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
                text, source = batch
                text = text.to(self.device)
                source = source.to(self.device)
                output = self.forward(text)
                loss = self.backward(output, source)
                _loss.append(loss.item())

            self.losses[i] = np.mean(_loss)

            print(f'Epoch: {i} loss:', np.mean(_loss))


class RNNModel(torch.nn.Module):

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
            return torch.nn.CrossEntropyLoss()  # I:(N,C) O:(N)

    @staticmethod
    def model_constructor(n_hl, units, dropout, num_features, rnn_type,
                          nonlinearity, bidirectional):
        model = None
        if rnn_type == "rnn":
            model = torch.nn.RNN(
                input_size=num_features,
                hidden_size=units,
                num_layers=n_hl,
                nonlinearity=nonlinearity,  # -> 'tanh' or 'relu'
                batch_first=True,           # -> (batch, seq, feature)
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "lstm":
            model = torch.nn.LSTM(
                input_size=num_features,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        elif rnn_type == "gru":
            model = torch.nn.GRU(
                input_size=num_features,
                hidden_size=units,
                num_layers=n_hl,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        return model

    def __init__(self, emb, n_hl: int = 1, num_features: int = 10,
                 n_classes: int = 2, dropout: float = 0.2,
                 epochs: int = 5, units: int = 25,
                 lr: float = 0.01, momentum: float = 0.9,
                 device: str = "cpu",
                 loss_funct: str = "cross-entropy",
                 random_state: int = None,
                 verbose: bool = False,
                 rnn_type: str = "rnn",
                 nonlinearity: str = 'tanh',
                 bidirectional: bool = True,
                 freeze: bool = False,
                 lr_scheduler: bool = False,
                 factor: float = 0.1,
                 patience: int = 2,
                 pool_type: str = 'cat'
                 ) -> None:
        """
        Creates a multilayer perceptron object with the specified
        parameters using the pytorch framework.
            ______________________________________________________________
            Parameters:
            n_hl: int = 1
                Number of hidden layers, defaults to 1
            num_features: int = 10
                Number of features, defaults to 10
            n_classes: int = 10
                Number of classes, defaults to 10
            dropout: float = 0.2
                Dropout value, defaults to 0.2
            epochs: int = 50
                Number of epochs to run, defaults to 20
            units: int = 25
                Number of units per hidden layer, defaults to 25
            lr: float = 0.01
                Learning rate, defaults to 0.01
            momentum: float = 0.9
                Momentum value, defaults to 0.9
            device: str = "cpu"
                Specifies how the model is run, defaults to "cpu"
            loss_funct: str = "cross-entropy"
                Loss function, defaults to "cross-entropy"
            random_state: int = None
                The seed for the random state
            verbose: bool = False
                If True: prints out progressive output, defaults to False
            ______________________________________________________________
            Returns:
            None
        """
        super().__init__()

        # seeding
        self.verbose = verbose
        self.random_state = random_state
        if self.random_state is not None:
            print(f'Setting torch random_state to {self.random_state}...') \
                if self.verbose else None
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # parameters:
        self.n_hl = n_hl                        # <- num_layers
        self.num_features = num_features        # <- input_size
        self.n_classes = n_classes
        self.dropout = dropout
        self.epochs = int(epochs)
        self.units = units                      # <- hidden_size
        self.device = device                    # -> 'cpu' or 'cuda'
        self.rnn_type = rnn_type                # -> 'rnn', 'lstm', 'gru'
        self.loss_funct_str = loss_funct
        self.loss_funct = self.lossFunct(lf_type=loss_funct)
        self.nonlinearity = nonlinearity
        self.bidirectional = bidirectional
        self.freeze = freeze
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.lr = lr
        self.momentum = momentum
        self.pool_type = pool_type

        self.losses = None
        self.val_losses = None

        self.model = self.model_constructor(
            n_hl=self.n_hl,
            units=self.units,
            dropout=self.dropout,
            num_features=self.num_features,
            rnn_type=self.rnn_type,
            nonlinearity=self.nonlinearity,
            bidirectional=self.bidirectional
        ).to(self.device)

        self.opt = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.opt,
            mode='min',
            factor=self.factor,
            patience=self.patience
        )

        # loading weights from pre-trained vocab
        self._word_embeddings = torch.nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(emb.vectors),
            freeze=self.freeze  # true: not update in the learning process
        ).to(self.device)

        # output layer
        if self.bidirectional is True:

            if self.pool_type == 'cat':
                self._linear = torch.nn.Linear(
                    in_features=self.units*2*2,
                    out_features=self.n_classes,
                    bias=True
                ).to(self.device)

            else:
                self._linear = torch.nn.Linear(
                    in_features=self.units*2,
                    out_features=self.n_classes,
                    bias=True
                ).to(self.device)

        else:
            if pool_type == 'cat':
                self._linear = torch.nn.Linear(
                    in_features=self.units * 2,
                    out_features=self.n_classes,
                    bias=True
                ).to(self.device)
                
            else:
                self._linear = torch.nn.Linear(
                    in_features=self.units,
                    out_features=self.n_classes,
                    bias=True
                ).to(self.device)

    def forward(self, x):
        """
        Performs a forward step on the model.
        ______________________________________________________________
        Parameters:
        x: torch.nn.tensor
            The input tensor to update
        ______________________________________________________________
        Returns:
        self.model: MLPModel
            The updated model
        """
        _X, lengths = x

        # getting embeddings for indices in _X
        # <- [batch:num_samples, longest_sentence] := words indices
        # -> [batch:num_samples, longest_sentence, emb_vectors:weights])
        emb = self._word_embeddings(_X)

        # -> [Tensor, Tensor, batch_size:bool, enforce_sorted:bool<-True]
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            input=emb,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        # randomly initializing weights
        if (self.bidirectional is True) and (self.rnn_type in ["rnn", "gru"]):
            # -> [num_layers*num_directions, batch, hidden_size]
            h0 = torch.zeros(
                self.n_hl*2,
                _X.size(0),
                self.units
            ).to(self.device)

            # fitting the model
            # <- [batch:n_samples, longest_sentence:seq_len, features:in_size]
            # -> [batch:n_samples, seq_len, units:hidden_size*num_directions]
            out, states = self.model(packed, h0)

        elif (self.bidirectional is False) and \
                (self.rnn_type in ["rnn", "gru"]):
            h0 = torch.zeros(
                self.n_hl,
                _X.size(0),
                self.units
            ).to(self.device)

            out, hn = self.model(packed, h0)

        elif (self.bidirectional is True) and (self.rnn_type == "lstm"):
            # [num_layers * num_directions, batch, hidden_size]
            h0 = torch.zeros(
                self.n_hl*2,
                _X.size(0),
                self.units
            ).to(self.device)

            # [num_layers * num_directions, batch, hidden_size]
            c0 = torch.zeros(
                self.n_hl*2,
                _X.size(0),
                self.units
            ).to(self.device)

            out, hn = self.model(packed, (h0, c0))

        elif (self.bidirectional is False) and (self.rnn_type == "lstm"):
            # [num_layers * num_directions, batch, hidden_size]
            h0 = torch.zeros(
                self.n_hl,
                _X.size(0),
                self.units
            ).to(self.device)

            # [num_layers * num_directions, batch, hidden_size]
            c0 = torch.zeros(
                self.n_hl,
                _X.size(0),
                self.units
            ).to(self.device)

            out, hn = self.model(packed, (h0, c0))

        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=out,
            batch_first=True
        )

        # permuting to fix/reduce the tensor's dimension from 3 to 2 later
        # <- [batch:n_samples, seq_len, units:hidden_size*num_directions]
        # -> [batch:n_samples, units:hidden_size*num_directions, seq_len]
        seq_unpacked = seq_unpacked.permute(0, 2, 1)

        # concatenating seq_len in units
        # <- [batch:n_samples, units:hidden_size*num_directions, seq_len]
        # -> [batch:n_samples, units:hidden_size*num_directions]
        if self.pool_type == 'cat':
            out_ = torch.cat(
                (seq_unpacked[:, :, 0], seq_unpacked[:, :, -1]),
                dim=-1
            )

        # <- [batch:n_samples, units:hidden_size*num_directions, seq_len]
        # -> [batch:n_samples, units:hidden_size*num_directions]
        elif self.pool_type == 'first':
            out_ = seq_unpacked[:, :, 0].squeeze(1)

        # <- [batch:n_samples, units:hidden_size*num_directions, seq_len]
        # -> [batch:n_samples, units:hidden_size*num_directions]
        elif self.pool_type == 'last':
            out_ = seq_unpacked[:, :, -1].squeeze(1)

        # <- [batch:n_samples, units:hidden_size*num_directions]
        return self._linear(out_)

    def predict_classes(self, test_data):
        """
        Makes predictions from a test tensor using the model.
        ______________________________________________________________
        Parameters:
        test_loader:
            DataLoader to make predictions on.
        ______________________________________________________________
        Returns:
        y_pred: np.array
            An array containing the predicted classes of the input
            tensors.
        """
        validation_data_copy = test_data
        X_test, y_test = next(iter(validation_data_copy)) # assuming entire input is one batch
        y_pred_dist = self.forward(X_test)  # TODO should this be detached?
        y_pred = y_pred_dist.max(dim=1)[1]  # subscript 0 return prob_dist
        return y_test[:, 0], y_pred

    def backward(self, output, target):
        """
        Performs a backpropogation step computing the loss.
        ______________________________________________________________
        Parameters:
        output:
            The output after forward with shape (batch_size, num_classes).
        target:
            The target it is optimizing towards
        ______________________________________________________________
        Returns:
        loss: float
            How close the estimate was to the gold standard.
        """
        loss = self.loss_funct(output, target[:, 0])

        # resetting the gradients from the optimizer
        # more info: https://pytorch.org/docs/stable/optim.html
        self.opt.zero_grad()

        # calculating gradients
        loss.backward()

        # updating weights from the model by calling optimizer.step()
        self.opt.step()

        return loss

    def fit(self, loader=None, verbose=False, test=None) -> None:
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

        if test is not None:
            self.val_losses = np.empty(shape=self.epochs, dtype=float)
            X_test, y_test = next(iter(test))

        for epoch in iterator:
            _loss = []
            _val_loss = []

            for n, batch in enumerate(loader):
                text, source = batch
                output = self.forward(text)
                loss = self.backward(output, source)
                _loss.append(loss.item())

                if test is not None:
                    y_valid = self.forward(X_test)
                    y_valid = y_valid.max(dim=1)[1]
                    _val_loss.append(accuracy_score(y_valid, y_test[:, 0]))

            self.losses[epoch] = np.mean(_loss)

            if self.lr_scheduler is True:
                self.scheduler.step(np.mean(_loss))

            if test is not None:
                self.val_losses[epoch] = np.mean(_val_loss)

                print(f'Epoch: {epoch} loss: {np.mean(_loss)} - '
                      f'Valid acc: {np.mean(_val_loss)}')

            else:
                print(f'Epoch: {epoch} loss: {np.mean(_loss)}')

    def best_epoch(self):
        if self.val_losses is None:
            raise ValueError("Error: Test argument was not provided while "
                             "training/fitting the model.")
        return np.argmax(self.val_losses), np.max(self.val_losses)
