# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no

from packages.preprocessing import BOW
import torch.nn.functional
from tqdm import tqdm
import numpy as np
import torch


class MLPModel(torch.nn.Module):
    show_tqdm = False

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
            return torch.nn.CrossEntropyLoss()

    def __init__(self, n_hl: int = 1, num_features: int = 10,
                 n_classes: int = 10, dropout: float = 0.2,
                 epochs: int = 20, units: int = 25, bias: float = 0.1,
                 lr: float = 0.01, momentum: float = 0.9,
                 device: torch.device = torch.device("cpu"),
                 weights_init: str = "xavier_normal",
                 hl_actfunct: str = "tanh",
                 out_actfunct: str = "relu",
                 loss_funct: str = "cross-entropy",
                 random_state: int = None,
                 verbose: bool = False) -> None:
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
        self.loss_funct = self.lossFunct(lf_type=loss_funct)

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

    def forward(self, input_tensor):
        """
        Performs a forward step on the model.
        ______________________________________________________________
        Parameters:
        input_tensor: torch.nn.tensor
            The input tensor to update
        ______________________________________________________________
        Returns:
        self.model: MLPModel
            The updated model
        """

        return self.model(input_tensor)

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
        y_pred = self.model(input_tensor)
        return y_pred.max(dim=1)[1]

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
        # calculating the loss
        loss_function = self.loss_funct
        loss = loss_function(output, target)

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

            for n, batch in enumerate(loader, 0):
                text, source = batch
                text = text.to(self.device)
                source = source.to(self.device)
                output = self.forward(text)
                loss = self.backward(output, source)
                _loss.append(loss.item())

            self.losses[i] = np.mean(_loss)

            if self.verbose is True:
                print(f'Epoch: {i} loss:', np.mean(_loss),
                      end="\r") if i % 10 == 0 else None


class MLPModel_wl(MLPModel):

    def __init__(self, n_hl: int = 1, num_features: int = 10,
                 n_classes: int = 10, dropout: float = 0.2,
                 epochs: int = 50, units: int = 128, bias: float = 0.1,
                 lr: float = 0.01, momentum: float = 0.9,
                 device: torch.device = torch.device("cpu"),
                 weights_init: str = "xavier_normal",
                 hl_actfunct: str = "sigmoid",
                 out_actfunct: str = "softmax",
                 loss_funct: str = "cross-entropy",
                 random_state: int = None,
                 verbose: bool = True,
                 loader: BOW=None):
        """
        Creates a multilayer perceptron object with the 
        specified parameters using the pytorch framework 
        without dataloader.
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
            Specifies how the weights are initialized, 
            defaults to "xavier_normal"
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

        super().__init__(n_hl, num_features, n_classes, dropout, epochs,
                         units, bias, lr, momentum, device, weights_init,
                         hl_actfunct, out_actfunct, loss_funct, random_state,
                         verbose)

        self.input_tensor, self.targets = None, None

        self.loader = loader

    def fit(self, loader=None, verbose=False, batch_size=32):
        """
        Fits the model to the training data using the models 
        initialized values. Runs for the models number of epochs.
        ______________________________________________________________
        Parameters:
        loader: torch.nn.tensor = None
            X, Y tensor tuple replacing dataloader, defaults to None
        verbose: bool = False
            If True: prints out progressive output, defaults to False
        batch_size: int = 32
            Size of the batches for the data, defaults to 32
        ______________________________________________________________
        Returns:
        None
        """

        input_tensor = loader[0].to(self.device)
        targets = loader[1].to(self.device)

        del loader

        self.losses = np.empty(shape=self.epochs, dtype=float)
        batch_iter = len(input_tensor) // batch_size

        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)

        for i, epoch in enumerate(iterator):

            idxs = np.arange(len(input_tensor))
            np.random.shuffle(idxs)
            _loss, j = [], 0

            for _ in range(batch_iter):

                rand_idxs = idxs[j * batch_size:(j + 1) * batch_size]
                X, Y = input_tensor[rand_idxs], targets[rand_idxs]

                output = self.forward(X)
                loss = self.backward(output, Y)
                _loss.append(loss)

                j += 1

            self.losses[i] = torch.mean(torch.stack(_loss))

            if self.verbose is True:
                print(f'Epoch: {i} loss:',
                      torch.mean(torch.stack(_loss)),
                      end="\r") if i % 2 == 0 else None
