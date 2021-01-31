# Authors: Fabio Rodrigues Pereira
# E-mails: fabior@uio.no

import torch.nn.functional
import numpy as np
import torch


class MLPModel(torch.nn.Module):

    @staticmethod
    def actFunct(af_type: str):

        if af_type == "sigmoid":
            return torch.nn.Sigmoid()

        elif af_type == "tanh":
            return torch.nn.Tanh()

        elif af_type == "softmax":
            return torch.nn.Softmax(dim=1)

    @staticmethod
    def lossFunct(lf_type: str):

        if lf_type == "cross-entropy":
            return torch.nn.CrossEntropyLoss()

    def __init__(self, n_hl: int = 1, num_features: int = 10,
                 n_classes: int = 10, dropout: float = 0.2,
                 epochs: int = 50, units: int = 128, bias: float = 0.1,
                 lr: float = 0.01, momentum: float = 0.9,
                 device: torch.device = torch.device("cpu"),
                 weights_init: str = "xavier_normal",
                 hl_actfunct: str = "sigmoid",
                 out_actfunct: str = "softmax",
                 loss_funct: str = "cross-entropy",
                 random_state: int = None) -> None:

        super().__init__()

        # seeding
        if random_state is not None:
            torch.manual_seed(random_state)

        # parameters
        self.epochs = epochs
        self.hl_actfunct = self.actFunct(af_type=hl_actfunct)
        self.out_actfunct = self.actFunct(af_type=out_actfunct)
        self.loss_funct = self.lossFunct(lf_type=loss_funct)

        if n_hl == 1:
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

            for i in range(n_hl):

                self.model.add_module(
                    name=f"HL{i+1}-Dropout",
                    module=torch.nn.Dropout(dropout)
                )

                self.model.add_module(
                    name=f"HL{i+1}-Linear",
                    module=torch.nn.Linear(units, units),
                )

                self.model.add_module(
                    name=f"HL{i+1}-ActFunction",
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
        return self.model(input_tensor)

    def predict_classes(self, input_tensor):
        y_pred = self.model(input_tensor)
        return y_pred.max(dim=1)[1]

    def backward(self, output, target):
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

    def fit(self, input_tensor, target, batch_size=2) -> None:

        self.losses = np.empty(shape=self.epochs, dtype=float)
        batch_iter = len(input_tensor) // batch_size

        for i, epoch in enumerate(range(self.epochs)):
            # output = self.forward(input_tensor)
            # loss = self.backward(output, target)
            # self.losses[i] = loss

            idxs = np.arange(len(input_tensor))
            np.random.shuffle(idxs)
            _loss, j = [], 0

            for _ in range(batch_iter):

                rand_idxs = idxs[j * batch_size:(j + 1) * batch_size]
                X, Y = input_tensor[rand_idxs], target[rand_idxs]

                output = self.forward(X)
                loss = self.backward(output, Y)
                _loss.append(loss)

                j += 1

            self.losses[i] = torch.mean(torch.stack(_loss))
