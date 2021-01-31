# Authors: Fabio Rodrigues Pereira
# E-mails: fabior@uio.no

from sklearn.model_selection import StratifiedKFold
from Oblig1.packages.ann_models import MLPModel
from Oblig1.packages.preprocessing import BOW
from torch.utils.data import DataLoader
import torch


# Common parameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 10

tensors = BOW(
    bow_type="counter",
    vocab_size=10,  # 0 means all words
    device=DEVICE,
    verbose=True,
    random_state=SEED
)

tensors.fit_transform(
    url='Oblig1/data/sample.tsv'
)

idx = torch.arange(0, 10, 2)

# kf = StratifiedKFold(n_splits=8, random_state=3829, shuffle=True)

# loader = DataLoader(
#     (tensors.input_tensor, tensors.target),
#     batch_size=2
# )

model = MLPModel(
    n_hl=3,
    num_features=len(tensors.features_names),
    n_classes=len(tensors.classes_names),
    dropout=0.2,
    epochs=50,
    units=5,
    bias=0.1,
    lr=0.01,
    momentum=0.9,
    device=DEVICE,
    weights_init="xavier_normal",
    hl_actfunct="sigmoid",
    out_actfunct="softmax",
    loss_funct="cross-entropy",
    random_state=SEED
)

model.fit(
    input_tensor=tensors.input_tensor,
    target=tensors.target,
    batch_size=2
)
