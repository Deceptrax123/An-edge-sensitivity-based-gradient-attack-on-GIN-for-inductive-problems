from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import GAE
from Defense.HIV.hiv_dataset import HIVDataset
import torch
from Defense.Models.isomorphism import GCNEncoder, GCNEncoderNoise
from Defense.Models.Model import DrugClassificationModel
from Attack.metrics import classification_binary_metrics
from torch.utils.data import ConcatDataset
from torch import nn
from dotenv import load_dotenv
import os
import wandb
import gc


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    train_folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5', 'Fold6']
    test_folds = ['Fold7', 'Fold8']

    train_set1 = HIVDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/Fold1"+"/data/", start=0)
    train_set2 = HIVDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/Fold2/"
                            + "/data/", start=5141)
    train_set3 = HIVDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/Fold3/"
                            + "/data/", start=10282)
    train_set4 = HIVDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/Fold4/"
                            + "/data/", start=15423)
    train_set5 = HIVDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/Fold5/"
                            + "/data/", start=20564)
    train_set6 = HIVDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/Fold6/"
                            + "/data/", start=25705)

    test_set1 = HIVDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/Fold7/"
                           + "/data/", start=30846)
    test_set2 = HIVDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=35987)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])
    test_set = ConcatDataset([test_set1, test_set2])

    params = {
        'batch_size': 256,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    wandb.init(
        project="Graph Classification Adversarial. Defence",
        config={
            "Method": "Random Matrix Theories",
        })

    r_enc_normal = GCNEncoder()
    model = DrugClassificationModel(num_labels=1, encoder=r_enc_normal)
    model.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=True), strict=False)

    embedder = model.nn.gcn_enc
    autoencoder = GAE(encoder=embedder)
