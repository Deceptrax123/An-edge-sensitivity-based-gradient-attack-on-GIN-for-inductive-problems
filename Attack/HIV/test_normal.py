from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.utils import to_dense_adj, dropout_adj, dense_to_sparse
from torch_geometric.nn import GAE
from Attack.HIV.hiv_dataset import HIVDataset
import torch
from Initial_Training.Models.isomorphism import GCNEncoder
from Initial_Training.Models.Model import DrugClassificationModel
from Attack.metrics import classification_binary_metrics
from Attack.contrastive_loss import InfoNCELoss
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
import gc
import os


def normal_performance():

    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(test_loader):
        _, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col.float())
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

    return epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


if __name__ == '__main__':

    load_dotenv('.env')

    params = {
        'batch_size': 2566,
        'shuffle': True
    }

    test_set = HIVDataset(fold_key='Fold8', root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=35987)

    test_loader = DataLoader(test_set, **params)

    r_enc = GCNEncoder()
    r_enc.eval()

    model = DrugClassificationModel(num_labels=1, encoder=r_enc)
    model.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=True), strict=False)
    model.eval()

    test_acc, test_prec, test_rec, test_f1, test_auc = normal_performance()

    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_prec}")
    print(f"Test Recall: {test_rec}")
    print(f"Test F1: {test_f1}")
    print(f"Test AUC: {test_auc}")
