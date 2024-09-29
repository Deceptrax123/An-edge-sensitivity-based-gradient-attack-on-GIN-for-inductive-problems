from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.nn import GAE
from Attack.Clintox.clintox_dataset import ClintoxDataset
import torch
from Initial_Training.Models.isomorphism import GCNEncoder
from Initial_Training.Models.Model import DrugClassificationModel
from Attack.metrics import classification_binary_metrics
from Attack.contrastive_loss import InfoNCELoss
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
from dotenv import load_dotenv
from torch import cpu
import gc
import os


def unit_vector(z):
    u1 = torch.randn(size=(1, z.size(1)))
    u2 = torch.randn(size=(1, z.size(1)))
    r1 = torch.sum(u1**2)**0.5
    r2 = torch.sum(u2**2)**0.5

    return u1/r1, u2/r2


def perform_attack():
    # Attack the model(edges) and features
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(test_loader):
        _, _, z = embedder.encode(graphs.x, edges=graphs.edge_index)
        zcap1, zcap2 = unit_vector(z)
        epsilon1, epsilon2 = torch.normal(
            0, torch.std(z)), torch.normal(0, torch.std(z))

        z1 = torch.add(z, torch.mul(epsilon1, zcap1))
        z2 = torch.add(z, torch.mul(epsilon2, zcap2))

        adj_y = torch.sigmoid(torch.matmul(z, z.t()))
        adj_y1 = torch.sigmoid(torch.matmul(z1, z1.t()))
        adj_y2 = torch.sigmoid(torch.matmul(z2, z2.t()))

        adj_y1 = torch.where(adj_y1 > 0.5, 1.0, 0.0)
        adj_y2 = torch.where(adj_y2 > 0.5, 1.0, 0.0)

        adj_x = to_dense_adj(edge_index=graphs.edge_index,
                             max_num_nodes=graphs.x.size(0))

        adj_x.requires_grad = True
        adj_y1.requires_grad = True
        adj_y2.requires_grad = True

        loss = embedder.recon_loss(z, pos_edge_index=graphs.edge_index)+(LAMBDA*information_loss(
            adj_y, adj_y1, adj_y2, adj_x))
        loss.backward()

        adj_x.retain_grad()
        adj_y1.retain_grad()
        adj_y2.retain_grad()

        gradient_adj = adj_x.grad+adj_y1.grad+adj_y2.grad
        mean_grad = torch.mean(gradient_adj)

        adversarial_matrix = torch.where(gradient_adj > mean_grad, 1, 0)
        adversarial_sparse, _ = dense_to_sparse(adversarial_matrix)

        # Pass into classifier
        logits, predictions = model(
            graphs.x, edges=adversarial_sparse, batch=graphs.batch)
        target_col = graphs.y.view(logits.size(0), 1)

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col.int())
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del predictions, logits, graphs, adversarial_matrix, adversarial_sparse, gradient_adj
        gc.collect()

    return epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


if __name__ == '__main__':

    load_dotenv('.env')

    params = {
        'batch_size': 32,
        'shuffle': True
    }

    test_set = ClintoxDataset(fold_key='Fold8', root=os.getenv(
        "graph_files")+"/Fold8"+"/data/", start=1295, stop=1479)

    test_loader = DataLoader(test_set, **params)

    r_enc = GCNEncoder()

    model = DrugClassificationModel(num_labels=1, encoder=r_enc)
    model.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=True))
    model.eval()

    embedder = GAE(encoder=model.nn.gcn_enc)

    LAMBDA = 0.4
    information_loss = InfoNCELoss(reduction=True)

    test_acc, test_prec, test_rec, test_f1, test_auc = perform_attack()

    print("------------Peformance post Attack-------------")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_prec}")
    print(f"Test Recall: {test_rec}")
    print(f"Test F1: {test_f1}")
    print(f"Test AUC: {test_auc}")
