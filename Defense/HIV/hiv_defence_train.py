from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse, erdos_renyi_graph
from torch_geometric.utils import get_laplacian
from torch_geometric.nn import GAE
from Defense.HIV.hiv_dataset import HIVDataset
import torch
import numpy as np
from Defense.Models.isomorphism import GCNEncoder, GCNEncoderNoise
from Initial_Training.Models.Model import DrugClassificationModel as OriginalClassifier
from Defense.Models.Model import DrugClassificationModel
from Attack.contrastive_loss import InfoNCELoss
from Attack.metrics import classification_binary_metrics
from torch.utils.data import ConcatDataset
from torch import nn
from dotenv import load_dotenv
import os
import wandb
import math
import gc


def unit_vector(z):
    u1 = torch.randn(size=(1, z.size(1)))
    u2 = torch.randn(size=(1, z.size(1)))
    r1 = torch.sum(u1**2)**0.5
    r2 = torch.sum(u2**2)**0.5

    return u1/r1, u2/r2


def attack(graphs):
    _, _, z = autoencoder.encode(graphs.x, edges=graphs.edge_index)
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
                         max_num_nodes=graphs.num_nodes)

    adj_x.requires_grad = True
    adj_y1.requires_grad = True
    adj_y2.requires_grad = True

    loss = autoencoder.recon_loss(z, pos_edge_index=graphs.edge_index)+(LAMBDA*information_loss(
        adj_y, adj_y1, adj_y2, adj_x))
    loss.backward()

    adj_x.retain_grad()
    adj_y1.retain_grad()
    adj_y2.retain_grad()

    gradient_adj = adj_x.grad+adj_y1.grad+adj_y2.grad
    mean_grad = torch.mean(gradient_adj)

    adversarial_matrix = torch.where(gradient_adj > mean_grad, 1, 0)
    adversarial_sparse, _ = dense_to_sparse(adversarial_matrix)

    return adversarial_sparse, adversarial_matrix


def attack_chance():
    rand_number = torch.sin(torch.rand(1)*math.pi)

    return rand_number


def train():
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(train_loader):
        attack_prob = attack_chance()
        # Generate Random Graph
        er_graph = erdos_renyi_graph(
            num_nodes=graphs.num_nodes, edge_prob=0.2)

        graph_edges = graphs.edge_index
        if attack_prob >= 0.5:
            graph_edges, adv_matrix = attack(graphs)

            # Laplacian of each graph
            adversarial_laplacian, adv_weight = get_laplacian(graph_edges)
            er_laplacian, er_weight = get_laplacian(er_graph)

            # Eigen distribution
            _, perturbed_eigen_vecs = torch.linalg.eig(to_dense_adj(
                adversarial_laplacian, edge_attr=adv_weight, max_num_nodes=graphs.num_nodes))
            _, er_eigen_vecs = torch.linalg.eig(to_dense_adj(
                er_laplacian, edge_attr=er_weight, max_num_nodes=graphs.num_nodes))

            # Similarity between eigen distributions
            eigen_distribution_similarity = torch.mean(torch.cosine_similarity(
                perturbed_eigen_vecs.real, er_eigen_vecs.real))
            # print(eigen_distribution_similarity)

            acti_vector = torch.tensor(
                [eigen_distribution_similarity, 1-eigen_distribution_similarity, 1])
        else:
            norm_laplacian, norm_weight = get_laplacian(graph_edges)
            er_laplacian, er_weight = get_laplacian(er_graph)

            _, norm_eigen_vecs = torch.linalg.eig(
                to_dense_adj(norm_laplacian, edge_attr=norm_weight, max_num_nodes=graphs.num_nodes))
            _, er_eigen_vecs = torch.linalg.eig(to_dense_adj(
                er_laplacian, edge_attr=er_weight, max_num_nodes=graphs.num_nodes))

            eigen_distribution_similarity = torch.mean(torch.cosine_similarity(
                norm_eigen_vecs.real, er_eigen_vecs.real))
            # print(eigen_distribution_similarity)
            acti_vector = torch.tensor(
                [eigen_distribution_similarity, 1-eigen_distribution_similarity, 0])

        logits, predictions = model_attack(
            graphs.x, attack_vector=acti_vector, edges=graph_edges, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)

        model_attack.zero_grad()
        loss = loss_trainable(logits, target_col.float())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col)

        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        # print("Step Train Values")
        # print("Epoch Loss: ", epoch_loss)
        # print("Epoch Accuracy: ", epoch_acc)
        # print("Epoch Precision: ", epoch_prec)
        # print("Epoch F1: ", epoch_f1)
        # print("Epoch AUC: ", epoch_auc)

        del graphs, predictions, logits, graph_edges, er_graph, er_laplacian
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), epoch_auc/(step+1)


def test():
    epoch_loss = 0
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(test_loader):
        attack_prob = attack_chance()
        # Generate Random Graph
        er_graph = erdos_renyi_graph(
            num_nodes=graphs.num_nodes, edge_prob=0.2)

        graph_edges = graphs.edge_index
        if attack_prob >= 0.5:
            graph_edges, adv_matrix = attack(graphs)

            # Laplacian of each graph
            adversarial_laplacian, adv_weight = get_laplacian(graph_edges)
            er_laplacian, er_weight = get_laplacian(er_graph)

            # Eigen distribution
            _, perturbed_eigen_vecs = torch.linalg.eig(to_dense_adj(
                adversarial_laplacian, edge_attr=adv_weight, max_num_nodes=graphs.num_nodes))
            _, er_eigen_vecs = torch.linalg.eig(to_dense_adj(
                er_laplacian, edge_attr=er_weight, max_num_nodes=graphs.num_nodes))

            # Similarity between eigen distributions
            eigen_distribution_similarity = torch.mean(torch.cosine_similarity(
                perturbed_eigen_vecs.real, er_eigen_vecs.real))

            acti_vector = torch.tensor(
                [eigen_distribution_similarity, 1-eigen_distribution_similarity, 1])
        else:
            norm_laplacian, norm_weight = get_laplacian(graph_edges)
            er_laplacian, er_weight = get_laplacian(er_graph)

            _, norm_eigen_vecs = torch.linalg.eig(
                to_dense_adj(norm_laplacian, edge_attr=norm_weight, max_num_nodes=graphs.num_nodes))
            _, er_eigen_vecs = torch.linalg.eig(to_dense_adj(
                er_laplacian, edge_attr=er_weight, max_num_nodes=graphs.num_nodes))

            eigen_distribution_similarity = torch.mean(torch.cosine_similarity(
                norm_eigen_vecs.real, er_eigen_vecs.real))

            acti_vector = torch.tensor(
                [eigen_distribution_similarity, 1-eigen_distribution_similarity, 0])

        logits, predictions = model_attack(
            graphs.x, attack_vector=acti_vector, edges=graph_edges, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)

        loss = loss_trainable(logits, target_col.float())

        epoch_loss += loss.item()

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col)
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        # print("Step Train Values")
        # print("Epoch Loss: ", epoch_loss)
        # print("Epoch Accuracy: ", epoch_acc)
        # print("Epoch Precision: ", epoch_prec)
        # print("Epoch F1: ", epoch_f1)
        # print("Epoch AUC: ", epoch_auc)

        del graphs, predictions, logits
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def attack_loop():
    for epoch in range(NUM_EPOCHS):
        model_attack.train()
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train()

        model_attack.eval()

        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test()

        print(f"Epoch: {epoch+1}")
        print("----------Train Metrics------------")
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_acc}")
        print(f"Train Precision: {train_prec}")
        print(f"Train Recall: {train_rec}")
        print(f"Train F1: {train_f1}")
        print(f"Train AUC: {train_auc}")
        print("------------Test Metrics-------------")
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Test Precision: {test_prec}")
        print(f"Test Recall: {test_rec}")
        print(f"Test F1: {test_f1}")
        print(f"Test AUC: {test_auc}")

        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Train Precision": train_prec,
            "Train Recall": train_rec,
            "Train F1": train_f1,
            "Train AUC": train_auc,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "Test F1": test_f1,
            "Test AUC": test_auc,
        })

        if (epoch+1) % 10 == 0:
            weights_path = f"Defense/HIV/weights/fine_tuned/model_{
                epoch+1}.pth"
            torch.save(model.state_dict(), weights_path)


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
        'batch_size': 64,
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
    model = OriginalClassifier(num_labels=1, encoder=r_enc_normal)
    model.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=True), strict=False)
    model.eval()

    embedder = model.nn.gcn_enc
    autoencoder = GAE(encoder=embedder)

    # Attack robust encoder
    r_enc_attack = GCNEncoderNoise()
    model_attack = DrugClassificationModel(num_labels=1, encoder=r_enc_attack)
    model_attack.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=True), strict=False)

    LAMBDA = 4e-1
    LR = 2e-4
    BETAS = (0.9, 0.999)
    NUM_EPOCHS = 1000
    information_loss = InfoNCELoss()
    loss_trainable = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model_attack.parameters(), lr=LR, betas=BETAS)

    attack_loop()
