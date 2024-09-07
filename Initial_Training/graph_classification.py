import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet, NeuroGraphDataset
from torch_geometric.graphgym import init_weights
from Models.Model import GraphClassificationModel
from metrics import classification_binary_metrics, classification_multilabel_metrics
from sklearn.model_selection import train_test_split
import torch.multiprocessing as tmp
from dotenv import load_dotenv
import wandb
import os
import gc

# Datasets Used:
# 1. HIV
# 2. Tox21
# 3. Neuro Graph


def train_epoch():
    epoch_loss = epoch_acc = epoch_prec = epoch_rec = epoch_auc = epoch_f1 = 0

    for step, graphs in enumerate(train_loader):

        model.zero_grad()
        logits, predictions = model(
            graphs.x.float(), edges=graphs.edge_index, batch=graphs.batch
        )

        target_vector = graphs.y.view(graphs.y.size(0), 1)
        loss = loss_function(logits, target_vector.float())

        loss.backward()
        # perform gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        # Losses
        epoch_loss += loss.item()

        if dataset.num_classes == 2:
            acc, f1, prec, rec, auc = classification_binary_metrics(
                predictions, target_vector.int())
        else:
            acc, f1, prec, rec, auc = classification_multilabel_metrics(
                predictions, target_vector.int())

        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def val_epoch():
    epoch_loss = epoch_acc = epoch_prec = epoch_rec = epoch_auc = epoch_f1 = 0

    for step, graphs in enumerate(val_loader):
        logits, predictions = model(
            graphs.x.float(), edges=graphs.edge_index, batch=graphs.batch)

        target_vector = graphs.y.view(graphs.y.size(0), 1)
        loss = loss_function(logits, target_vector.float())

        epoch_loss += loss.item()

        if dataset.num_classes == 2:
            acc, f1, prec, rec, auc = classification_binary_metrics(
                predictions, target_vector)
        else:
            acc, f1, prec, rec, auc = classification_multilabel_metrics(
                predictions, target_vector.int())

        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def test_epoch():
    epoch_loss = epoch_acc = epoch_prec = epoch_rec = epoch_auc = epoch_f1 = 0

    for step, graphs in enumerate(train_loader):
        logits, predictions = model(
            graphs.x.float(), edges=graphs.edge_index, batch=graphs.batch)

        target_vector = graphs.y.view(graphs.y.size(0), 1)
        loss = loss_function(logits, target_vector.float())

        epoch_loss += loss.item()

        if dataset.num_classes == 2:
            acc, f1, prec, rec, auc = classification_binary_metrics(
                predictions, target_vector)
        else:
            acc, f1, prec, rec, auc = classification_multilabel_metrics(
                predictions, target_vector.int())

        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
        gc.collect()

    return epoch_loss/(step+1), epoch_acc/(step+1), epoch_prec/(step+1), epoch_rec/(step+1), epoch_f1/(step+1), \
        epoch_auc/(step+1)


def training_loop():
    for epoch in range(NUM_EPOCHS):
        model.train(True)

        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_epoch()
        model.eval()

        with torch.no_grad():
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = val_epoch()

            print(f"Epoch: {epoch+1}")
            print("----------Train Metrics------------")
            print(f"Train Loss: {train_loss}")
            print(f"Train Accuracy: {train_acc}")
            print(f"Train Precision: {train_prec}")
            print(f"Train Recall: {train_rec}")
            print(f"Train F1: {train_f1}")
            print(f"Train AUC: {train_auc}")
            print("------------Validation Metrics-------------")
            print(f"Test Loss: {val_loss}")
            print(f"Test Accuracy: {val_acc}")
            print(f"Test Precision: {val_prec}")
            print(f"Test Recall: {val_rec}")
            print(f"Test F1: {val_f1}")
            print(f"Test AUC: {val_auc}")

            wandb.log({
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
                "Train Precision": train_prec,
                "Train Recall": train_rec,
                "Train F1": train_f1,
                "Train AUC": train_auc,
                "Test Loss": val_loss,
                "Test Accuracy": val_acc,
                "Test Precision": val_prec,
                "Test Recall": val_rec,
                "Test F1": val_f1,
                "Test AUC": val_auc
            })

            if (epoch+1) % 10 == 0:
                test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_epoch()
                print("------------Test Metrics-------------")
                print(f"Test Loss: {test_loss}")
                print(f"Test Accuracy: {test_acc}")
                print(f"Test Precision: {test_prec}")
                print(f"Test Recall: {test_rec}")
                print(f"Test F1: {test_f1}")
                print(f"Test AUC: {test_auc}")

                wandb.log({
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    "Test Precision": test_prec,
                    "Test Recall": test_rec,
                    "Test F1": test_f1,
                    "Test AUC": test_auc
                })

                # Save weights
                weights_path = f"Initial_Training/{
                    task}/run_1/model_{epoch+1}.pth"
                torch.save(model.state_dict(), weights_path)


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    params = {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    }

    # Get the datasets
    hiv_path = os.getenv('HIV')
    tox21_path = os.getenv("Tox21")
    neuro_path = os.getenv("NeuroGraph")

    task = input("Enter the dataset you want to work with: ")
    if task == 'hiv':
        dataset = MoleculeNet(root=hiv_path, name='HIV')
        category = 'binary'
        num_labels = 1
    elif task == 'tox21':
        dataset = MoleculeNet(root=tox21_path, name='Tox21')
        category = 'multilabel'
        num_labels = dataset.num_classes
    elif task == 'neuro':
        dataset = NeuroGraphDataset(root=neuro_path, name='HCPActivity')
        catgeory = 'binary'
        num_labels = 1

    # Split as Train, Validation and Test Folds
    dataset = dataset.shuffle()
    train_ratio = 0.75
    val_ratio = 0.15
    test_ratio = 0.10
    train_set, test_set = train_test_split(
        dataset, test_size=1-train_ratio)
    val_set, test_set = train_test_split(
        test_set, test_size=test_ratio/(test_ratio+val_ratio))

    # Loaders
    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)
    val_loader = DataLoader(val_set, **params)

    # Logger
    wandb.init(
        project="Graph Classification Raw Training",
        config={
            "Method": "Graph Convolution",
        })

    model = GraphClassificationModel(num_features=dataset[0].num_node_features, num_labels=num_labels,
                                     category=category)
    init_weights(model)  # Weight Initialisation
    LR = 0.001
    NUM_EPOCHS = 10000
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)

    if dataset.num_classes == 2 or category == 'multilabel':
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    train_steps = (len(train_set)+params['batch_size'])//params['batch_size']
    val_steps = (len(val_set)+params['batch_size'])//params['batch_size']
    test_steps = (len(test_set)+params['batch_size'])//params['batch_size']

    training_loop()
