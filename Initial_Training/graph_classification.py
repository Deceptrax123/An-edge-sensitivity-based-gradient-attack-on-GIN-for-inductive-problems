import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Amazon, MoleculeNet, Reddit, NeuroGraphDataset, TUDataset
from torch_geometric.graphgym import init_weights
from Initial_Training.Models.Model import GraphClassificationModel
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
    elif task == 'tox21':
        dataset = MoleculeNet(root=tox21_path, name='Tox21')
        category = 'multilabel'
    elif task == 'neuro':
        dataset = NeuroGraphDataset(root=neuro_path, name='HCPActivity')

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

    model = GraphClassificationModel(num_features=dataset[0].num_node_features, num_labels=dataset[0].num_classes,
                                     category=category)
    init_weights(model)  # Weight Initialisation
    LR = 0.001
    NUM_EPOCHS = 10000
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True)

    train_steps = (len(train_set)+params['batch_size'])//params['batch_size']
    val_steps = (len(val_set)+params['batch_size'])//params['batch_size']
    test_steps = (len(test_set)+params['batch_size'])//params['batch_size']
