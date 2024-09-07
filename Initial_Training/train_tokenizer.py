from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import VGAE
from torch_geometric.datasets import ZINC, NeuroGraphDataset, Amazon, PPI
from Models.tokenizer import DrugTokenizer
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import torch
import wandb
import os
import gc
import torch.multiprocessing as tmp


def train_epoch():

    epoch_loss = 0

    for step, graphs in enumerate(train_loader):
        z = model.encode(graphs.x.float(), edge_index=graphs.edge_index)

        model.zero_grad()
        loss = model.recon_loss(z, graphs.edge_index) + \
            (model.kl_loss()/graphs.x.size(0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        epoch_loss += loss.item()

        del graphs
        del z

    return epoch_loss/(step+1)


def test_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(test_loader):
        z = model.encode(graphs.x.float(), edge_index=graphs.edge_index)

        loss = model.recon_loss(z, graphs.edge_index)

        epoch_loss += loss.item()

        del graphs
        del z

    return epoch_loss/(step+1)


def training_loop():
    for epoch in range(EPOCHS):
        model.train(True)

        train_loss = train_epoch()
        model.eval()

        with torch.no_grad():
            test_loss = test_epoch()

            print("Epoch {epoch}: ".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            wandb.log({
                "Train Loss": train_loss,
                "Test Reconstruction Loss": test_loss,
                "Learning Rate": optimizer.param_groups[0]['lr']
            })

            if (epoch+1) % 10 == 0:
                path = f"Initial_Training/tokenizers/{task}/model{epoch+1}.pth"

                torch.save(model.encoder.state_dict(), path)

        # Update learning rate
        scheduler.step()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    torch.autograd.set_detect_anomaly(True)
    load_dotenv('.env')

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    # Get the datasets
    zinc_path = os.getenv('Zinc')
    neuro_path = os.getenv("NeuroGraph")

    task = input("Enter the domain you want to tokenize: ")
    if task == 'drugs':
        train_set = ZINC(root=zinc_path, split='train')
        test_set = ZINC(root=zinc_path, split='val')
        encoder = DrugTokenizer(in_features=train_set.num_features)
    elif task == 'neurograph':
        dataset = NeuroGraphDataset(root=neuro_path, name='HCPActivity')

    train_loader = DataLoader(train_set, **params)
    test_loader = DataLoader(test_set, **params)

    init_weights(encoder)

    model = VGAE(encoder=encoder)

    EPOCHS = 10000
    LR = 0.005
    BETAS = (0.9, 0.999)

    wandb.init(
        project="Tokenizers Training for Drugs, Neurographs, Amazon and PPI",
        config={
            "Architecture": "Graph VAE",
        }
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    training_loop()
