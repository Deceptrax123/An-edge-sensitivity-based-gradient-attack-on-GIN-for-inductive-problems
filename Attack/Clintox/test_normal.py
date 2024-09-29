import torch
from Attack.Clintox.clintox_dataset import ClintoxDataset
from torch_geometric.loader import DataLoader
from Attack.metrics import classification_binary_metrics
from Initial_Training.Models.isomorphism import GCNEncoder
from Initial_Training.Models.Model import DrugClassificationModel
import torch.multiprocessing as tmp
from dotenv import load_dotenv
import os
import gc


def evaluate():
    epoch_acc = 0
    epoch_prec = 0
    epoch_rec = 0
    epoch_auc = 0
    epoch_f1 = 0
    epoch_auc = 0

    for step, graphs in enumerate(test_loader):
        logits, predictions = model(
            graphs.x, edges=graphs.edge_index, batch=graphs.batch)
        target_col = graphs.y.view(graphs.y.size(0), 1)

        acc, f1, prec, rec, auc = classification_binary_metrics(
            predictions, target_col.int())
        epoch_acc += acc.item()
        epoch_prec += prec.item()
        epoch_f1 += f1.item()
        epoch_rec += rec.item()
        epoch_auc += auc.item()

        del graphs, predictions, logits
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
    # Trained weights are loaded here
    model.load_state_dict(torch.load(
        os.getenv("graph_weights"), weights_only=False))

    model.eval()

    test_acc, test_prec, test_rec, test_f1, test_auc = evaluate()

    print("------------Test Metrics-------------")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Precision: {test_prec}")
    print(f"Test Recall: {test_rec}")
    print(f"Test F1: {test_f1}")
    print(f"Test AUC: {test_auc}")
