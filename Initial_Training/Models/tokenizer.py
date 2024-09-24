from torch.nn import Module
from torch_geometric.nn import GCNConv,ChebConv
import torch.nn.functional as F



class DrugTokenizer(Module):
    def __init__(self, in_features):
        super(DrugTokenizer, self).__init__()

        self.gcn1 = ChebConv(in_channels=in_features,
                             out_channels=128, normalization='sym', K=3)
        self.gcn2 = ChebConv(
            in_channels=128, out_channels=256, normalization='sym', K=3)
        self.gcn3 = ChebConv(
            in_channels=256, out_channels=512, normalization='sym', K=3)

    def forward(self, v, edge_index):
        x = self.gcn1(v, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        x = self.gcn3(x, edge_index).relu()

        return x

class NeuroGenderTokenizer(Module):
    def __init__(self, in_features):
        super(NeuroGenderTokenizer, self).__init__()

        self.layer1 = GCNConv(
            in_channels=in_features, out_channels=1024, normalize=True)
        self.layer2 = GCNConv(
            in_channels=1024, out_channels=2000, normalize=True)

        self.conv_mu = GCNConv(in_channels=2000,
                               out_channels=1024)
        self.conv_std = GCNConv(in_channels=2000,
                                out_channels=1024)

    def forward(self, v, edge_index):
        x = self.layer1(v, edge_index).relu()
        x = self.layer2(x, edge_index).relu()

        return self.conv_mu(x, edge_index), self.conv_std(x, edge_index)


class NeuroGraphTokenizer(Module):
    def __init__(self, in_features):
        super(NeuroGraphTokenizer, self).__init__()

        self.layer1 = GCNConv(
            in_channels=in_features, out_channels=in_features*2, normalize=True)
        self.layer2 = GCNConv(
            in_channels=in_features*2, out_channels=in_features*4, normalize=True)
        self.layer3 = GCNConv(
            in_channels=in_features*4, out_channels=in_features*8, normalize=True)

        self.conv_mu = GCNConv(in_channels=in_features*8,
                               out_channels=in_features*4)
        self.conv_std = GCNConv(in_channels=in_features*8,
                                out_channels=in_features*4)

    def forward(self, v, edge_index):
        x = self.layer1(v, edge_index).relu()
        x = self.layer2(x, edge_index).relu()
        x = self.layer3(x, edge_index).relu()

        return self.conv_mu(x, edge_index), self.conv_std(x, edge_index)
