from torch.nn import Module
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class DrugTokenizer(Module):
    def __init__(self, in_features):
        super(DrugTokenizer, self).__init__()

        self.layer1 = GCNConv(
            in_channels=in_features, out_channels=128, normalize=True)
        self.layer2 = GCNConv(
            in_channels=128, out_channels=256, normalize=True)
        self.layer3 = GCNConv(
            in_channels=256, out_channels=512, normalize=True)

        self.conv_mu = GCNConv(in_channels=512, out_channels=256)
        self.conv_std = GCNConv(in_channels=512, out_channels=256)

    def forward(self, v, edge_index):
        x = self.layer1(v, edge_index).relu()
        x = self.layer2(x, edge_index).relu()
        x = self.layer3(x, edge_index).relu()

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
