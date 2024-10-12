
import torch
from torch.autograd import Variable
from torch.nn import Module, Linear, ReLU, BatchNorm1d, Dropout1d
from torch_geometric.nn import GINConv, ChebConv
import torch.nn.functional as F


class GaussianNoise(Module):
    def __init__(self, shape, std=1):
        self.noise = Variable(torch.zeros(shape))
        self.std = std

    def forward(self):
        self.noise.data.normal_(0, std=self.std)

        return self.noise


class GCNEncoderNoise(Module):
    def __init__(self, noise_shape1, noise_shape2, noise_shape3, std):
        super(GCNEncoderNoise, self).__init__()
        self.gcn1 = ChebConv(in_channels=86, out_channels=128, K=3)
        self.gcn2 = ChebConv(in_channels=128, out_channels=256, K=3)
        self.gcn3 = ChebConv(in_channels=256, out_channels=512, K=3)

        self.gaussian1 = GaussianNoise(shape=noise_shape1, std=std)
        self.gaussian2 = GaussianNoise(shape=noise_shape2, std=std)
        self.gaussian3 = GaussianNoise(shape=noise_shape3, std=std)

    def forward(self, v, edges):
        B1 = self.gaussian1()
        B2 = self.gaussian2()
        B3 = self.gaussian3()

        x1 = F.relu(self.gcn1(v, edge_index=edges))
        x1 = torch.add(x1, B1)

        x2 = F.relu(self.gcn2(x1, edge_index=edges))
        x2 = torch.add(x2, B2)

        x3 = F.relu(self.gcn3(x2, edge_index=edges))
        x3 = torch.add(x3, B3)

        return x1, x2, x3


class GCNEncoder(Module):
    def __init__(self):
        super(GCNEncoder, self).__init__()
        self.gcn1 = ChebConv(in_channels=86, out_channels=128, K=3)
        self.gcn2 = ChebConv(in_channels=128, out_channels=256, K=3)
        self.gcn3 = ChebConv(in_channels=256, out_channels=512, K=3)

    def forward(self, v, edges):
        x1 = F.relu(self.gcn1(v, edge_index=edges))
        x2 = F.relu(self.gcn2(x1, edge_index=edges))
        x3 = F.relu(self.gcn3(x2, edge_index=edges))

        return x1, x2, x3


class GraphIsomorphismNetwork(Module):
    def __init__(self, encoder):
        super(GraphIsomorphismNetwork, self).__init__()
        self.gin1 = GINConv(nn=PhiLinearMapping(86, 128), train_eps=True)
        self.gin2 = GINConv(nn=PhiLinearMapping(128, 256), train_eps=True)
        self.gin3 = GINConv(nn=PhiLinearMapping(256, 512), train_eps=True)
        self.gcn_enc = encoder

    def forward(self, v, edges):
        v1, v2, v3 = self.gcn_enc.forward(v, edges)
        x1 = self.gin1(v, edge_index=edges)
        xv1 = torch.add(x1, v1)

        x2 = self.gin2(xv1, edge_index=edges)
        xv2 = torch.add(x2, v2)

        x3 = self.gin3(xv2, edge_index=edges)
        xv3 = torch.add(x3, v3)

        return xv3


class PhiLinearMapping(Module):
    def __init__(self, in_channels, out_channels):
        super(PhiLinearMapping, self).__init__()
        self.linear = Linear(in_features=in_channels,
                             out_features=out_channels)
        self.relu = ReLU()
        self.bn = BatchNorm1d(num_features=out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
