from abc import ABC
import torch_geometric as pyg
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn
import torch_geometric.data as tgd
from torch_geometric.data import InMemoryDataset
import torch_geometric.utils as utils
from torch.autograd.function import Function
from scipy.io import loadmat
import numpy as np


class MultiTaskGNN(nn.Module, ABC):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, class_num, dropout):
        super(MultiTaskGNN, self).__init__()
        self.gc1 = gnn.GCNConv(input_feat_dim, hidden_dim1)
        self.gc2 = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.gc3 = gnn.GCNConv(hidden_dim1, hidden_dim2)
        self.gc4 = gnn.GCNConv(hidden_dim1, hidden_dim1)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim1, class_num),
        )
        self.drop = dropout
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.binact = StraightBin

    def encode(self, features, edge_index, edge_weight=None):
        # print(features, edge_index, edge_weight)
        hidden1 = self.gc1(features, edge_index, edge_weight)
        hidden1 = self.binact.apply(hidden1)
        hidden2 = self.gc4(hidden1, edge_index, edge_weight)
        return self.gc2(hidden1, edge_index, edge_weight), self.gc3(hidden1, edge_index, edge_weight), \
               self.seq(hidden2)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data: tgd.Data):
        mu, logvar, classifer = self.encode(data.x, data.edge_index, data.edge_attr)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar, classifer


class InnerProductDecoder(nn.Module, ABC):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class KYXLDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.pre_filter = pre_transform
        self.transform = transform
        super(KYXLDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def deal_with_mat(self):
        """
        将.mat 转化为 [Data]
        :return: DataList: [Data]
        """
        print("dealing with mat...")
        m = loadmat(self.raw_paths[0])
        A = utils.from_scipy_sparse_matrix(m['network'])
        att = torch.from_numpy(m['attributes'].todense().astype(np.float32))
        y = torch.from_numpy(m['labels'].reshape(-1)).to(torch.long)
        # 如果y最小值不是0，则认为idx从1开始
        if int(torch.min(y)) != 0:
            y -= 1
        dt = tgd.Data(x=att, edge_index=A[0], edge_weight=A[1].to(torch.float32), y=y)
        # print(dt)
        return [dt]

    @property
    def raw_file_names(self):
        return [self.name + ".mat"]

    @property
    def processed_file_names(self):
        return [self.name]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # data_list = [...]
        data_list = self.deal_with_mat()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class StraightBin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input) -> torch.tensor:
        # ctx.save_for_backward(torch.abs(input))
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        # result, = ctx.saved_tensors
        # return grad_output * result
        return grad_output


class BinActive(torch.autograd.Function):
    """
    Binarize the input activations and calculate the mean across channel dimension.
    """

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinLinear(nn.Module):  # change the name of BinConv2d
    def __init__(self, input_num, output_num):
        super(BinLinear, self).__init__()
        self.layer_type = 'BinLinear'
        self.alpha = 0.0
        self.Linear = nn.Linear(input_num, output_num)

    def forward(self, x):
        x = self.Linear(x)
        x = BinActive.apply(x)
        return x


# TODO 完善BinarizeLinear
class BinarizeLinear(nn.Linear):

    def __init__(self, input_num, output_num, trans=None):
        super(BinarizeLinear, self).__init__(input_num, output_num)
        if trans is None:
            self.trans = StraightBin
        else:
            self.trans = trans

    def forward(self, input):
        w = self.trans.apply(self.weight)
        out = F.linear(input, w, self.bias)
        return out


class BinarizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # save input for backward pass

        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = input.clone()
        grad_input[torch.abs(input) <= 1.] = 1.
        grad_input[torch.abs(input) > 1.] = 0.
        grad_input = grad_input * grad_output

        return grad_input


class Binarization(nn.Module):

    def __init__(self, _min=-1, _max=1, stochastic=False):
        super(Binarization, self).__init__()
        self.stochastic = stochastic
        self.min = _min
        self.max = _max

    def forward(self, input):
        return 0.5 * (BinarizeFunction.apply(input) * (self.max - self.min) + self.min + self.max)


class BinarizedLinear(nn.Linear):

    def __init__(self, min_weight=-1, max_weight=1, *kargs, **kwargs):
        super(BinarizedLinear, self).__init__(*kargs, **kwargs)
        self.binarization = Binarization(_min=min_weight, _max=max_weight)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.noise_on = False
        self.noise_std = 0.2
        self.noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data) * self.noise_std)

    def forward(self, input):
        device_num = self.weight.get_device()
        device = torch.device("cuda:%d" % device_num)
        self.noise = self.noise.to(device)
        self.weight.data = nn.functional.hardtanh_(self.weight.data)

        if self.noise_on:
            out = nn.functional.linear(input, self.binarization(self.weight) + self.noise, bias=self.bias)
        else:
            out = nn.functional.linear(input, self.binarization(self.weight),
                                       bias=self.bias)  # linear layer with binarized weights
        return out

    def quantize_accumulative_weigths(self):
        self.weight.data = self.binarization(self.weight.data)
        return

    def set_noise_std(self, std=0.2):
        self.noise_std = std
        self.noise = torch.normal(mean=0.0, std=torch.ones_like(self.weight.data) * self.noise_std)
        return

    def set_noise(self, noise_on=True):
        self.noise_on = noise_on
        return

    def calc_prop_grad(self, prob_rate=0):
        with torch.no_grad():
            tmp = torch.abs(self.weight.grad.data).add(1e-20).clone()
            self.weight.grad.data.div_(tmp)  # norm of grad values

            # tmp = F.tanh(prob_rate*tmp).clone()
            # tmp.mul_(prob_rate).pow_(2).mul_(-1).exp_().mul_(-1).add_(1) # 1 - exp(-x^2)
            # tmp.mul_(prob_rate).mul_(-1).exp_().mul_(-1).add_(1) # 1 - exp(-x)
            tmp.mul_(prob_rate).pow_(2).add_(1).reciprocal_().mul_(-1.).add_(1.)  # 1 - 1/(1+x^2)

            # print(tmp)

            tmp = torch.bernoulli(tmp).clone()
            self.weight.grad.data.mul_(tmp)
            # self.weight.grad.mul_(0)
            del tmp
            # print(self.weight)
        return

    def add_bit_error(self, bit_error_rate=0):
        probs = torch.ones_like(self.weight.data).mul_(1 - bit_error_rate)  # switching probabilities
        switching_tensor = torch.bernoulli(probs).mul(2.).add(-1.)
        self.weight.data.mul_(switching_tensor)
        return
