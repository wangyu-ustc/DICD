import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from locally_connected import LocallyConnected


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1 = nn.Linear(d, d*dims[1], bias=bias)
        # self.fc1 = nn.ModuleList()
        # for i in range(d):
        #     self.fc1.append(nn.Linear(d-1, dims[1], bias=bias))

        # self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        # self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        # self.fc1_pos.weight.bounds = self._bounds()
        # self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        # self.fc2 = nn.ModuleList()
        # for l in range(len(dims) - 2):
        #     self.fc2.append(nn.Linear(d, dims[l+1], dims[l+1], bias=bias))

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x, dummy_A=None):  # [n, d] -> [n, d]
        if dummy_A is not None:
            x = F.linear(x, self.fc1.weight * dummy_A, self.fc1.bias)
        else:
            x = self.fc1(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def D_lin(self, x):
        # self.dummy_A = self.dummy_A.cuda()
        dummy_A = torch.nn.Parameter(torch.ones_like(self.fc1.weight)).to(x.device)
        l = squared_loss(self.forward(x, dummy_A), x)
        g = torch.autograd.grad(l, dummy_A, create_graph=True)[0]
        g[torch.where(g < 0.1)] = 0
        return torch.norm(g)


    def get_A(self):
        d = self.dims[0]
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        return A

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        A = self.get_A()
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(d).to(A.device) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(abs(self.fc1.weight))
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


class LocallyConnectedMNIST(nn.Module):
    """Local linear layer, i.e. Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers, i.e.
        in_features: m1
        out_features: m2
        bias: whether to include bias or not

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """

    def __init__(self, num_linear, input_features, output_features, bias=True):
        super(LocallyConnectedMNIST, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features

        self.weight_x = nn.Parameter(torch.Tensor(4, input_features, 512))
        self.weight_y = nn.Parameter(torch.Tensor(1, input_features, 2))

        if bias:
            self.bias_x = nn.Parameter(torch.Tensor(4, 512))
            self.bias_y = nn.Parameter(torch.Tensor(1, 2))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight_x, -bound, bound)
        nn.init.uniform_(self.weight_y, -bound, bound)
        if self.bias_x is not None:
            nn.init.uniform_(self.bias_x, -bound, bound)
            nn.init.uniform_(self.bias_y, -bound, bound)

    def forward(self, input: torch.Tensor):

        # input: (bsz, (5 * 64))
        # out_x: [n, 4, 1, 512] = [n, 4, 1, m1] @ [1, 4, m1, m2] ==> [n, 4, 512]
        out_x = torch.matmul(input[:, :4, :].unsqueeze(dim=2), self.weight_x.unsqueeze(dim=0)).squeeze(dim=2)
        # out_y: [n, 1, 1, 2] = [n, 1, 1, m1] @ [1, 1, m1, m2] ==> [n, 1, 2]
        out_y = torch.matmul(input[:, 4:, :].unsqueeze(dim=2), self.weight_y.unsqueeze(dim=0)).squeeze(dim=2)

        bsz = out_x.shape[0]

        # out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        # out = out.squeeze(dim=2)
        if self.bias_x is not None:
            # [n, d, m2] += [d, m2]
            out_x += self.bias_x
        if self.bias_y is not None:
            out_y += self.bias_y
        return torch.cat([out_x.reshape(bsz, -1), out_y.reshape(bsz,-1)], dim=1)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'num_linear={}, input_features={}, output_features={}, bias={}'.format(
            self.num_linear, self.input_features, self.output_features,
            self.bias is not None
        )



class NotearsMLPMNIST(nn.Module):
    def __init__(self, dims, d, bias=True):
        super(NotearsMLPMNIST, self).__init__()
        assert len(dims) >= 2
        # assert dims[-1] == 1
        # d = dims[0]
        self.dims = dims
        self.d = d
        # fc1: variable splitting for l1
        self.fc1 = nn.Linear(dims[0], d*dims[1], bias=bias)
        self.lengths = torch.cumsum(torch.tensor([0] + [dims[0] // d] * d), dim=0)
        # self.lengths = lengths
        # self.fc1 = nn.ModuleList()
        # for i in range(d):
        #     self.fc1.append(nn.Linear(d-1, dims[1], bias=bias))

        # self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        # self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        # self.fc1_pos.weight.bounds = self._bounds()
        # self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        # self.fc2 = nn.ModuleList()
        # for l in range(len(dims) - 2):
        #     self.fc2.append(nn.Linear(d, dims[l+1], dims[l+1], bias=bias))

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x, dummy_A=None):  # [n, d] -> [n, d]
        if dummy_A is not None:
            x = F.linear(x, self.fc1.weight * dummy_A, self.fc1.bias)
        else:
            x = self.fc1(x)  # [n, d * m1]
        x = x.view(-1, self.d, self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        # x = x.squeeze(dim=2)  # [n, d]
        return x

    def D_lin(self, x):
        # self.dummy_A = self.dummy_A.cuda()
        dummy_A = torch.nn.Parameter(torch.ones_like(self.fc1.weight)).to(x.device)
        l = squared_loss(self.forward(x.reshape(len(x), -1), dummy_A), x)
        g = torch.autograd.grad(l, dummy_A, create_graph=True)[0]
        return torch.norm(g)


    def get_A(self):
        # d = self.dims[0]
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = self.fc1.weight
        # fc1_weight = fc1_weight.view(self.dims[0], -1, self.d)  # [j, m1, i]
        fc1_weight = fc1_weight.view(self.d, -1, self.dims[0])  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]

        # A: 4 * (4 * 64) -> 4 * 4
        # A = torch.cat([torch.sum(A[:, self.lengths[idx]:self.lengths[idx+1]], dim=1, keepdim=True)
        #                for idx in range(len(self.lengths)-1)], dim=1)
        A = A.reshape(self.d, -1, self.d)
        A = torch.sum(A, dim=1)
        return A

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        A = self.get_A()
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        M = torch.eye(self.d).to(A.device) + A / self.d  # (Yu et al. 2019)
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(abs(self.fc1.weight))
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        # d = self.dims[0]
        # fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        # fc1_weight = self.fc1.weight
        # fc1_weight = fc1_weight.view(self.d, -1, self.dims[0])  # [j, m1, i]
        # A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # W = torch.sqrt(A)  # [i, j]
        # W = W.cpu().detach().numpy()  # [i, j]
        # return W
        W = torch.sqrt(self.get_A())
        return W

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    # loss = 0.5 * torch.sum((output - target) ** 2)
    return loss
