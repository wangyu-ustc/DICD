from locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
import torch
import torch.nn as nn
import numpy as np

# sys.path.append("/home/heyue/home/Discovery/")
import igraph as ig
import torch.optim as optim


class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):  # extend to MLP
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)
        # fc3: Discriminator #extend to MLP
        layers = []
        layers.append(LocallyConnected(d, d, 1, bias=bias))
        self.fc3 = nn.ModuleList(layers)

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

    def forward(self, x):  # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def forward__(self, x):  # [n, d] -> [n, d]
        W = 1 - torch.eye(x.shape[1])
        W = W.reshape(1, W.shape[0], -1).expand(x.shape[0], -1, -1)
        x = x.reshape(x.shape[0], 1, -1).expand(-1, x.shape[1], -1)
        x = x * W
        for _, fc in enumerate(self.fc3):
            if _ != 0:
                x = torch.relu(x)  # [n, d, m1]
            # else :
            # x = torch.nn.functional.dropout (x, 0.5)
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(A) - d  # (Zheng et al. 2018)
        M = torch.eye(d) + A / d  # (Yu et al. 2019)
        E = torch.matrix_power(M, d - 1)
        h = torch.trace(E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def l2_reg__(self):
        """Take 2-norm-squared of all parameters in Discriminator """
        reg = 0.
        for fc in self.fc3:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def squared_loss(output, target, I=0, J=0):
    if I == 0:
        loss = 0.5 * (((output - target) ** 2).mean(0)).sum()
    else:
        N, D = target.shape
        EXY = output.t().mm(target) / N
        EX = output.mean(0).view(-1, 1)
        EY = target.mean(0).view(-1, 1)
        DX = output.std(0).view(-1, 1)
        DY = target.std(0).view(-1, 1)
        L = ((EXY - EX.mm(EY.t())) / (DX.mm(DY.t()))).diagonal().pow(2)
        if J == 0:
            loss = L.sum()
        else:
            loss = (L - L.min().detach()).sum() / D
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    optimizer__ = optim.SGD(model.fc3.parameters(), lr=1e-2, momentum=0.9)
    X_torch = torch.from_numpy(X)
    while rho < rho_max:
        def closure__():
            optimizer__.zero_grad()
            X_hat = model(X_torch)
            xxx = X_torch - X_hat.detach()
            X_R = model.forward__(xxx)
            loss = - squared_loss(X_R, xxx, 1)
            l2_reg = 10 * 0.5 * lambda2 * model.l2_reg__()
            primal_obj = loss + l2_reg
            primal_obj.backward()
            return primal_obj

        def closure():

            global COUNT
            if COUNT % 2 == 0:
                for i in range(1):
                    LLL = closure__()
                    optimizer__.step()
            COUNT += 1

            optimizer.zero_grad()
            X_hat = model(X_torch)
            X_R = model.forward__(X_torch - X_hat)
            loss = squared_loss(X_hat, X_torch)
            lossD = squared_loss(X_R, X_torch - X_hat, 1, 1)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg + lambda3 * lossD
            primal_obj.backward()
            for p in model.fc3.parameters():
                p.grad.zero_()
            return primal_obj

        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def threshold_till_dag(B):
    if ig.Graph.Weighted_Adjacency(B.tolist()).is_dag():
        return B

    nonzero_indices = np.where(B != 0)
    weight_indices_ls = list(zip(B[nonzero_indices], nonzero_indices[0], nonzero_indices[1]))
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, i, j in sorted_weight_indices_ls:
        B[i, j] = 0
        if ig.Graph.Weighted_Adjacency(B.tolist()).is_dag():
            break

    return B


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      lambda3: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, lambda3,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    W_est = threshold_till_dag(W_est)
    return W_est


def get_result(X, B_true):
    torch.set_default_dtype(torch.float32)
    np.set_printoptions(precision=3)

    global COUNT
    COUNT = 0

    model = NotearsMLP(dims=[X.shape[1], 1], bias=True)
    W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01, lambda3=1.)  # if lambda3==0, it equals to NOTEARS

    return W_est

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str, default=None, help='Path to data files')
#     parser.add_argument('--d', type=int, default=10, help='variable')
#     opt = parser.parse_args()
#
#     import utils as ut
#     seed = random.randint(1, 10000)
#     print(seed)
#     ut.set_random_seed(seed)
#
#     torch.set_default_dtype(torch.double)
#     np.set_printoptions(precision=3)
#
#     global COUNT
#     COUNT = 0
#
#     B_true = np.load(opt.data_path, allow_pickle=True).item()['graph']
#     X = np.load(opt.data_path, allow_pickle=True).item()['data']
#
#     model = NotearsMLP(dims=[opt.d, 1], bias=True)
#     W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01, lambda3=1.)  # if lambda3==0, it equals to NOTEARS
#
#     np.savetxt('W_est.csv', W_est, delimiter=',')
#     acc = metric.evaluation(B_true, W_est != 0)
#     print(acc)


if __name__ == '__main__':
    # main()
    pass