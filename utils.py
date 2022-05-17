import os
import torch
import numpy as np
import random
import networkx as nx
import torch.nn as nn
from scipy.special import expit as sigmoid
from igraph import *

def _loss(X, W, loss_type='l2', reduction='mean'):
    """Evaluate value and gradient of loss."""
    M = X @ W
    if loss_type == 'l2':
        R = X - M
        if reduction == 'mean':
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
        elif reduction == 'none':
            loss = (R ** 2).sum(axis=-1)
        else:
            print(f"reduction {reduction} not recognized")
            return NotImplementedError
    elif loss_type == 'logistic':
        loss = 1.0 / X.shape[0] * (torch.logaddexp(0, M) - X * M).sum()
        # G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
    elif loss_type == 'poisson':
        S = torch.exp(M)
        loss = 1.0 / X.shape[0] * (S - X * M).sum()
        # G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
    else:
        raise ValueError('unknown loss type')
    return loss

def matrix_poly(matrix, d):
    x = torch.eye(d).double().to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)

def _h_A(A):
    m = len(A)
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def _D_lin(W, X):
    dummy_w = nn.Parameter(torch.ones(W.shape)).to(X.device)
    l = _loss(X, W*dummy_w)
    # h_A = _h_A(W*dummy_w)
    # l += args.lambda1 * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(W_est * W_est)
    g = torch.autograd.grad(l, dummy_w, create_graph=True)[0]
    # return (g**2).sum()
    # g[torch.where(g < 0.1)] = 0
    return torch.norm(g)

def load_linear_graph(args):
    # print("number of points:", n)
    # print("noises:", noises)
    # print("graph type:", graph_type)
    # num_string = "_".join([str(i) for i in n])
    # if not os.path.exists(
    #         f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}.csv'):
    #     print("generating new graph....")
    #     B_true = simulate_dag(args.d, args.s0, args.graph_type)
    #     W_true = simulate_parameter(B_true)
    #     np.savetxt(f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}.csv',
    #                W_true,
    #                delimiter=',')
    # else:
    #     print("loading existing graph...")
    #     W_true = np.loadtxt(
    #         f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}.csv',
    #         delimiter=',')
    # return W_true
    W_true = np.loadtxt(f"{args.data_dir}/linear_{args.d}_{args.graph_type}{args.s0 // args.d}/W_true.csv", delimiter=',')
    return W_true

def load_nonlinear_graph(args):
    W_true = np.loadtxt(f"{args.data_dir}/nonlinear_{args.d}_{args.graph_type}{args.s0 // args.d}_{args.sem_type}/W_true.csv", delimiter=',')
    return W_true


def load_linear_data(args, ratio=None, group_num=None):
    if ratio is not None:
        X = np.load(f"{args.data_dir}/linear_{args.d}_{args.graph_type}{args.s0 // args.d}/data_{ratio:.1f}_{args.seed}.npy",
                    allow_pickle=True)
    elif group_num is not None:
        X = np.load(f"{args.data_dir}/linear_{args.d}_{args.graph_type}{args.s0 // args.d}/data_{group_num}_{args.seed}.npy")
    else:
        X = np.load(f"{args.data_dir}/linear_{args.d}_{args.graph_type}{args.s0 // args.d}/data_{args.seed}.npy")
    return X


def load_nonlinear_data(args, ratio=None, group_num=None):
    if ratio is not None:
        X = np.load(f"{args.data_dir}/nonlinear_{args.d}_{args.graph_type}{args.s0 // args.d}_{args.sem_type}/data_{ratio:.1f}_{args.seed}.npy",
                    allow_pickle=True)
    elif group_num is not None:
        X = np.load(f"{args.data_dir}/nonlinear_{args.d}_{args.graph_type}{args.s0 // args.d}_{args.sem_type}/data_{group_num}_{args.seed}.npy")
    else:
        X = np.load(f"{args.data_dir}/nonlinear_{args.d}_{args.graph_type}{args.s0//args.d}_{args.sem_type}/data_{args.seed}.npy", allow_pickle=True)

    return X
    # num_string = "_".join([str(i) for i in n])
    # if not os.path.exists(
    #         f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}_nonlinear_{sem_type}.npy'):
    #     print("generating new datasets...")
    #     X = []
    #     selected_idxes = np.random.choice(np.arange(args.d), size=int(args.d * 0.3))
    #
    #     for i in range(args.group_num):
    #         noises_list = np.ones(args.d)
    #         noises_list[selected_idxes] = noises[i]
    #         X.append(simulate_nonlinear_sem(B_true, n[i], sem_type, noise_scale=noises_list))
    #     X = np.array(X)
    #     np.save(
    #         f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}_nonlinear_{sem_type}.npy',
    #         X)
    # else:
    #     print("loading existing datasets...")
    #     X = np.load(
    #         f'data/W_true_{num_string}_{args.d}_{args.s0}_{args.graph_type}_{args.sem_type}_{args.seed}_nonlinear_{sem_type}.npy',
    #         allow_pickle=True)
    # return X




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def is_dag(W):
    G = Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_sem(G: nx.DiGraph,
                 n: int, x_dims: int,
                 sem_type: str,
                 linear_type: str,
                 noise_scale: list) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss,linear-exp,linear-gumbel}
        noise_scale: scale parameter of noise distribution in linear SEM

    Returns:
        X: [n,d] sample matrix
    """
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    X = np.zeros([n, d, x_dims])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        if linear_type == 'linear':
            eta = X[:, parents, 0].dot(W[parents, j])
        elif linear_type == 'nonlinear_1':
            eta = np.cos(X[:, parents, 0] + 1).dot(W[parents, j])
        elif linear_type == 'nonlinear_2':
            eta = (X[:, parents, 0]+0.5).dot(W[parents, j])
        else:
            raise ValueError('unknown linear data type')

        if sem_type == 'linear-gauss':
            if linear_type == 'linear':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale[j], size=n)
            elif linear_type == 'nonlinear_1':
                X[:, j, 0] = eta + np.random.normal(scale=noise_scale[j], size=n)
            elif linear_type == 'nonlinear_2':
                X[:, j, 0] = 2.*np.sin(eta) + eta + np.random.normal(scale=noise_scale[j], size=n)
        elif sem_type == 'linear-exp':
            X[:, j, 0] = eta + np.random.exponential(scale=noise_scale[j], size=n)
        elif sem_type == 'linear-gumbel':
            X[:, j, 0] = eta + np.random.gumbel(scale=noise_scale[j], size=n)
        else:
            raise ValueError('unknown sem type')
    if x_dims > 1 :
        for i in range(x_dims-1):
            X[:, :, i+1] = np.random.normal(scale=noise_scale, size=1)*X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=noise_scale, size=1) * X[:, :, 0] + np.random.normal(scale=noise_scale, size=1) + np.random.normal(scale=noise_scale, size=(n, d))
    return X



def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, np.linalg.sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        # z = np.array([np.random.normal(scale=s) for s in scale])
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale is None else np.ones(d)
    X = np.zeros([n, d])
    G = Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}