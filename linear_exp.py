import argparse

from utils import *
from utils import _loss, _h_A, _D_lin

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", default=100, type=int)
parser.add_argument("--h_tol", default=1e-8, type=float)
parser.add_argument("--rho_max", default=1e16, type=float)
parser.add_argument("--w_threshold", default=0.3, type=float)
parser.add_argument("--c_A", default=1, type=float)
parser.add_argument("--lambda1", default=0.01, type=float)
parser.add_argument("--lambda2", default=0.01, type=float)
parser.add_argument("--lambda_D", default=1, type=float)
parser.add_argument("--beta", default=1000, type=int)
parser.add_argument("--group_num", default=5, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--method", default="NOTEARS")
parser.add_argument("--n_attributes", default=5, type=int)
parser.add_argument("--noise_variance_type", default=1, type=int)
parser.add_argument("--loss_type", default='l2')
parser.add_argument("--non_linear_type", default=1, type=int)
parser.add_argument("--batch_size", default=1000, type=int)
parser.add_argument("--linear_type", default="linear")
parser.add_argument("--sem_type", default="linear-gauss")
parser.add_argument("--graph_type", default='ER')
parser.add_argument("--s0", default=40, type=int)
parser.add_argument("--d", default=10, type=int)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--data_dir", default='./data/')
parser.add_argument('--scheduler', default=1, type=int)
parser.add_argument("--inner_max_iter", default=5000, type=int)
parser.add_argument("--case", default=None, type=str)
args = parser.parse_args()

def optimization_step(W_est, data_loaders, lambda1, lambda2, lambda_D, rho, alpha, h, rho_max, count, step_k, t_total, inner_max_iter):

    h_new = None
    optimizer = torch.optim.Adam([W_est], lr=0.001)
    length = []
    data_iter = {}

    for i in range(len(data_loaders)):
        length.append(len(data_loaders[i]))
        data_iter[i] = iter(data_loaders[i])

    while rho < rho_max:
        for i in range(inner_max_iter):
            count += 1
            if args.scheduler == 1:
                lam_D = min(max((count - 4000), 0)/ (t_total // 2), 1) * lambda_D * step_k ** 1.6
            else:
                if count < t_total // 3:
                    lam_D = min(count / (t_total // 3), 1) * lambda_D
                elif count < t_total // 3 and count > t_total // 3 * 2:
                    lam_D = lambda_D
                else:
                    lam_D = min((t_total - count) / (t_total // 3), 1) * lambda_D
            optimizer.zero_grad()

            loss = torch.tensor(0.).to(args.device)
            for i in range(len(data_loaders)):
                try:
                    tmp_x = data_iter[i].next()[0]
                except:
                    data_iter[i] = iter(data_loaders[i])
                    tmp_x = data_iter[i].next()[0]

                loss += _loss(tmp_x, W_est) + _D_lin(W_est, tmp_x) * lam_D

            h_val = _h_A(W_est)
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * torch.norm(W_est, 2)
            l1_reg = lambda1 * torch.norm(W_est, 1)
            loss += penalty + l2_reg + l1_reg
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_new = _h_A(W_est).item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break


    print("lambda_D:", lam_D)

    alpha += rho * h_new
    return rho, alpha, h_new, count

def evaluate_optimality(W, X, graph_name):
    print("==" * 20)
    print(f"For the {graph_name} Graph, the optimality loss is:")
    for i, x in enumerate(X):
        print(f"Group [{i}]", _D_lin(W, x).item())

    print("==" * 20)

def main(args, **kwargs):
    for argname, argval in vars(args).items():
        print(f'{argname.replace("_", " ").capitalize()}: {argval}')

    set_random_seed(args.seed)

    W_true = load_linear_graph(args)

    B_true = (W_true != 0).astype(int)

    X = load_linear_data(args, **kwargs)

    print("Number of Groups:")
    print([len(x) for x in X])

    if args.method in ['DICD', 'NOTEARS']:

        from torch.utils.data import DataLoader
        from torch.utils.data.dataset import TensorDataset
        loss_type = 'l2'
        d = X[0].shape[1]
        n = [len(group) for group in X]
        W_est = nn.Parameter(torch.zeros(d, d))
        if loss_type == 'l2':
            for i in range(len(X)):
                X[i] = X[i] - np.mean(X[i], axis=0, keepdims=True)
        W_est = W_est.to(args.device)

        X = [torch.tensor(x, dtype=torch.float32) for x in X]

        evaluate_optimality(torch.tensor(W_true, dtype=torch.float32), X, 'True')

        if args.method == 'NOTEARS':
            X = [torch.cat(X)]
            args.lambda_D = 0
            n = [np.sum(n)]

        data_loaders = []
        for group in X:
            data_loaders.append(
                DataLoader(TensorDataset(torch.FloatTensor(group)), batch_size=args.batch_size // len(X)))

        rho, alpha, h = 1.0, 0.0, np.inf

        t_total = 20 * args.inner_max_iter * 10
        count = 0

        for step_k in range(args.max_iter):
            print(f"Iter {step_k}, h = {h}， rho = {rho}")
            for group_id, group in enumerate(X):
                print(f"group [{group_id}], loss={_loss(X[group_id].to(args.device), W_est, loss_type).item()}")

            if step_k >= 1:
                evaluate_optimality(W_est, X, 'Estimated')

            sys.stdout.flush()

            rho, alpha, h, count = optimization_step(W_est, data_loaders, args.lambda1, args.lambda2, args.lambda_D,
                                                    rho, alpha, h, args.rho_max, count, step_k, t_total, args.inner_max_iter)

            if h <= args.h_tol or rho >= args.rho_max:
                break

        W_est = W_est.detach().cpu().numpy()
        W_est[np.abs(W_est) < args.w_threshold] = 0

    elif args.method == 'GranDAG':
        from gran_dag.main import get_result
        from torch.utils.data import DataLoader
        from torch.utils.data.dataset import TensorDataset

        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        X = torch.cat(X)
        data_loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=args.batch_size)
        W_est = get_result(data_loader, B_true, X, X.shape[1], args.graph_type, args.s0//args.d)

    elif args.method == 'DAG-GNN':
        from daggnn.main import get_result
        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        X = torch.cat(X)
        W_est = get_result(X.numpy(), B_true)

    elif args.method == 'DARING':
        from daring.main import get_result
        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        X = torch.cat(X)
        W_est = get_result(X.numpy(), B_true)

    elif args.method == 'NoCurl':
        from nocurl.main import get_result
        X = [torch.tensor(x, dtype=torch.float32) for x in X]
        X = torch.cat(X)
        W_est = get_result(X.numpy(), B_true)

    else:
        raise ValueError(f"method {args.method} not recognized")

    W_est = (W_est != 0).astype(int)
    assert is_dag(W_est)

    acc = count_accuracy(B_true, W_est != 0)
    print(acc)
    for name, value in acc.items():
        print(value, end=' & ')
    print()

    try:
        os.makedirs(f"./estimated_graphs", exist_ok=True)
        np.savetxt(f"./estimated_graphs/{args.linear_type}_{args.graph_type}{args.s0//args.d}_{args.method}_{args.seed}_{args.sem_type}_W_est.csv", W_est, delimiter=',')
    except:
        pass

    return acc

if __name__ == '__main__':
    main(args)
