import argparse

from model import NotearsMLP, squared_loss
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", default=100, type=int)
parser.add_argument("--inner_max_iter", default=5000, type=int)
parser.add_argument("--h_tol", default=1e-8, type=float)
parser.add_argument("--rho_max", default=1e16, type=float)
parser.add_argument("--w_threshold", default=0.3, type=float)
parser.add_argument("--c_A", default=1, type=float)
parser.add_argument("--lambda1", default=0.01, type=float)
parser.add_argument("--lambda2", default=0.01, type=float)
parser.add_argument("--lambda_D", default=0.1, type=float)
parser.add_argument("--beta", default=1000, type=int)
parser.add_argument("--group_num", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--method", default="NoCurl")
parser.add_argument("--n_attributes", default=5, type=int)
parser.add_argument("--noise_variance_type", default=1, type=int)
parser.add_argument("--loss_type", default='l2')
parser.add_argument("--non_linear_type", default=1, type=int)
parser.add_argument("--batch_size", default=1000, type=int)
parser.add_argument("--sem_type", default="mlp")
parser.add_argument("--graph_type", default='SF')
parser.add_argument("--s0", default=40, type=int)
parser.add_argument("--d", default=10, type=int)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--scheduler", default=2, type=int)
parser.add_argument("--data_dir", default='./data/')
parser.add_argument("--case", default=None, type=str)
args = parser.parse_args()


def evaluate_optimality(model, X, graph_name):
    print("==" * 20)
    print(f"For the {graph_name} Graph, the optimality loss is:")
    for i, x in enumerate(X):
        print(f"Group [{i}]", model.D_lin(x).item())
    print("==" * 20)

def optimization_step(model, data_loaders, lambda1, lambda2, lambda_D, rho, alpha, h, rho_max, count, step_k, t_total, iters):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    # optimizer = LBFGSBScipy(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    length = []
    data_iter = {}

    for i in range(len(data_loaders)):
        length.append(len(data_loaders[i]))
        data_iter[i] = iter(data_loaders[i])

    while rho < rho_max:
        for i in range(iters):
            count += 1
            if args.scheduler == 1:
                lam_D = min(count / (t_total // 2), 1) * lambda_D * step_k ** 1.6
            elif args.scheduler == 2:
                if count < t_total // 3:
                    lam_D = min(count / (t_total // 3), 1) * lambda_D
                elif count < t_total // 3 and count > t_total // 3 * 2:
                    lam_D = lambda_D
                else:
                    lam_D = max(min((t_total - count) / (t_total // 3), 1), 0) * lambda_D
            else:
                raise NotImplementedError
            optimizer.zero_grad()

            primal_obj = torch.tensor(0.).to(args.device)

            for i in range(len(data_loaders)):
                try:
                    tmp_x = data_iter[i].next()[0]
                except:
                    data_iter[i] = iter(data_loaders[i])
                    tmp_x = data_iter[i].next()[0]

                tmp_x = tmp_x.to(args.device)

                primal_obj += (squared_loss(model(tmp_x), tmp_x) + model.D_lin(tmp_x) * lam_D) / len(data_loaders)

            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj += penalty + l2_reg + l1_reg
            primal_obj.backward()
            optimizer.step()

        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break

    alpha += rho * h_new

    print(f"count = {count}, lambda D = {lam_D}")

    return rho, alpha, h_new, count





def main(args, **kwargs):
    from utils import count_accuracy

    for argname, argval in vars(args).items():
        print(f'{argname.replace("_", " ").capitalize()}: {argval}')

    np.random.seed(args.seed)

    W_true = load_nonlinear_graph(args)

    B_true = (W_true != 0).astype(int)

    X = load_nonlinear_data(args, **kwargs)

    print("Shape:", X.shape)

    if args.method in ['DICD', 'NOTEARS']:
        from torch.utils.data import DataLoader
        from torch.utils.data.dataset import TensorDataset
        model = NotearsMLP(dims=[args.d, 10, 1], bias=True).to(args.device)
        X = [torch.tensor(x, dtype=torch.float32) for x in X]

        if args.method == 'NOTEARS':
            X = [torch.cat(X)]
            args.lambda_D = 0

        data_loaders = []
        for group in X:
            data_loaders.append(
                DataLoader(TensorDataset(torch.FloatTensor(group)), batch_size=args.batch_size // len(X)))

        rho, alpha, h = 1.0, 0.0, np.inf

        t_total = 20 * args.inner_max_iter
        count = 0

        for step_k in range(args.max_iter):

            print(f"Iter {step_k}, h = {h}， rho = {rho}")
            model.to('cpu')
            for i, x in enumerate(X):
                print(f"Group [{i}]: loss = {squared_loss(model(x), x)}")
            model.to(args.device)

            sys.stdout.flush()

            rho, alpha, h, count = optimization_step(model, data_loaders, args.lambda1, args.lambda2, args.lambda_D,
                                                    rho, alpha, h, args.rho_max, count, step_k, t_total, args.inner_max_iter)

            if h <= args.h_tol or rho >= args.rho_max:
                break

        W_est = model.fc1_to_adj()
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

    if not is_dag(W_est):
        print("!!! Result is not DAG, Now cut the edge until it's DAG")
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(W_est)
        for step, t in enumerate(thresholds):
            print("Edges/thresh", W_est.sum(), t)
            to_keep = torch.Tensor(W_est > t + 1e-8).numpy()
            W_est = W_est * to_keep

            if is_dag(W_est):
                break

    W_est = (W_est != 0).astype(int)

    acc = count_accuracy(B_true, W_est != 0)
    print(acc)
    return acc


if __name__ == '__main__':

    main(args)



