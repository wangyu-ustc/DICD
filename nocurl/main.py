
''''
Main function for traininng DAG NoCurl

'''


from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple

import nocurl.BPR


def get_args():
    parser = argparse.ArgumentParser()

    # -----------data parameters ------
    # configurations
    parser.add_argument('--data_type', type=str, default= 'synthetic',
                        choices=['synthetic', 'nonlinear1', 'nonlinear2', 'nonlinear3'],
                        help='choosing which experiment to do.')
    parser.add_argument('--data_sample_size', type=int, default=1000,
                        help='the number of samples of data')
    parser.add_argument('--data_variable_size', type=int, default=10,
                        help='the number of variables in synthetic generated data')
    parser.add_argument('--graph_type', type=str, default='erdos-renyi',
                        choices=['barabasi-albert','erdos-renyi'],
                        help='the type of DAG graph by generation method')
    parser.add_argument('--graph_degree', type=int, default=3,
                        help='the number of degree in generated DAG graph')
    parser.add_argument('--graph_sem_type', type=str, default='linear-gauss',
                        choices=['linear-gauss','linear-gumbel'],
                        help='the structure equation model (SEM) parameter type')
    parser.add_argument('--x_dims', type=int, default=1, # data dimension
                        help='The number of input dimensions: default 1.')

    # -----------training hyperparameters
    parser.add_argument('--repeat', type=int, default= 100,
                        help='the number of times to run experiments to get mean/std')

    parser.add_argument('--methods', type=str, default='nocurl',
                        choices=['notear',                   # notear
                                 'nocurl',             # dag no curl
                                 'CAM', 'GES', 'MMPC', 'FGS'                           # baselines
                                 ] ,
                        help='which method to test') # BPR_all = notear

    parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                        help = 'threshold for learned adjacency matrix binarization')
    parser.add_argument('--lambda1',  type = float, default= 1000., #corresponding to lambda2
                        help='coefficient for the first penalty parameter for h(A) in step 1.')
    parser.add_argument('--lambda2',  type = float, default= 1000., #corresponding to lambda2
                        help='coefficient for the second penalty parameter for h(A) in step 1.')
    parser.add_argument('--rho_A_max', type=float, default=1e+16,  # corresponding to rho, needs to  be >> lambda
                        help='coefficient for notears.')
    parser.add_argument('--h_tol', type=float, default = 1e-8,
                        help='the tolerance of error of h(A) to zero')
    parser.add_argument('--train_epochs', type=int, default= 1e4,
                        help='Max Number of iteration in notears.')
    parser.add_argument('--generate_data', type=int, default=1,
                        help='generate new data or use old data')
    parser.add_argument('--file_name', type = str, default = 'test_')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')

    # -----------parsing
    args = parser.parse_args()


    return args

def get_result(X, B):

    args = {
        "repeat": 100,
        "data_sample_size": X.shape[0],
        "degree": X.shape[1],
        'graph_threshold': 0.3,
        'lambda1': 10.0,
        'lambda2': 10.0,
        'rho_A_max': 1e+16,
        'h_tol': 1e-08,
        'train_epochs': 10000.0,
        'generate_data': 1,
        'file_name': 'test_',
        'save_folder': 'logs',
        'load_folder': ''
    }
    args = namedtuple('opt', field_names=args.keys())(**args)

    return main(args, X, B)



def main(args, X, G):

    # FOR TO BE 2D, so no nonlinear for now
    if X.ndim > 2: # args.graph_linear_type !='linear':
        X = X[:, :, 0]

    bpr = nocurl.BPR.BPR(args)

    A, h, alpha, rho = bpr.fit(X)

    return A
