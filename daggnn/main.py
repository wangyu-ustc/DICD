import os
import sys
import math
import time
import torch
import networkx as nx
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from .utils import encode_onehot, get_tril_offdiag_indices, \
    get_triu_offdiag_indices, Variable, nll_gaussian, \
    kl_gaussian_sem, A_connect_loss, A_positive_loss, count_accuracy
from .modules import MLPEncoder, SEMEncoder, \
    MLPDecoder, SEMDecoder

from collections import namedtuple
from .utils import matrix_poly

def _h_A(A):
    m = A.shape[1]
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def get_result(X, B_true):
    from .config import opt
    if len(X.shape) == 3:
        opt['x_dims'] = X.shape[-1]
        opt['z_dims'] = X.shape[-1]

    opt = namedtuple('opt', field_names=opt.keys())(**opt)

    batch_size = 1000

    feat_train = torch.FloatTensor(X)
    feat_valid = torch.FloatTensor(X)
    feat_test = torch.FloatTensor(X)

    if len(X.shape) < 3:
        feat_train.unsqueeze_(-1)
        feat_valid.unsqueeze_(-1)
        feat_test.unsqueeze_(-1)

    # reconstruct itself
    train_data = TensorDataset(feat_train, feat_train)
    valid_data = TensorDataset(feat_valid, feat_train)
    test_data = TensorDataset(feat_test, feat_train)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    ground_truth_G = nx.DiGraph(B_true)

    # ===================================
    # load modules
    # ===================================
    # Generate off-diagonal interaction graph
    off_diag = np.ones([opt.data_variable_size, opt.data_variable_size]) - np.eye(opt.data_variable_size)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    # num_nodes = opt.data_variable_size
    # adj_A = np.zeros((num_nodes, num_nodes))
    num_nodes = X.shape[1]
    adj_A = np.zeros((num_nodes, num_nodes))

    if opt.encoder == 'mlp':
        encoder = MLPEncoder(opt.data_variable_size * opt.x_dims, opt.x_dims, opt.encoder_hidden,
                             int(opt.z_dims), adj_A,
                             batch_size=opt.batch_size,
                             do_prob=opt.encoder_dropout, factor=opt.factor).double()

    elif opt.encoder == 'sem':
        encoder = SEMEncoder(opt.data_variable_size * opt.x_dims, opt.encoder_hidden,
                             int(opt.z_dims), adj_A,
                             batch_size=opt.batch_size,
                             do_prob=opt.encoder_dropout, factor=opt.factor).double()

    if opt.decoder == 'mlp':
        decoder = MLPDecoder(opt.data_variable_size * opt.x_dims,
                             opt.z_dims, opt.x_dims, encoder,
                             data_variable_size=opt.data_variable_size,
                             batch_size=opt.batch_size,
                             n_hid=opt.decoder_hidden,
                             do_prob=opt.decoder_dropout).double()

    elif opt.decoder == 'sem':
        decoder = SEMDecoder(opt.data_variable_size * opt.x_dims,
                             opt.z_dims, 2, encoder,
                             data_variable_size=opt.data_variable_size,
                             batch_size=opt.batch_size,
                             n_hid=opt.decoder_hidden,
                             do_prob=opt.decoder_dropout).double()

    if opt.load_folder:
        encoder_file = os.path.join(opt.load_folder, 'encoder.pt')
        encoder.load_state_dict(torch.load(encoder_file))
        decoder_file = os.path.join(opt.load_folder, 'decoder.pt')
        decoder.load_state_dict(torch.load(decoder_file))

        opt.save_folder = False

    # ===================================
    # set up training parameters
    # ===================================
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr)
    elif opt.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                                lr=opt.lr)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                              lr=opt.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay,
                                    gamma=opt.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(opt.data_variable_size)
    tril_indices = get_tril_offdiag_indices(opt.data_variable_size)

    if opt.prior:
        prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
        print("Using prior")
        print(prior)
        log_prior = torch.DoubleTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)

        if opt.cuda:
            log_prior = log_prior.cuda()

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)

    prox_plus = torch.nn.Threshold(0., 0.)


    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1


    def update_optimizer(optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr


    # ===================================
    # training:
    # ===================================

    def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer, print_loss):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []
        shd_trian = []

        encoder.train()
        decoder.train()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, opt.lr, c_A)

        for batch_idx, (data, relations) in enumerate(train_loader):

            if opt.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data).double(), Variable(relations).double()

            # reshape data
            relations = relations.unsqueeze(2)

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec,
                                                                                              rel_send)  # logits is of size: [num_sims, z_dims]
            edges = logits

            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, opt.data_variable_size * opt.x_dims, rel_rec,
                                                        rel_send, origin_A, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.

            # reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = opt.tau_A * torch.sum(torch.abs(one_adj_A))

            # other loss term
            if opt.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, opt.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if opt.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # compute h(A)
            h_A = _h_A(origin_A)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(
                origin_A * origin_A) + sparse_loss  # +  0.01 * torch.sum(variance * variance)

            loss.backward()
            loss = optimizer.step()
            scheduler.step()

            myA.data = stau(myA.data, opt.tau_A * lr)

            if torch.sum(origin_A != origin_A):
                print('nan error\n')

            # compute metrics
            graph = origin_A.data.clone().numpy()
            graph[np.abs(graph) < opt.graph_threshold] = 0

            fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            shd_trian.append(shd)

        if print_loss:
            print(h_A.item())
            nll_val = []
            acc_val = []
            kl_val = []
            mse_val = []

            print('Epoch: {:04d}'.format(epoch),
                  'nll_train: {:.10f}'.format(np.mean(nll_train)),
                  'kl_train: {:.10f}'.format(np.mean(kl_train)),
                  'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
                  'mse_train: {:.10f}'.format(np.mean(mse_train)),
                  'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                  'time: {:.4f}s'.format(time.time() - t))

            if opt.save_folder and np.mean(nll_val) < best_val_loss:
                torch.save(encoder.state_dict(), encoder_file)
                torch.save(decoder.state_dict(), decoder_file)
                print('Best model so far, saving...')
                print('Epoch: {:04d}'.format(epoch),
                      'nll_train: {:.10f}'.format(np.mean(nll_train)),
                      'kl_train: {:.10f}'.format(np.mean(kl_train)),
                      'ELBO_loss: {:.10f}'.format(np.mean(kl_train) + np.mean(nll_train)),
                      'mse_train: {:.10f}'.format(np.mean(mse_train)),
                      'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
                      'time: {:.4f}s'.format(time.time() - t))
                # log.flush()
                sys.stdout.flush()

        if 'graph' not in vars():
            print('error on assign')

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(
            mse_train), graph, origin_A


    # ===================================
    # main
    # ===================================

    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # optimizer step on hyparameters
    c_A = opt.c_A
    lambda_A = opt.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = opt.h_tol
    k_max_iter = int(opt.k_max_iter)
    h_A_old = np.inf

    try:
        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(opt.epochs):
                    print_loss = True if epoch == opt.epochs - 1 else False
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(step_k, best_ELBO_loss, ground_truth_G,
                                                                           lambda_A, c_A, optimizer, print_loss)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                print("Optimization Finished!")
                print("Best Epoch: {:04d}".format(best_epoch))
                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # update parameters
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new)
                if h_A_new.item() > 0.25 * h_A_old:
                    c_A *= 10
                else:
                    break

                # update parameters
                # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break

        if opt.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch))
            # log.flush()
            sys.stdout.flush()
        # test()
        # print(best_ELBO_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        # print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        #
        # print(best_NLL_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
        # print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        #
        # print(best_MSE_graph)
        # print(nx.to_numpy_array(ground_truth_G))
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        # print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        graph = origin_A.data.clone().numpy()
        # graph[np.abs(graph) < 0.1] = 0
        # # print(graph)
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
        # print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        #
        # graph[np.abs(graph) < 0.2] = 0
        # # print(graph)
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
        # print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        graph[np.abs(graph) < 0.3] = 0
        # print(graph)
        # fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
        # print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
        # return {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "nnz": nnz}


    except KeyboardInterrupt:
        # print the best anway
        print(best_ELBO_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_ELBO_graph))
        print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        print(best_NLL_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_NLL_graph))
        print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        print(best_MSE_graph)
        print(nx.to_numpy_array(ground_truth_G))
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(best_MSE_graph))
        print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

        graph[np.abs(graph) < 0.3] = 0
        # print(graph)
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))
        print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

    # f = open('trueG', 'w')
    # matG = np.matrix(nx.to_numpy_array(ground_truth_G))
    # for line in matG:
    #     np.savetxt(f, line, fmt='%.5f')
    # f.close()

    # f1 = open('predG', 'w')
    # matG1 = np.matrix(origin_A.data.clone().numpy())
    # for line in matG1:
    #     np.savetxt(f1, line, fmt='%.5f')
    return graph
