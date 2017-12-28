import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch.autograd import Variable
from sklearn.utils import shuffle as skshuffle
from complexE import ComplexE
from utils import *
from visualize import make_dot

# Set random seed
randseed = 9999
np.random.seed(randseed)
torch.manual_seed(randseed)

# Data Loading
# Load dictionary lookups
idx2ent = np.load('data/kinship/bin/idx2ent.npy')
idx2rel = np.load('data/kinship/bin/idx2rel.npy')

n_e = len(idx2ent)
n_r = len(idx2rel)

# Load dataset
X_train = np.load('data/kinship/bin/train.npy')
X_val = np.load('data/kinship/bin/val.npy')
y_val = np.load('data/kinship/bin/y_val.npy')
# Load evaluation filters
filter_s_val = np.load('data/kinship/bin/filter_s_val.npy')
filter_o_val = np.load('data/kinship/bin/filter_o_val.npy')

X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

M_train = X_train.shape[0]
M_val = X_val.shape[0]

# Model Parameters
k = 50
embeddings_lambda = 0
gpu = True
loss_type = 'logloss'
normalize_embed = False
C = 10 # Negative Samples
lr = 0.01
lr_decay_every = 20
n_epoch = 50
mb_size = 100  
print_every = 1000
gamma = 1
average = True
hits_ks = [1,3,10]
model = ComplexE(n_e=n_e, n_r=n_r, k=k, lam=embeddings_lambda, gpu= gpu)
#model = HolE(n_e=n_e, n_r=n_r, k=k, gpu= gpu)
solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#g = make_dot(model.forward(X_train))
#g.view()
# Begin training
for epoch in range(n_epoch):
    print('Epoch-{}'.format(epoch+1))
    print('----------------')
    it = 0
    # Shuffle and chunk data into minibatches
    mb_iter = get_minibatches(X_train, mb_size, shuffle=True)

    # Anneal learning rate
    lr = lr * (0.5 ** (epoch // lr_decay_every))
    for param_group in solver.param_groups:
        param_group['lr'] = lr

    for X_mb in mb_iter:
        start = time()

        # Build batch with negative sampling
        m = X_mb.shape[0]
        if loss_type == 'rankloss':
            # C x M negative samples
            X_neg_mb = np.vstack([sample_negatives(X_mb, n_e) for _ in range(C)])
        else:
            X_neg_mb = sample_negatives(X_mb, n_e)
    
        X_train_mb = np.vstack([X_mb, X_neg_mb])
        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        if loss_type == 'logloss' or loss_type =='MSE':
            X_train_mb, y_true_mb = skshuffle(X_train_mb, y_true_mb)

        # Training step
        y = model.forward(X_train_mb)

        if loss_type == 'rankloss':
            y_pos, y_neg = y[:m], y[m:]

            loss = model.ranking_loss(
                y_pos, y_neg, margin=gamma, C=C, average=average
            )  
        elif loss_type == 'MSE':
            loss = model.mse_loss(y, y_true_mb, average=average) 
        else:
            loss = model.log_loss(y, y_true_mb, average=average)
        
        loss.backward()
        solver.step()
        solver.zero_grad()
        if normalize_embed:
            model.normalize_embeddings()

        end = time()
        # Training logs
        if it % print_every == 0:
            if loss_type == 'logloss':
                # Training auc
                pred = model.predict(X_train_mb, sigmoid=True)
                train_acc = accuracy(pred, y_true_mb)
            
                # Per class accuracy
                pos_acc = accuracy(pred[:m], y_true_mb[:m])
                neg_acc = accuracy(pred[m:], y_true_mb[m:])

                # Validation accuracy
                y_pred_val = model.forward(X_val)
                y_prob_val = F.sigmoid(y_pred_val)
            
                val_acc = accuracy(y_prob_val.data.cpu().numpy(), y_val)
                
                # Validation loss
                val_loss = model.log_loss(y_pred_val, y_val, average=average)

                print('Iter-{}; loss: {:.4f}; train_acc: {:.4f}; val_acc: {:.4f}; val_loss: {:.4f}; time per batch: {:.2f}s'
                        .format(it, loss.data[0], train_acc, val_acc, val_loss.data[0], end-start))
            else:            
                mr, mrr, hits = eval_embeddings_vertical(model, X_val_pos, n_e, hits_ks, filter_s_val, filter_o_val, n_sample=100, gpu= gpu)

                hits1, hits3, hits10 = hits
                print('Iter-{}; loss: {:.4f}; val_mr: {:.4f}; val_mrr: {:.4f}; val_hits@1: {:.4f}; val_hits@3: {:.4f}; val_hits@10: {:.4f}; time per batch: {:.2f}s'
                    .format(it, loss.data[0], mr, mrr, hits1, hits3, hits10, end-start))

        it += 1





