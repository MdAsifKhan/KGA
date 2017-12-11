
#Some libraries to import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch.autograd import Variable
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from complexE import *
from utils import *



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

X_val_pos = X_val[y_val.ravel() == 1, :]  # Take only positive samples

M_train = X_train.shape[0]
M_val = X_val.shape[0]

# Model Parameters
k = 50
embeddings_lambda = 0
model = ComplexE(n_e=n_e, n_r=n_r, k=k, lam=embeddings_lambda, gpu= False)

normalize_embed = True
C = 10 # Negative Samples
# Optimizer Initialization
nepoch = 20
lr = 0.1
lr_decay_every = 20
solver = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
n_epoch = nepoch
mb_size = 100  # 2x with negative sampling
print_every = 9999
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
        # C x M negative samples
        X_neg_mb = sample_negatives(X_mb, n_e)
        X_train_mb = np.vstack([X_mb, X_neg_mb])

        y_true_mb = np.vstack([np.ones([m, 1]), np.zeros([m, 1])])

        # Training step
        y = model.forward(X_train_mb)
        loss = model.log_loss(y, y_true_mb, average=True)
        
        loss.backward()
        solver.step()
        solver.zero_grad()
        if normalize_embed:
            model.normalize_embeddings()

        end = time()
        # Training logs
        if it % print_every == 0:
            # Training auc
            pred = model.predict(X_train_mb, sigmoid=True)
            train_acc = auc(pred, y_true_mb)
            
            # Per class accuracy
            # pos_acc = accuracy(pred[:m], y_true_mb[:m])
            # neg_acc = accuracy(pred[m:], y_true_mb[m:])

            # Validation auc
            y_pred_val = model.forward(X_val)
            y_prob_val = F.sigmoid(y_pred_val)
            
            val_acc = auc(y_prob_val.data.numpy(), y_val)
            # Validation loss
            val_loss = model.log_loss(y_pred_val, y_val, True)

            print('Iter-{}; loss: {:.4f}; train_auc: {:.4f}; val_auc: {:.4f}; val_loss: {:.4f}; time per batch: {:.2f}s'
                    .format(it, loss.data[0], train_acc, val_acc, val_loss.data[0], end-start))

        it += 1





