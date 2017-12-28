import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from inspect import getmembers, isfunction
import scipy.stats as st
import pdb
import torch

# Some Utilities
def eval_embeddings_vertical(model, X_test, n_e, k, filter_h=None, filter_t=None, n_sample=100, gpu= True):
    M = X_test.shape[0]

    if n_sample is not None:
        sample_idxs = np.random.randint(M, size=n_sample)
    else:
        sample_idxs = np.arange(M)

    ranks_h = np.zeros(sample_idxs.shape[0], dtype=int)
    ranks_t = np.zeros(sample_idxs.shape[0], dtype=int)

    for i, idx in enumerate(sample_idxs):
        x = X_test[idx]
        h, t = int(x[0]), int(x[2])

        x = x.reshape(1, -1)
        y_h, y_t = model.predict_all(x)
        # Filtered setting
        y_h, y_t = y_h.data, y_t.data
        y_h = y_h.view(y_h.numel())
        y_t = y_t.view(y_h.numel())
        true_h, true_t = y_h[h], y_t[t]

        if filter_h is not None:
            if gpu:  
                y_h[torch.LongTensor(filter_h[idx]).cuda()] = np.inf
            else:
                y_h[torch.LongTensor(filter_h[idx])] = np.inf                
        if filter_t is not None:
            if gpu:  
                y_t[torch.LongTensor(filter_t[idx]).cuda()] = np.inf
            else:
                y_t[torch.LongTensor(filter_t[idx])] = np.inf 
        y_h[h] = true_h
        y_t[t] = true_t

        # Do ranking
        _, ranking_h = torch.sort(y_h)
        _, ranking_t = torch.sort(y_t)

        ranking_h = ranking_h.cpu().numpy()
        ranking_t = ranking_t.cpu().numpy()

        ranks_h[i] = np.where(ranking_h == h)[0][0] + 1
        ranks_t[i] = np.where(ranking_t == t)[0][0] + 1

    # Mean rank
    mr = (np.mean(ranks_h) + np.mean(ranks_t)) / 2

    # Mean reciprocal rank
    mrr = (np.mean(1/ranks_h) + np.mean(1/ranks_t)) / 2

    # Hits@k
    if isinstance(k, list):
        hitsk = [(np.mean(ranks_h <= r) + np.mean(ranks_t <= r)) / 2 for r in k]
    else:
        hitsk = (np.mean(ranks_h <= k) + np.mean(ranks_t <= k)) / 2

    return mr, mrr, hitsk


def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    minibatches = []
    X_shuff = np.copy(X)

    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]

def sample_negatives(X, n_e):
    """
    Perform negative sampling by corrupting head or tail of each triplets in
    dataset.

    Params:
    -------
    X: int matrix of M x 3, where M is the (mini)batch size
        First column contains index of head entities.
        Second column contains index of relationships.
        Third column contains index of tail entities.

    n_e: int
        Number of entities in dataset.

    Returns:
    --------
    X_corr: int matrix of M x 3, where M is the (mini)batch size
        Similar to input param X, but at each column, either first or third col
        is subtituted with random entity.
    """
    M = X.shape[0]

    corr = np.random.randint(n_e, size=M)
    e_idxs = np.random.choice([0, 2], size=M)

    X_corr = np.copy(X)
    X_corr[np.arange(M), e_idxs] = corr

    return X_corr

def accuracy(y_pred, y_true, thresh=0.5, reverse=False):
    """
    Compute accuracy score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.

    thresh: float, default: 0.5
        Classification threshold.

    reverse: bool, default: False
        If it is True, then classify (y <= thresh) to be 1.
    """
    y = (y_pred >= thresh) if not reverse else (y_pred <= thresh)
    return np.mean(y == y_true)

def auc(y_pred, y_true):
    """
    Compute area under ROC curve score.

    Params:
    -------
    y_pred: np.array
        Predicted (Bernoulli) probabilities.

    y_true: np.array, binary
        True (Bernoulli) labels.
    """
    return roc_auc_score(y_true, y_pred)


def inherit_docstrings(cls):
    """
    Decorator to inherit docstring of class/method
    """
    for name, func in getmembers(cls, isfunction):
        if func.__doc__:
            continue

        parent = cls.__mro__[1]

        if hasattr(parent, name):
            func.__doc__ = getattr(parent, name).__doc__

    return cls    
