import numpy as np
from sklearn.utils import shuffle as skshuffle
from sklearn.metrics import roc_auc_score
from inspect import getmembers, isfunction
import scipy.stats as st

# Some Utilities
def eval_embeddings(model, X_test, n_e, k, n_sample=1000, X_lit_s_ori=None, X_lit_o_ori=None, X_lit_img=None, X_lit_txt=None):
    """
    Compute Mean Reciprocal Rank and Hits@k score of embedding model.
    The procedure follows Bordes, et. al., 2011.

    Params:
    -------
    model: kga.Model
        Embedding model to be evaluated.

    X_test: M x 3 matrix, where M is data size
        Contains M test triplets.

    n_e: int
        Number of entities in dataset.

    k: int or list
        Max rank to be considered, i.e. to be used in Hits@k metric.

    n_sample: int, default: 1000
        Number of negative entities to be considered. These n_sample negative
        samples are randomly picked w/o replacement from [0, n_e). Consider
        setting this to get the (fast) approximation of mrr and hits@k.

    X_lit: n_e x n_l matrix
        Matrix containing all literals for all entities.


    Returns:
    --------
    mrr: float
        Mean Reciprocal Rank.

    hitsk: float or list
        Hits@k.
    """
    M = X_test.shape[0]

    X_corr_h = np.copy(X_test)
    X_corr_t = np.copy(X_test)

    N = n_sample+1 if n_sample is not None else n_e+1

    scores_h = np.zeros([M, N])
    scores_t = np.zeros([M, N])

    # Gather scores for correct entities
    y = model.predict(X_test).ravel()
    scores_h[:, 0] = y
    scores_t[:, 0] = y

    if n_sample is not None:
        # Gather scores for some random negative entities
        ents = np.random.choice(np.arange(n_e), size=n_sample, replace=False)
    else:
        ents = np.arange(n_e)

    for i, e in enumerate(ents):
        idx = i+1  # as i == 0 is for correct triplet score

        X_corr_h[:, 0] = e
        X_corr_t[:, 2] = e

        y_h = model.predict(X_corr_h).ravel()
        y_t = model.predict(X_corr_t).ravel()

        scores_h[:, idx] = y_h
        scores_t[:, idx] = y_t

    ranks_h = np.array([st.rankdata(s)[0] for s in scores_h])
    ranks_t = np.array([st.rankdata(s)[0] for s in scores_t])

    mrr = (np.mean(1/ranks_h) + np.mean(1/ranks_t)) / 2

    if isinstance(k, list):
        hitsk = [(np.mean(ranks_h <= r) + np.mean(ranks_t <= r)) / 2 for r in k]
    else:
        hitsk = (np.mean(ranks_h <= k) + np.mean(ranks_t <= k)) / 2

    return mrr, hitsk


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
