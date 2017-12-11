import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import inherit_docstrings
import torch.nn.functional as F

class Model(nn.Module):
    """
    Base class of all models
    """

    def __init__(self, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu
        self.embeddings = []

    def forward(self, X):
        """
        Given a (mini)batch of triplets X of size M, predict the validity.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        Returns:
        --------
        y: Mx1 vectors
            Contains the probs result of each M data.
        """
        raise NotImplementedError

    def predict(self, X, sigmoid=False):
        """
        Predict the score of test batch.

        Params:
        -------
        X: int matrix of M x 3, where M is the (mini)batch size
            First row contains index of head entities.
            Second row contains index of relationships.
            Third row contains index of tail entities.

        sigmoid: bool, default: False
            Whether to apply sigmoid at the prediction or not. Useful if the
            predicted result is scores/logits.

        Returns:
        --------
        y_pred: np.array of Mx1
        """
        y_pred = self.forward(X).view(-1, 1)

        if sigmoid:
            y_pred = F.sigmoid(y_pred)

        if self.gpu:
            return y_pred.cpu().data.numpy()
        else:
            return y_pred.data.numpy()

    def log_loss(self, y_pred, y_true, average=True):
        """
        Compute log loss (Bernoulli NLL).

        Params:
        -------
        y_pred: vector of size Mx1
            Contains prediction logits.

        y_true: np.array of size Mx1 (binary)
            Contains the true labels.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        if self.gpu:
            y_true = Variable(torch.from_numpy(y_true.astype(np.float32)).cuda())
        else:
            y_true = Variable(torch.from_numpy(y_true.astype(np.float32)))

        nll = F.binary_cross_entropy_with_logits(y_pred, y_true, size_average=average)

        norm_E_real = torch.norm(self.emb_E_real.weight, 2, 1)
        norm_E_imag = torch.norm(self.emb_E_imag.weight, 2, 1)

        norm_R_real = torch.norm(self.emb_R_real.weight, 2, 1)
        norm_R_imag = torch.norm(self.emb_R_imag.weight, 2, 1)
        # Penalize when embeddings norms larger than one
        nlp1_real = torch.sum(torch.clamp(norm_E_real - 1, min=0))
        nlp1_imag = torch.sum(torch.clamp(norm_E_imag - 1, min=0))

        nlp2_real = torch.sum(torch.clamp(norm_R_real - 1, min=0))
        nlp2_imag = torch.sum(torch.clamp(norm_R_imag - 1, min=0))

        if average:
            nlp1_real /= nlp1_real.size(0)
            nlp2_real /= nlp2_real.size(0)
            nlp1_imag /= nlp1_imag.size(0)
            nlp2_imag /= nlp2_imag.size(0)
        return nll + self.lam*nlp1_real + self.lam*nlp1_imag + self.lam*nlp2_real + self.lam*nlp2_imag

    def ranking_loss(self, y_pos, y_neg, margin=1, C=1, average=True):
        """
        Compute loss max margin ranking loss.

        Params:
        -------
        y_pos: vector of size Mx1
            Contains scores for positive samples.

        y_neg: np.array of size Mx1 (binary)
            Contains the true labels.

        margin: float, default: 1
            Margin used for the loss.

        C: int, default: 1
            Number of negative samples per positive sample.

        average: bool, default: True
            Whether to average the loss or just summing it.

        Returns:
        --------
        loss: float
        """
        M = y_pos.size(0)

        y_pos = y_pos.view(-1).repeat(C)  # repeat to match y_neg
        y_neg = y_neg.view(-1)

        # target = [-1, -1, ..., -1], i.e. y_neg should be higher than y_pos
        target = -np.ones(M*C, dtype=np.float32)

        if self.gpu:
            target = Variable(torch.from_numpy(target).cuda())
        else:
            target = Variable(torch.from_numpy(target))

        loss = F.margin_ranking_loss(
            y_pos, y_neg, target, margin=margin, size_average=average
        )

        return loss

    def normalize_embeddings(self):
        for e in self.embeddings:
            e.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def initialize_embeddings(self):
        r = 6/np.sqrt(self.k)

        for e in self.embeddings:
            e.weight.data.uniform_(-r, r)

        self.normalize_embeddings()


@inherit_docstrings
class ComplexE(Model):


    def __init__(self, n_e, n_r, k, lam, gpu=False):
        """

        Params:
        -------
            n_e: int
                Number of entities in dataset.

            n_r: int
                Number of relationships in dataset.

            k: int
                Embedding size.

            lam: float
                Prior strength of the embeddings. Used to constaint the
                embedding norms inside a (euclidean) unit ball. The prior is
                Gaussian, this param is the precision.

            gpu: bool, default: False
                Whether to use GPU or not.
        """
        super(ComplexE, self).__init__(gpu)

        # Hyperparams
        self.n_e = n_e
        self.n_r = n_r
        self.k = k
        self.lam = lam

        # Nets
        self.emb_E_real = nn.Embedding(self.n_e, self.k)
        self.emb_E_imag = nn.Embedding(self.n_e, self.k)

        self.emb_R_real = nn.Embedding(self.n_r, self.k)
        self.emb_R_imag = nn.Embedding(self.n_r, self.k)

        self.embeddings = [self.emb_E_real, self.emb_E_imag, self.emb_R_real , self.emb_R_imag]
        self.initialize_embeddings()

        # Copy all params to GPU if specified
        if self.gpu:
            self.cuda()

    def forward(self, X):
        # Decompose X into head, relationship, tail
        hs, ls, ts = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            hs = Variable(torch.from_numpy(hs).cuda())
            ls = Variable(torch.from_numpy(ls).cuda())
            ts = Variable(torch.from_numpy(ts).cuda())
        else:
            hs = Variable(torch.from_numpy(hs))
            ls = Variable(torch.from_numpy(ls))
            ts = Variable(torch.from_numpy(ts))

        # Project to embedding, each is M x k
        e_hs_real = self.emb_E_real(hs)
        e_hs_imag = self.emb_E_imag(hs)
        e_ts_real = self.emb_E_real(ts)
        e_ts_imag = self.emb_E_imag(ts)

        W_real = self.emb_R_real(ls)
        W_imag = self.emb_R_imag(ls)

        # Forward
        score = e_hs_real * W_real * e_ts_real + e_hs_imag * W_real * e_ts_imag + e_hs_real * W_imag * e_ts_imag - e_hs_imag * W_imag * e_ts_real

        f = torch.sum(score, 1)

        return f.view(-1, 1)
