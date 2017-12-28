import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import inherit_docstrings
import torch.nn.functional as F
from torch.nn.init import xavier_normal
from base import Model


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

        # Xavier Initialization
        """
        xavier_normal(self.emb_E_real.weight.data)
        xavier_normal(self.emb_E_imag.weight.data)
        xavier_normal(self.emb_R_real.weight.data)
        xavier_normal(self.emb_R_imag.weight.data)
        """
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

        score = torch.sum(e_hs_real * W_real * e_ts_real,1) \
                + torch.sum(e_hs_imag * W_real * e_ts_imag,1) \
                 + torch.sum(e_hs_real * W_imag * e_ts_imag,1) \
                  - torch.sum(e_hs_imag * W_imag * e_ts_real,1)


        return score.view(-1, 1)

    def predict_all(self, X):
        s, p, o = X[:, 0], X[:, 1], X[:, 2]

        if self.gpu:
            s = Variable(torch.from_numpy(s).cuda())
            p = Variable(torch.from_numpy(p).cuda())
            o = Variable(torch.from_numpy(o).cuda())
        else:
            s = Variable(torch.from_numpy(s))
            p = Variable(torch.from_numpy(p))
            o = Variable(torch.from_numpy(o))
        batch_size = len(X)
        e_hs_real = self.emb_E_real(s).view(batch_size, -1)
        W_real = self.emb_R_real(p).view(batch_size, -1)
        e_hs_imag = self.emb_E_imag(s).view(batch_size, -1)
        W_imag = self.emb_R_imag(p).view(batch_size, -1)

        # Pred t
        realrealreal = torch.mm(e_hs_real*W_real, self.emb_E_real.weight.transpose(1,0))
        realimgimg = torch.mm(e_hs_real*W_imag, self.emb_E_imag.weight.transpose(1,0))
        imgrealimg = torch.mm(e_hs_imag*W_real, self.emb_E_imag.weight.transpose(1,0))
        imgimgreal = torch.mm(e_hs_imag*W_imag, self.emb_E_real.weight.transpose(1,0))
        pred_t = realrealreal + realimgimg + imgrealimg - imgimgreal

        # Pred h
        e_ts_real = self.emb_E_real(o).view(batch_size, -1)
        e_ts_imag = self.emb_E_imag(o).view(batch_size, -1)
        score1 = torch.mm(e_hs_real*W_real, self.emb_E_real.weight.transpose(1,0))
        score2 = torch.mm(e_hs_real*W_imag, self.emb_E_imag.weight.transpose(1,0))
        score3 = torch.mm(e_hs_imag*W_real, self.emb_E_imag.weight.transpose(1,0))
        score4 = torch.mm(e_hs_imag*W_imag, self.emb_E_real.weight.transpose(1,0))
        pred_h = score1 + score2 + score3 - score4
        
        return pred_h, pred_t