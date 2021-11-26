"""Implementation of some transformers"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class AffineElementwiseTransform():
    """Invertible elementwise affine transformation."""
    def __init__(self, z_dim=1):
        super(AffineElementwiseTransform, self).__init__()
        self.z_dim = z_dim
        loc, scale = nn.Parameter(torch.ones(z_dim,)), nn.Parameter(torch.ones(z_dim,))
        self.parameters = [loc, scale]

    def forward_transform(self,u):
        loc, scale = self.parameters
        out = scale*u + loc
        log_jac_det = torch.log(scale).sum()
        return out, log_jac_det

    def inverse_transform(self, x):
        loc, scale = self.parameters
        if not (scale>0).all():
            raise ValueError("Scale must be non-zero.")
        out = (x-loc) / scale
        log_jac_det = -torch.log(scale).sum()
        return out, log_jac_det

class PositiveLinearTransform():
    """Linear transformation x = Au where A is a matrix with non-negative values only."""
    def __init__(self, z_dim=2):
        super(PositiveLinearTransform, self).__init__()
        self.log_weight = nn.Parameter(torch.Tensor(z_dim,z_dim))
        self.reset_parameters()
        self.bias = nn.Parameter(torch.ones(z_dim,))
        self.parameters = [self.log_weight, self.bias]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def param_generator(self):
        for param in [self.log_weight, self.bias]:
            yield param

    def forward_transform(self, u):
        A = self.log_weight.exp()
        out = F.linear(u, A) + self.bias
        _, log_jac_det = torch.slogdet(A)
        return out, log_jac_det

    def inverse_transform(self, x):
        A = self.log_weight.exp()
        A_inv = torch.inverse(A)
        out = F.linear(x - self.bias, A_inv)
        _, log_jac_det = torch.slogdet(A_inv)
        return out, log_jac_det
