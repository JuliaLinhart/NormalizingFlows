"""Implementation of different flow architectures."""

import torch
import torch.nn as nn
from torch.utils import data

class Flow(object):
    def __init__(self, base_dist, transform):
        super(Flow, self).__init__()
        self.base_dist = base_dist
        self.transform = transform

    def train(self, x_samples, num_epochs, batch_size=10, lr=0.03):
        dataset = data.TensorDataset(x_samples)
        train_loader = data.DataLoader(x_samples, batch_size=batch_size)


        opt = torch.optim.Adam(self.transform.parameters, lr=lr)

        for e in range(num_epochs):
            for sample_batch in train_loader:
                opt.zero_grad()
                
                p_x = self.learned_pdf(sample_batch)
                
                loss = -torch.log(p_x).sum()
                loss.backward() 
                opt.step()

    def sample(self, nb_samples):
        u_samples = self.base_dist.sample((nb_samples,))
        x_samples, _ = self.transform.forward_transform(u_samples)
        return x_samples
        

    def learned_pdf(self,x):
        u, log_jac_det_inv = self.transform.inverse_transform(x)
        p_u = torch.exp(self.base_dist.log_prob(u))
        p_x = p_u * torch.exp(log_jac_det_inv)
        return p_x