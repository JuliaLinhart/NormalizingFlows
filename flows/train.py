
import torch
from torch.utils import data

from eval import plot_contours_true_learned_nf

def train(flow, x_samples, context=None, num_epochs=5000, batch_size=50, lr=0.005, plot=True, plt_min=0, plt_max=10):
    tensors = [x_samples]
    if context is not None:
        tensors.append(context)

    dataset = data.TensorDataset(*tensors)
    train_loader = data.DataLoader(dataset, batch_size=batch_size)

    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    for i in range(num_epochs):
        for sample_batch in train_loader:
            opt.zero_grad()
            loss = -flow.log_prob(*sample_batch).mean()
            loss.backward()
            opt.step()
        
        if plot and ((i % 500 == 0) or (i == num_epochs-1)):
            plot_contours_true_learned_nf(flow, x_samples, plt_min, plt_max)