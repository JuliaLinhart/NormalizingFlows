
import torch
from torch.utils import data

from eval import plot_contours_true_learned_nf

def train(flow, x_samples, context=None, num_epochs=5000, batch_size=50, lr=0.005, plot=True, plt_min=0, plt_max=10, validation = False):
    tensors = [x_samples]
    if context is not None:
        tensors.append(context)

    dataset = data.TensorDataset(*tensors)

    if validation:
        val_size = int(0.1* len(dataset))
        train_size = len(dataset) - val_size
        train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = data.DataLoader(train_set, batch_size=batch_size)
        val_loader = data.DataLoader(val_set, batch_size=batch_size)
    else:
        train_loader = data.DataLoader(dataset, batch_size=batch_size)

    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    best_val_loss = 0
    epochs_since_last_improvement = 0
    epoch = 0
    while (epoch < num_epochs) and (epochs_since_last_improvement < 20):

        for sample_batch in train_loader:
            opt.zero_grad()
            loss = -flow.log_prob(*sample_batch).mean()
            loss.backward()
            opt.step()

        if validation:
            with torch.no_grad():
                val_loss = 0 
                for sample_batch in val_loader:
                    batch_loss = -flow.log_prob(*sample_batch).mean()
                    val_loss+=batch_loss.sum().item()
                # Take mean over all validation samples.
                val_loss = val_loss / (
                    len(val_loader) * val_loader.batch_size
                )
            if epoch == 0 or best_val_loss > val_loss:
                best_val_loss = val_loss
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement+=1
        
        if plot and ((epoch % 500 == 0) or (epoch == num_epochs-1) or (epochs_since_last_improvement == 20)):
            plot_contours_true_learned_nf(flow, x_samples, plt_min, plt_max)
        
        epoch+=1

    return best_val_loss, epoch
