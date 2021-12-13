"""Functions to evaluate the learnt transformation/distribution."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

############### Plot functions for myflow Flows ############################

# 1D distributions: Evaluate the learnt transformation by plotting the learnt p_x 
# against the true distribution and the training samples (drawn from the true distribution)
def plot_pdfs_1D(target_dist, x_samples, flow, title=None, x_i=0.1, x_f=15, n=100):
    eval_x = torch.linspace(x_i, x_f, n)

    p_x_learned = torch.exp(flow.learned_log_pdf(eval_x))

    p_x_true = torch.exp(target_dist.log_prob(eval_x))

    fig = plt.figure(figsize=(6, 2))
    plt.plot(x_samples, np.zeros_like(x_samples), 'bx', alpha=0.5, markerfacecolor='none', markersize=6)
    plt.plot(eval_x.numpy(), p_x_true.detach().numpy(),'--', color='blue')
    plt.plot(eval_x.numpy(), p_x_learned.detach().numpy(), color='orange')
    plt.legend(["Samples", "True", "Learned"], loc="upper right")
    _ = plt.xlim([0.1, 15]); _ = plt.ylim([-0.12, 3.2])
    plt.title(title)
    plt.show()

# 2D distributions: Evaluate the learnt transformation by plotting the learnt 2D pdf-contours 
# against the true pdf-contours and the training samples (drawn from the true distribution)
def plot_2d_pdf_contours(target_dist, x_samples, flow, title=None, n=500, gaussian=False):
    x_true = target_dist.sample((n,))  # Sample from groundtruth
    x_learned = flow.sample(n).detach().numpy()  # Sample from learned

    plt.scatter(x=x_samples[:,0], y=x_samples[:,1]) # Plot training samples
    sns.kdeplot(x=x_true[:,0], y=x_true[:,1])  # Plot true distribution
    sns.kdeplot(x=x_learned[:,0], y=x_learned[:,1])  # Plot learned distribution
    plt.legend(["Samples", "True", "Learned"], loc="upper right")

    if gaussian:
        means_learned = np.mean(x_learned, axis=0)  # Learned mean
        plt.scatter(x=target_dist.mean[0], y=target_dist.mean[1], color='cyan')
        plt.scatter(x=means_learned[0], y=means_learned[1], color='magenta')
    
    plt.title(title)
    plt.show()

############### Plot functions for nf_flows Flows ############################

def plot_contours_true_learned_nf(flow, target_samples, plt_min=0, plt_max=10):
    fig, ax = plt.subplots(1, 2)
    xline = torch.linspace(plt_min, plt_max)
    yline = torch.linspace(plt_min, plt_max)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid0 = flow.log_prob(xyinput).exp().reshape(100, 100)

    ax[0].contourf(xgrid.numpy(), ygrid.numpy(), zgrid0.numpy())
    ax[0].set_xlim(left=plt_min, right=plt_max)
    ax[0].set_ylim(bottom=plt_min, top=plt_max)
    ax[0].set_title('Learned')

    ax[1].scatter(target_samples[:,0], target_samples[:,1], )
    ax[1].set_xlim(left=plt_min, right=plt_max)
    ax[1].set_ylim(bottom=plt_min, top=plt_max)
    ax[1].set_title('True')
    plt.show()