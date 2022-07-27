from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision import transforms


def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    heatmap = sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax)

    return heatmap



def plot_classes_preds(images: list, labels: list,
                       preds: list,
                       shown_batch_size: int = 4) -> plt.figure:
    """
    Plots a batch processed by a computer vision model as a matplotlib figure
    Parameters
    ----------
    images : list
        list of torch.tensor of the images to plot
    labels : list
        list of int of the labels associated with each image
    probas : list
        list of float of the probabilities scoring for each image
    preds : list
        list of int of the prediction for each image
    names : list
        list of string of the name each image
    shown_batch_size : int, optional
        number of images to be shown within the batch
        (must for <= to batch size), by default 4
    Returns
    -------
    plt.figure
        matplotlib figure with the images, the associated edges and metadata
    """ 
    
    # plot the images in the batch, along with predicted and true labels

    n_img_per_line = 6
    n_lines = shown_batch_size // n_img_per_line
    fig = plt.figure(figsize=(12 * n_img_per_line, 12 * n_lines))
    axes = []
    for idx in np.arange(shown_batch_size):
        axes.append(fig.add_subplot(n_lines, n_img_per_line,
                                    idx+1, xticks=[], yticks=[],
                                    label=f'color_{idx}')) 
        image = images[idx].detach().clone()
      
        img_permute = torch.permute(image, (1,2,0))
        img = img_permute.cpu().numpy()
        # img_np[0] = (img_np[0]-img_np[0].min())/img_np[0].max()
        # img_np[1] = (img_np[1]-img_np[1].min())/img_np[1].max()
        # img_np[2] = (img_np[2]-img_np[2].min())/img_np[2].max()
        norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        #axes[-1].imshow(((img_np-img_np.min())/img_np.max() * 255).astype('uint8'))
        axes[-1].imshow((norm * 255).astype('uint8'))
        plt.figure()
        img_np = img
        plt.imshow(((img_np-img_np.min())/img_np.max() * 255).astype('uint8'))
        plt.show()
        axes[-1].set_title(
            "(pred: {0}) \n(label: {1})".format(
                int(preds[idx]),
                labels[idx]),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
            fontdict={'fontsize': 30}) 
    plt.tight_layout()
    return fig