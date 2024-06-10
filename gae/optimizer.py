import torch
import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    # Binary Cross-Entropy Loss with logits
    bce = F.binary_cross_entropy_with_logits(preds.flatten(1).T, labels.flatten(1).T, pos_weight=pos_weight, reduction='sum')
    bce = bce / n_nodes  # Normalize by the number of nodes

    cost = norm * bce

    # Kullback-Leibler Divergence (KLD)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), -1))

    return cost + KLD
