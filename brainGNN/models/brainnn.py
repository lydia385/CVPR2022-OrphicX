import torch
from collections import defaultdict
import numpy as np 
from itertools import permutations
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F
from brainGNN.dataset.brain_dataset import dense_to_ind_val
from brainGNN.models.gcn import GCN
from brainGNN.models.mlp import MLP
from brainGNN.models.gat import GAT


class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data, adj=None):
        if(adj != None):
            x = data
            edge_index, edge_attr = dense_to_ind_val(adj)
            batch = None
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g = self.gnn(x, edge_index, edge_attr, batch)
        log_logits = F.log_softmax(g, dim=-1)
        
        return log_logits

def build_model(args, device, model_name, num_features, num_nodes):
    if model_name == 'gcn':
        model = BrainNN(args,
                      GCN(num_features, args, num_nodes, num_classes=2),
                      MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    elif model_name == 'gat':
        model = BrainNN(args,
                      GAT(num_features, args, num_nodes, num_classes=2),
                      MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2),
                      ).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model


def load_checkpoint(model, load_path):
    
    checkpoint = torch.load(load_path,map_location=torch.device('cpu') )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch'],
    print(f"model has trained for {epoch} of epochs")

    return model, epoch