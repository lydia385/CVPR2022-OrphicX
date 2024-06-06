import torch

def to_edge_idx(adj):
    
    row, col, edge_attr = adj[0].t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_attr