from __future__ import print_function
from __future__ import division

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from gae.BGCN.utils import normalize_torch
from gae.layers import BGraphConvolution
 

# torch.cuda.device_count()


# seed = 5
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.normal_(tensor=self.weight, std=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class BBGDC(nn.Module):
    def __init__(self, num_pars, alpha=0.8, kl_scale=1.0):
        super(BBGDC, self).__init__()
        self.num_pars = num_pars
        self.alpha = alpha
        self.kl_scale = kl_scale
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)
    
    def get_params(self):
        a = F.softplus(self.a_uc.clamp(min=-10.))
        b = F.softplus(self.b_uc.clamp(min=-10., max=50.))
        return a, b
    
    def sample_pi(self):
        a, b = self.get_params()
        u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6)
        if torch.cuda.is_available():
            u = u.cuda()
        return (1 - u.pow_(1./b)).pow_(1./a)
    
    def get_weight(self, num_samps, training, samp_type='rel_ber'):
        temp = torch.Tensor([0.67])
        if torch.cuda.is_available():
            temp = temp.cuda()
        
        if training:
            pi = self.sample_pi()
            p_z = RelaxedBernoulli(temp, probs=pi)
            z = p_z.rsample(torch.Size([num_samps]))
        else:
            if samp_type=='rel_ber':
                pi = self.sample_pi()
                p_z = RelaxedBernoulli(temp, probs=pi)
                z = p_z.rsample(torch.Size([num_samps]))
            elif samp_type=='ber':
                pi = self.sample_pi()
                p_z = torch.distributions.Bernoulli(probs=pi)            
                z = p_z.sample(torch.Size([num_samps]))
        return z, pi
    
    def get_reg(self):
        #learn droprate: change it from a hyperparameter to learnable param
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 - torch.digamma(b) - 1./b) + torch.log(a*b + 1e-10) - math.log(self.alpha) - (b-1)/b
        kld = (self.kl_scale) * kld.sum()
        return kld



class BBGDCGCN(nn.Module):
    def __init__(self, nfeat_list, dropout, nblock, nlay, num_edges):
        super(BBGDCGCN, self).__init__()
        
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.nblock = nblock
        self.num_edges = num_edges
        self.num_nodes = int(np.sqrt(num_edges))
        self.drpcon_list = []
        self.dropout = dropout
        gcs_list = []
        idx = 0
        for i in range(nlay):
            if i==0:
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
                idx += 1
            else:
                self.drpcon_list.append(BBGDC(1))
                for j in range(self.nblock):
                    gcs_list.append([str(idx), GraphConvolution(int(nfeat_list[i]/self.nblock), nfeat_list[i+1])])
                    idx += 1
        
        self.drpcons = nn.ModuleList(self.drpcon_list)
        self.gcs = nn.ModuleDict(gcs_list)
        # feature list is number of features in each layer
        self.nfeat_list = nfeat_list
    
    def forward(self, x, labels, adj, obs_idx, warm_up, adj_normt, training=True
                , mul_type='norm_first', samp_type='rel_ber'):
        h_perv = x
        kld_loss = 0.0
        drop_rates = []
        for i in range(self.nlay):
            # get mask and drop rate probality in the next layer 
            mask_vec, drop_prob = self.drpcons[i].get_weight(self.nblock*self.num_edges, training, samp_type)
            # squeeze to play on dimensions if input is of shape: (Ax1xBxCx1xD) result :  (AxBxCxD)(AxBxCxD) 
            mask_vec = torch.squeeze(mask_vec)
            # append drop prob to the list where each layer has its own drop rate
            drop_rates.append(drop_prob)
            # start by first layer: it has a special treatment 
            if i==0:
                # cut mast matrix to the number of edges and reshape it based on Num_nodes X Num_nodes aka adj matrix
                mask_mat = torch.reshape(mask_vec[:self.num_edges], (self.num_nodes, self.num_nodes)).cpu()
                
                if mul_type=='norm_sec':
                    # multiply adj by mask add self loop then normalize
                    adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cpu())
                elif mul_type=='norm_first':
                    # normalize adj matr multiply adj by mask  
                    adj_lay = torch.mul(mask_mat, adj_normt).cpu()
                
                x = F.relu(self.gcs[str(i)](x, adj_lay))
                x = F.dropout(x, self.dropout, training=training)
            
            else:
                # devide features on number of nblocks 
                feat_pblock = int(self.nfeat_list[i]/self.nblock)
                for j in range(self.nblock):
                    # Reshape the appropriate segment of the mask vector to form a mask matrix
                    # for the current block, matching the adjacency matrix dimensions
                    mask_mat = torch.reshape(mask_vec[j*self.num_edges:(j+1)*self.num_edges]
                                             , (self.num_nodes, self.num_nodes)).cpu()
                     
                    # same as layer 1
                    if mul_type=='norm_sec':
                        adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cpu())
                    elif mul_type=='norm_first':
                        adj_lay = torch.mul(mask_mat, adj_normt).cpu()
                    
                    # if we are not in last layer : (last layer is for output)
                    if i<(self.nlay-1):
                        if j==0:
                            # first block: get the second 
                            x_out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            x_out = x_out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                    else:
                        if j==0:
                            out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            out = out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                
                if i<(self.nlay-1):
                    x = x_out
                    x = F.dropout(F.relu(x), self.dropout, training=training)
            
            
            kld_loss += self.drpcons[i].get_reg()
            
        
        output = F.log_softmax(out, dim=1)
        
        nll_loss = self.loss(labels, output, obs_idx)
        tot_loss = nll_loss + warm_up * kld_loss
        drop_rates = torch.stack(drop_rates)
        return output, tot_loss, nll_loss, kld_loss, drop_rates
    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])


class VBGAE(nn.Module):
    def __init__(self, nfeat_list, dropout, nlay, nblock, num_edges,dc=None, device='cpu'):
        super(VBGAE, self).__init__()
        
        # assert len(nfeat_list)==nlay+1
        self.device = device
        self.nlay=nlay
        self.dc = dc
        self.nblock = nblock
        self.num_edges = num_edges
        self.num_nodes = int(np.sqrt(num_edges))
        self.drpcon_list = []
        self.dropout = dropout
        gcs_list = []
        # self.batch_norm = []
        idx = 0
        for i in range(nlay-1):
            if i==0:
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx), BGraphConvolution(nfeat_list[i], nfeat_list[i+1])])
                # self.batch_norm.append(nn.BatchNorm1d(nfeat_list[i+1], device=device))
                idx += 1
            elif (i<(nlay-2)):
                self.drpcon_list.append(BBGDC(1))
                for j in range(self.nblock):
                    gcs_list.append([str(idx), BGraphConvolution(int(nfeat_list[i]/self.nblock), nfeat_list[i+1])])
                    # self.batch_norm.append(nn.BatchNorm1d(nfeat_list[i+1], device=device))
                    idx += 1
            else:
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx), BGraphConvolution(int(nfeat_list[i]), nfeat_list[i+1])])
                self.drpcon_list.append(BBGDC(1))
                gcs_list.append([str(idx+1), BGraphConvolution(int(nfeat_list[i]), nfeat_list[i+1])])
                idx +=2

        
        # self.batch_norm = nn.ModuleList(self.batch_norm)
        self.drpcons = nn.ModuleList(self.drpcon_list)
        print("list gcs",len(gcs_list))

        self.gcs = nn.ModuleDict(gcs_list)
        # feature list is number of features in each layer
        self.nfeat_list = nfeat_list
    
    def forward(self, x, adj_normt, warm_up=1, adj=None, training=True
                , mul_type='norm_first', samp_type='rel_ber', graph_size=None):

        logvar = 0  
        mu = 0
        h_perv = x
        kld_loss = 0.0
        drop_rates = []
        num_edges = graph_size ** 2
        num_nodes = graph_size 
        for i in range(self.nlay-1):
            # get mask and drop rate probality in the next layer 
            mask_vec, drop_prob = self.drpcons[i].get_weight(self.nblock*num_edges, training, samp_type)
            # squeeze to play on dimensions if input is of shape: (Ax1xBxCx1xD) result :  (AxBxCxD)(AxBxCxD) 
            mask_vec = torch.squeeze(mask_vec)
            # append drop prob to the list where each layer has its own drop rate
            drop_rates.append(drop_prob)
            # start by first layer: it has a special treatment 
            if i==0:
                # cut mast matrix to the number of edges and reshape it based on Num_nodes X Num_nodes aka adj matrix
                mask_mat = torch.reshape(mask_vec[:num_edges], (num_nodes, num_nodes)).to(self.device)
                if mul_type=='norm_sec':
                    # multiply adj by mask add self loop then normalize
                    adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cpu())
                elif mul_type=='norm_first':
                    # normalize adj matr multiply adj by mask
                    adj_lay = torch.mul(mask_mat, adj_normt).to(self.device)

                x = torch.squeeze(x)
                adj_lay = torch.squeeze(adj_lay)
                x = F.relu(self.gcs[str(i)](x, adj_lay))
                # x = self.batch_norm[i](x)
                x = F.dropout(x, self.dropout, training=training)

            
            else:
                # devide features on number of nblocks 
                feat_pblock = int(self.nfeat_list[i]/self.nblock)
                for j in range(self.nblock):
                    # Reshape the appropriate segment of the mask vector to form a mask matrix
                    # for the current block, matching the adjacency matrix dimensions
                    mask_mat = torch.reshape(mask_vec[j*num_edges:(j+1)*num_edges]
                                             , (num_nodes, num_nodes)).to(self.device)
                     
                    # same as layer 1
                    if mul_type=='norm_sec':
                        adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).to(self.device))
                    elif mul_type=='norm_first':
                        adj_lay = torch.mul(mask_mat, adj_normt).to(self.device)
                    x = torch.squeeze(x)
                    adj_lay = torch.squeeze(adj_lay)                
                    # if we are not in last layer : (last 2 layers are for output)
                    if i<(self.nlay-2):
                        if j==0:
                            # first block: get the second 
                            x_out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                            # x_out = self.batch_norm[i](x_out)
                        else:
                            x_out = x_out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                            # x_out = self.batch_norm[i](x_out)
                    else:
                        mu = self.gcs[str((i-1)*self.nblock+1)](x[:,0:self.nfeat_list[i]], adj_lay)
                        logvar = self.gcs[str((i)*self.nblock)](x[:,0:self.nfeat_list[i]], adj_lay)
                        mu = F.dropout(F.relu(mu), self.dropout, training=training)
                        logvar = F.dropout(F.relu(logvar), self.dropout, training=training)
                #print(i,self.nlay-1)
                if i<(self.nlay-2):
                    x = x_out

                    x = F.dropout(F.relu(x), self.dropout, training=training)
            
            
            kld_loss += self.drpcons[i].get_reg()
            
        
        #reparemtrize track : this is used to bring out the randomness from z and apply backprp
        if self.training:
            std = torch.exp(logvar)
            #sample(noisy simple)
            eps = torch.randn_like(std)
            z =  eps.mul(std).add_(mu)
        else:
            z= mu
        kld_loss =  warm_up * kld_loss
        drop_rates = torch.stack(drop_rates)
        z = z.unsqueeze(dim=0)
        mu = mu.unsqueeze(dim=0)
        return self.dc(z), z, mu, logvar, kld_loss, drop_rates
