import numpy as np
import torch
import torch.nn.functional as F

from brainGNN.dataset.brain_dataset import dense_to_ind_val
from gae.renyi import calculate_conditional_MI


"""
joint_uncond:
    Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
Inputs:
    - params['Nalpha'] monte-carlo samples per causal factor
    - params['Nbeta']  monte-carlo samples per noncausal factor
    - params['K']      number of causal factors
    - params['L']      number of noncausal factors
    - params['M']      number of classes (dimensionality of classifier output)
    - decoder
    - classifier
    - device
Outputs:
    - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond(params, decoder, classifier, adj, feat, node_idx=None, act=torch.sigmoid, mu=0, std=1, device=None, brain=False, labels=None):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)
            # Ensure feat and adj are not repeated since batch size is 1
    org_adj = adj
    feat_new = feat.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    adj = adj.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    if torch.is_tensor(mu):
        alpha_mu = mu[:,:params['K']]
        beta_mu = mu[:,params['K']:]
        
        alpha_std = std[:,:params['K']]
        beta_std = std[:,params['K']:]
    else:
        alpha_mu = 0
        beta_mu = 0
        alpha_std = 1
        beta_std = 1

    alpha = torch.randn((params['Nalpha'], adj.shape[-1], params['K']), device=device).mul(alpha_std).add_(alpha_mu).repeat(1,params['Nbeta'],1).view(params['Nalpha'] * params['Nbeta'] , adj.shape[-1], params['K'])
    beta = torch.randn((params['Nalpha'] * params['Nbeta'], adj.shape[-1], params['L']), device=device).mul(beta_std).add_(beta_mu)
    zs = torch.cat([alpha, beta], dim=-1)  
    xhat = act(decoder(zs)) * adj
    if  brain:
        logits=[]
        
        for i in range(xhat.shape[0]):
            x_hat_i=xhat[i]
            if node_idx is None:
                logits_i = classifier(feat, x_hat_i)[0]

            else:
                logits_i = classifier(feat, x_hat_i)[0][:,node_idx,:]
            logits.append(logits_i)
            # return
        logits = torch.stack(logits)
        
    else:
        if node_idx is None:
            logits = classifier(feat_new, xhat)[0]
        else:
            logits = classifier(feat_new, xhat)[0][:,node_idx,:]
    # print("feat, adj", feat.shape, org_adj.shape)
    # org_label = classifier(feat, org_adj)
    # print(org_label)
    # # fairness = calculate_conditional_MI(alpha, org_label, beta) 
    fairness = 0
    if labels != None:
        print(len(labels), "len")
        fairness = calculate_conditional_MI(alpha[0], labels, beta[0])    
        print("fairness")


    yhat = F.softmax(logits, dim=1).view(params['Nalpha'], params['Nbeta'] ,params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p+eps)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    print(fairness)
    return -I,fairness


def beta_info_flow(params, decoder, classifier, adj, feat, node_idx=None, act=torch.sigmoid, mu=0, std=1, device=None, brain=False):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat_new = feat.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    adj = adj.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    if torch.is_tensor(mu):
        alpha_mu = mu[:,:params['K']]
        beta_mu = mu[:,params['K']:]
        
        alpha_std = std[:,:params['K']]
        beta_std = std[:,params['K']:]
    else:
        alpha_mu = 0
        beta_mu = 0
        alpha_std = 1
        beta_std = 1

    alpha = torch.randn((params['Nalpha'] * params['Nbeta'], adj.shape[-1], params['K']), device=device).mul(alpha_std).add_(alpha_mu)
    beta = torch.randn((params['Nalpha'], adj.shape[-1], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(1,params['Nbeta'],1).view(params['Nalpha'] * params['Nbeta'] , adj.shape[-1], params['L'])
    zs = torch.cat([alpha, beta], dim=-1)
    xhat = act(decoder(zs)) * adj
    if  brain:
        logits=[]
        
        for i in range(xhat.shape[0]):
            x_hat_i=xhat[i]
            if node_idx is None:
                logits_i = classifier(feat, x_hat_i)[0]

            else:
                logits_i = classifier(feat, x_hat_i)[0][:,node_idx,:]
            logits.append([logits_i[0].item(),logits_i[1].item()])
            # return
        logits = torch.as_tensor(logits)
        
    else:
        if node_idx is None:
            logits = classifier(feat_new, xhat)[0]
        else:
            logits = classifier(feat_new, xhat)[0][:,node_idx,:]
    yhat = F.softmax(logits, dim=1).view(params['Nalpha'], params['Nbeta'] ,params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p+eps)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    return -I, None
    for i in range(0, params['Nalpha']):
        # alpha = torch.randn((100, params['K']), device=device)
        # zs = torch.zeros((params['Nbeta'], 100, params['z_dim']), device=device)  
        # for j in range(0, params['Nbeta']):
        #     beta = torch.randn((100, params['L']), device=device)
        #     zs[j,:,:params['K']] = alpha
        #     zs[j,:,params['K']:] = beta
        
        alpha = torch.randn((100, params['K']), device=device).mul(alpha_std).add_(alpha_mu).unsqueeze(0).repeat(params['Nbeta'],1,1)
        beta = torch.randn((params['Nbeta'], 100, params['L']), device=device).mul(beta_std).add_(beta_mu)
        zs = torch.cat([alpha, beta], dim=-1)
        # decode and classify batch of Nbeta samples with same alpha
        xhat = torch.sigmoid(decoder(zs)) * adj
        yhat = F.softmax(classifier(feat, xhat)[0], dim=1)
        p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
        I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
        q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    negCausalEffect = -I
    info = {"xhat" : xhat, "yhat" : yhat}
    return negCausalEffect, info


"""
joint_uncond_singledim:
    Sample-based estimate of "joint, unconditional" causal effect
    for single latent factor, -I(z_i; Yhat). Note the interpretation
    of params['Nalpha'] and params['Nbeta'] here: Nalpha is the number
    of samples of z_i, and Nbeta is the number of samples of the other
    latent factors.
Inputs:
    - params['Nalpha']
    - params['Nbeta']
    - params['K']
    - params['L']
    - params['M']
    - decoder
    - classifier
    - device
    - dim (i : compute -I(z_i; Yhat) **note: i is zero-indexed!**)
Outputs:
    - negCausalEffect (sample-based estimate of -I(z_i; Yhat))
    - info['xhat']
    - info['yhat']
"""
def joint_uncond_singledim(params, decoder, classifier, adj, feat, dim, node_idx=None, act=torch.sigmoid, mu=0, std=1, device=None,brain=False):
    eps = 1e-8
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat_new = feat.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    adj = adj.repeat(params['Nalpha'] * params['Nbeta'], 1, 1)
    if torch.is_tensor(mu):
        alpha_mu = mu
        beta_mu = mu[:,dim]
        
        alpha_std = std
        beta_std = std[:,dim]
    else:
        alpha_mu = 0
        beta_mu = 0
        alpha_std = 1
        beta_std = 1

    alpha = torch.randn((params['Nalpha'], adj.shape[-1]), device=device).mul(alpha_std).add_(alpha_mu).repeat(1,params['Nbeta']).view(params['Nalpha'] * params['Nbeta'] , adj.shape[-1])
    zs = torch.randn((params['Nalpha'] * params['Nbeta'], adj.shape[-1], params['z_dim']), device=device).mul(beta_std).add_(beta_mu)
    zs[:,:,dim] = alpha
    xhat = act(decoder(zs)) * adj
    if  brain:
        logits=[]
        
        for i in range(xhat.shape[0]):
            x_hat_i=xhat[i]
            if node_idx is None:
                logits_i = classifier(feat, x_hat_i)[0]

            else:
                logits_i = classifier(feat, x_hat_i)[0][:,node_idx,:]
            logits.append([logits_i[0].item(),logits_i[1].item()])
            # return
        logits = torch.as_tensor(logits)
        
    else:
        if node_idx is None:
            logits = classifier(feat_new, xhat)[0]
        else:
            logits = classifier(feat_new, xhat)[0][:,node_idx,:]
    yhat = F.softmax(logits, dim=1).view(params['Nalpha'], params['Nbeta'] ,params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p+eps)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    return -I, None
    # eps = 1e-8
    # I = 0.0
    # q = torch.zeros(params['M']).to(device)
    # zs = np.zeros((params['Nalpha']*params['Nbeta'], params['z_dim']))
    # for i in range(0, params['Nalpha']):
    #     z_fix = np.random.randn(1)
    #     zs = np.zeros((params['Nbeta'],params['z_dim']))  
    #     for j in range(0, params['Nbeta']):
    #         zs[j,:] = np.random.randn(params['K']+params['L'])
    #         zs[j,dim] = z_fix
    #     # decode and classify batch of Nbeta samples with same alpha
    #     xhat = decoder(torch.from_numpy(zs).float().to(device))
    #     yhat = classifier(xhat)[0]
    #     p = 1./float(params['Nbeta']) * torch.sum(yhat,0) # estimate of p(y|alpha)
    #     I = I + 1./float(params['Nalpha']) * torch.sum(torch.mul(p, torch.log(p+eps)))
    #     q = q + 1./float(params['Nalpha']) * p # accumulate estimate of p(y)
    # I = I - torch.sum(torch.mul(q, torch.log(q+eps)))
    # negCausalEffect = -I
    # info = {"xhat" : xhat, "yhat" : yhat}
    # return negCausalEffect, info