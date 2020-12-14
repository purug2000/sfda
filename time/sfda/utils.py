import torch 
import numpy as np

def tensor_l2normalization(q):
    qn = torch.norm(q, p=2, dim=-1).clone().detach().unsqueeze(-1)
    q = q.div(qn.expand_as(q))
    return q

def ComputeSimilarity(prototypes,inp_feats):
    norm_prototype = tensor_l2normalization(prototypes)
    norm_inp_feats = tensor_l2normalization(inp_feats)
    sim_mat = torch.einsum('bij,cnj->bicn',norm_inp_feats, norm_prototype)
    return sim_mat


def compute_plabel_and_conf(prototypes, inp_feats , num_labels, cf_ratio = 1.0):
        
    #compute similarity scores between prototypes and batch tensors
    sim_mat = ComputeSimilarity(torch.tensor(prototypes),inp_feats)

    #compute class similarity scores for batch tensors
    sim = torch.mean(sim_mat,3)
    
    #Find pseudo labels
    
    labels = torch.argsort(sim, dim=2, descending = True)
    p_label = labels[:,:,0] # B x N x 1  primary label
    s_label =  labels[:,:,1] # B x N x 1 secondary label
    p_idx = p_label.cpu().numpy()
    s_idx = s_label.cpu().numpy()

    idx_0 = np.arange(sim.shape[0])[:,None]
    idx_1 = np.arange(sim.shape[1])[None,:]
   #finding prototypes for both  classes
    # print(p_idx)
    # print(s_idx)
    
    p_mat = sim_mat[idx_0,idx_1,p_idx,:]
    s_mat = sim_mat[idx_0,idx_1,s_idx,:]
  
    #finding max and min similarity from closest and second closest classes respectively

    min_sim_from_p_class,_  = torch.min(p_mat, dim = 2)
    max_sim_from_s_class,_  = torch.max(s_mat, dim = 2)
    # print(min_sim_from_p_class.shape)
    conf_mask =  (min_sim_from_p_class > cf_ratio*max_sim_from_s_class)
    return p_label, conf_mask