import torch 

def tensor_l2normalization(q):
    qn = torch.norm(q, p=2, dim=1).detach().unsqueeze(1)
    q = q.div(qn.expand_as(q))
    return q

def ComputeSimilarity(prototype,inp_feats):
    norm_prototype = tensor_l2normalization(prototype)
    norm_inp_feats = tensor_l2normalization(inp_feats)
    sim_mat = torch.mm(norm_inp_feats, norm_prototype.permute(1,0))
    return sim_mat


def compute_plabel_and_conf(prototype_p,prototype_f,inp_feats , cf_ratio = 1.0):
        
    #compute similarity scores between prototypes and batch tensors
    sim_mat_p = ComputeSimilarity(prototype_p,inp_feats)
    sim_mat_f = ComputeSimilarity(prototype_f,inp_feats)
    
    
    #compute class similarity scores for batch tensors
    sim_p = torch.mean(sim_mat_p,1)
    sim_f = torch.mean(sim_mat_f,1)
    
    #Find pseudo labels
    p_label  = torch.where(sim_p<sim_f, torch.zeros_like(sim_p , dtype =  torch.long ),torch.ones_like(sim_p , dtype =  torch.long ))
    
    #finding max similarity from each prototype class 
    max_p,_ =  torch.max(sim_mat_p,axis = 1)
    max_f,_ =  torch.max(sim_mat_f,axis = 1)
    
    #finding min similarity from each prototype class 
    min_p,_ =  torch.min(sim_mat_p,axis = 1)
    min_f,_ =  torch.min(sim_mat_f,axis = 1)
    
    #finding max and min similarity from close and far classes respectively
    min_sim_from_close_class =  torch.where(p_label == 1,min_p,min_f )
    max_sim_from_far_class =  torch.where(p_label == 1,max_f,max_p )
    
    conf_mask =  (min_sim_from_close_class > cf_ratio*max_sim_from_far_class)
    return p_label, conf_mask