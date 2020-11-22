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


def compute_plabel_and_conf(prototypes, inp_feats , num_labels, cf_ratio = 1.0):
        
    #compute similarity scores between prototypes and batch tensors

    sim_mat = []

    for i in range(0, num_labels):
        sim_mat_i = ComputeSimilarity(torch.tensor(prototypes[i]),inp_feats)
        # print("Sim Mat i shape: ", sim_mat_i.shape)
        sim_mat.append(sim_mat_i) # N x num_prototypes_in_class_i
    
    #compute class similarity scores for batch tensors
    sim = torch.zeros(sim_mat[0].shape[0], num_labels)

    for i in range(0, num_labels):
        sim[:,i] =  torch.mean(sim_mat[i],1)

    #Find pseudo labels
    # print("Sim_shape: ", sim)
    
    labels = torch.argsort(sim, dim=1, descending = True)
    p_label = labels[:,0] # N x 1  primary label
    s_label =  labels[:,1] # N x 1 secondary label
    print(p_label.shape)
    # p_label  = torch.where(sim_p<sim_f, torch.zeros_like(sim_p , dtype =  torch.long ),torch.ones_like(sim_p , dtype =  torch.long ))
    
    #finding max and min similarity from each prototype class 
    max_a = []
    min_a = []
    for i in range(0, num_labels):
       max_i,_ =  torch.max(sim_mat[i],dim = 1) # max distance from ith prototypes set for each eg.
       min_i,_ =  torch.min(sim_mat[i],dim = 1) # min distance from ith prototypes set for each eg.
       max_a.append(max_i)
       min_a.append(min_i)
    
    #finding max and min similarity from closest and second closest classes respectively

    min_sim_from_p_class  = torch.zeros(sim_mat[0].shape[0], 1)
    max_sim_from_s_class = torch.zeros(sim_mat[0].shape[0], 1)

    for i in range(0, sim_mat[0].shape[0]):
      p = p_label[i]
      s = s_label[i]
      min_sim_from_p_class[i] = min_a[p][i]
      max_sim_from_s_class[i] = max_a[s][i]

    conf_mask =  (min_sim_from_p_class > cf_ratio*max_sim_from_s_class)
    return p_label, conf_mask