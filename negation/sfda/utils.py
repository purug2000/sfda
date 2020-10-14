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