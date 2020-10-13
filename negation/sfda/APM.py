import time
import torch
import numpy as np

def APM_update(prediction_dict, thresh_t = 0.11206, thresh_f = 0.00386): # threshold values optimized on the given data (~74 prototypes in each class)
    start_time = time.time()
    feat_matrix = prediction_dict.feat_matrix # N x 768
    values = torch.from_numpy(prediction_dict.predictions) # N x 2
    soft = torch.nn.Softmax(dim=1)
    logits = soft(values).numpy()
    max_index = np.argmax(logits, axis =1) # indices 0 and 1
    label = np.where(max_index == 0, -1, 1)

    entropy = np.sum(- logits * np.log(logits + 1e-10), axis=1)
    # print(entropy[:20])
    entropy_norm = entropy / np.log(2)
    # print(entropy_norm[:20])

    # divide into 2 classes

    feat_matrix_t = feat_matrix[np.where(label == 1)]
    feat_matrix_f = feat_matrix[np.where(label == -1)]

    # confidence_t = logits[:,1](np.where(label == 1))
    # confidence_f = logits[:,0](np.where(label == -1))

    entropy_t = entropy_norm[np.where(label == 1)]
    entropy_f = entropy_norm[np.where(label == -1)]

    # print("True Negative Entropies", entropy_t[:20])
    # print("False Negative Entropies", entropy_f[:20])

     # Selective threshoding

    prototypes_t = feat_matrix_t[np.where(entropy_t < thresh_t)] # 1
    prototypes_f = feat_matrix_f[np.where(entropy_f < thresh_f)] #-1

    # print("True Negative Prototypes: ", prototypes_t.shape)
    # print("False Negative Prototypes: ", prototypes_f.shape)
    
    num_prototypes = prototypes_t.shape[0] + prototypes_f.shape[0]

    return prototypes_t, prototypes_f, num_prototypes
