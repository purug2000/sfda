import time
import numpy as np
from scipy.special import softmax
from .utils import compute_plabel_and_conf
import logging

logger = logging.getLogger(__name__)


def APM_update(prediction_dict, num_labels, flag = "top_k", thresh_arr = None, k = 500, cf_ratio = 1.0): 
    start_time = time.time()
    print(prediction_dict.feat_matrix.shape)
    feat_matrix = prediction_dict.feat_matrix.reshape(-1, 768) # N x 768
    # values = torch.from_numpy(prediction_dict.predictions.reshape(-1,num_labels)) # N x num_labels
    # values = values.reshape(-1,1, num_labels).squeeze()
    # soft = torch.nn.Softmax(dim=1)
    # logits = soft(values).numpy()
    values = prediction_dict.predictions.reshape(-1,num_labels) # N x num_labels
    logits = softmax(values, axis=1) # softmax along each row

    
    if (flag == "top_k"):
      # take the top k scores of each class
        prototypes = []
        num_prototypes = 0
        for i in range(0, num_labels):
            prototypes.append(feat_matrix[np.argpartition(logits[:,i], np.size(logits[:,1])-k)[-k:]])
            num_prototypes += prototypes[i].shape[0]
            logger.info(F"Number of Prototypes for class {i}:{prototypes[i].shape[0]}    ")

        # _,conf_mask =  compute_plabel_and_conf(prototypes, torch.Tensor(feat_matrix), num_labels, cf_ratio)
        # print(F"Conf_mask {conf_mask.float().sum()} / {feat_matrix.shape[0]}")
        # print(F"{conf_mask}")

        return prototypes, num_prototypes

    elif (flag == "Thresholding"):
     # take the scores greater than threshold
        max_index = np.argmax(logits, axis =1) # indices 0 to num_labels-1
        entropy = np.sum(- logits * np.log(logits + 1e-10), axis=1)
        entropy_norm = entropy / np.log(2)

        prototypes = []
        num_prototypes = 0
        for i in range(0, num_labels):
            feat_matrix_i =  feat_matrix[np.where(max_index == i)]
            entropy_i = entropy_norm[np.where(max_index == i)]
            prototypes_i = feat_matrix_i[np.where(entropy_i < thresh_arr[i])] 
            prototypes.append(prototypes_i)
            num_prototypes += prototypes[i].shape[0]
            logger.info(F"Number of Prototypes for class {i}:{prototypes[i].shape[0]}    ")

        #  _,conf_mask =  compute_plabel_and_conf(prototypes, torch.Tensor(feat_matrix), num_labels, cf_ratio)
        # print(F"Conf_mask {conf_mask.float().sum()} / {feat_matrix.shape[0]}")
        # print(F"{conf_mask}")

        return prototypes, num_prototypes

    else:
        print("Wrong flag in APM Update")
        return;