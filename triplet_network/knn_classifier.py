"""
KNN Implementation in Pytorch

Created on Fri Jul 21 12:22:06 2023

@author: jack
"""

import torch as T
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from utils.model_eval import model_eval
from typing import Optional, Callable, Tuple, List, Union

# --- distance functions
def euclidean_dist(x1: Tensor, x2: Tensor) -> Tensor:
    return T.cdist(x1, x2, p = 2.0)

def manhattan_dist(x1: Tensor, x2: Tensor) -> Tensor:
    return T.cdist(x1, x2, p=1.0)

def cosine_dist(x1: Tensor,x2: Tensor) -> Tensor:
    return 1 - cosine_sim(x1,x2)

def dist_factory(name: str) -> Callable[[Tensor], Tensor]:
    dist_dict = dict(
        euclidean = euclidean_dist,
        manhattan = manhattan_dist,
        cosine = cosine_dist,
    )
    return dist_dict[name]

# -- similarity functions
def euclidean_sim(x1: Tensor, x2: Tensor) -> Tensor:
    return -1 * euclidean_dist(x1,x2)

def manhattan_sim(x1: Tensor, x2: Tensor) -> Tensor:
    return -1 * manhattan_dist(x1, x2)

def cosine_sim(x1: Tensor, x2: Tensor):
    x1 = F.normalize(x1, dim = -1)
    x2 = F.normalize(x2, dim = -1)
    return x1 @ x2.t()

def sim_factory(name: str) -> Callable[[Tensor], Tensor]:
    sim_dict = dict(
        euclidean = euclidean_sim,
        manhattan = manhattan_sim,
        cosine = cosine_sim,
    )
    return sim_dict[name]

# -- weighting functions
def hard_voting(distances, **kwargs):
    return distances.fill_(1.0)

def soft_voting(distances, **kwrags):
    return distances

def dino_voting(distances, temp, **kwargs):
    return distances.div_(temp).exp_()

def voting_factory(name: str) -> Callable[[Tensor], Tensor]:
    if name is None:
        return None
    
    voting_dict = dict(
        hard = hard_voting,
        soft = soft_voting,
        dino = dino_voting,
    )
    return voting_dict[name]


# --- knn functions
@T.no_grad()
def knn_classifier_(
    x_train: Tensor,
    y_train: Tensor, 
    x_test: Tensor,
    k: int,
    dist_fn: Callable[[Tensor, Tensor], Tensor] = cosine_sim,
    weight_fn: Callable[[Tensor], Tensor] = hard_voting,
    num_chunks: int = 100, 
    num_classes: Optional[int] = None, 
    temp: Optional[float] = None
) -> Tuple[ndarray, ndarray]:
    """Knn classifier on gpu using pytorch

    Args:
        x_train (Tensor): Training data, size B x F, where F is number of features
        y_train (Tensor): Labels for traning data
        x_test (Tensor): Test data
        k (int): number of neighbours to consider when making classifications
        dist_fn (Callable[[Tensor, Tensor], Tensor], optional): Function to calculate distance matrix between two sets of tensors. Defaults to cosine_sim.
        weight_fn (Callable[[Tensor], Tensor], optional): Functions which converts distances into scores for classification. Defaults to hard_voting.
        num_chunks (int, optional): Number of chunks to split test set into to make predictions in batches. Defaults to 100.
        num_classes (Optional[int], optional): Numebr of classes in dataset, infers from y_train if not provided. Defaults to None.
        temp (Optional[float], optional): Temperature parameter used for dino weight function. Defaults to None.

    Returns:
        Tuple[ndarray, ndarray]: Returns tuple of numpy array, first contains test sample scores for each label, second contains label predictions
    """
    if not isinstance(k, int):
        k = int(k)
    
    if isinstance(y_train, ndarray):
        y_train = T.tensor(y_train, dtype = T.int64, device = x_train.device)
    
    # get number of classes from train labels if not provided
    if num_classes is None:
        num_classes = T.unique(y_train).size(0)

    # init lists to store distances and classes
    dist_list = []
    class_list = []

    # calculate chunk size
    num_test_samples = x_test.size(0)
    samples_per_chunk = max(num_test_samples // num_chunks, 1)
    
    retrieval_one_hot = T.zeros(k, num_classes).to(x_train.device)
    chunk_i =  0
    
    for idx in range(0, num_test_samples, samples_per_chunk):
        chunk_i += 1

        # chunk features to compare to all test samples
        features = x_test[idx : min((idx + samples_per_chunk), num_test_samples), :] # use rest of test data if not enough for chunk
        batch_size = features.size(0)
        
        dists = dist_fn(features, x_train) # Returns B x T, T is number of train samples

        # get most similar samples
        # distances, indicies are both B x k. 
        # distances contains distance of k neighbours to test sample, indicies contains index
        distances, indicies = dists.topk(k, largest = True, sorted = True) 
        distances_transform = weight_fn(distances.clone(), temp = temp) if weight_fn is not None else distances.clone()

        canidates = y_train.view(1, -1).expand(batch_size, -1)  # size is B x T. repeats y_train for each test sample
        retrieved_neighbours = T.gather(canidates, 1, indicies)  # B x k. Contains labels of nearest neighbours to each test sample
        
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()  # get zero tensor size B*k x C, C is number of classes
        retrieval_one_hot.scatter_(1, retrieved_neighbours.view(-1,1), 1)  # one hot encoder neighbour class for each neighbour for each sample, size is B*k x C

        # calculate scores as e^d/t, where d is distance and t is temp
        dists = T.mul(retrieval_one_hot.view(batch_size, -1, num_classes), distances_transform.view(batch_size, -1, 1),) # B x k x C, one hot distance of each neighbour
        probs = T.sum(dists, 1) # B x C, summed distance of each test sample to each class
        y_pred = T.argmax(probs, dim = 1)  # get class predictions as class with highest dist (similarity)

        # store predictions
        dist_list.append(probs.cpu().detach().numpy()) # store dists as ndarray
        class_list.append(y_pred.cpu().detach().numpy())  # get class predictions and store as ndarray
    
    # concat predictions from all chunks
    dist_list = np.concatenate(dist_list)
    class_list = np.concatenate(class_list)
    return dist_list, class_list
 
def knn_classifier(
    x_train: Tensor,
    y_train: Tensor, 
    x_test: Tensor,
    y_test,
    k: int,
    dist_fn: str = 'euclidean',
    weight_fn: str = 'hard',
    num_chunks: int = 100, 
    num_classes: Optional[int] = None, 
    temp: Optional[float] = None,
    label_prefix: str = '',
) -> Tuple[ndarray, ndarray]:
    ''' Wrapper to call knn_classifier using string names for callable args
    '''
    dist_fn = sim_factory(dist_fn)
    weight_fn = voting_factory(weight_fn)

    _, y_pred =  knn_classifier_(
        x_train = x_train,
        y_train = y_train, 
        x_test = x_test,
        k = k,
        dist_fn = dist_fn,
        weight_fn = weight_fn,
        num_chunks = num_chunks,
        num_classes = num_classes,
        temp = temp,
    )

    if isinstance(y_test, Tensor):
        y_test = y_test.clone().cpu().detach().numpy()
    
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.clone().cpu().detach().numpy()
        
    return y_test, y_pred


# -- functions for hyperparam fitting
def fit_knn(
    x_train: Tensor,
    y_train: Tensor, 
    x_test: Tensor,
    y_test: Tensor,
    k_vals: Union[List[int], int],
    dist_fn: str = 'euclidean',
    weight_fn: Union[List[str], str] = 'hard',
    num_classes: Optional[int] = None, 
    temp_vals: Optional[Union[List[float], float]] = None,
    label_prefix: str = '',
    device = None,
) -> List[dict]:
    """Function for fitting k value, temp value, and weight function for a knn classifier

    Args:
        x_train (Tensor): Training data
        y_train (Tensor): Training labels
        x_val (Tensor): Validation Data
        y_val (Tensor): Validation labels
        k_vals (Union[List[int], int]): List of values of k to test
        dist_fn (str, optional): List of distance function (in string format) to test. Defaults to 'euclidean'.
        weight_fn (Union[List[str], str], optional): List of voting functions (in string format) to test. Defaults to 'hard'.
        num_classes (Optional[int], optional): Number of classes in data, infers from training data if None. Defaults to None.
        temp_vals (Optional[Union[List[float], float]], optional): List of temperature values to test. Defaults to None.

    Returns:
        List[dict]: Contains list of results for each voting, k, temp combination, Metrics have k_{k}_temp_{t}_w_fn_{weight_fn_str}_ prefix
    """
    
    if not isinstance(x_train, Tensor):
        x_train = T.tensor(x_train, dtype = T.float32, device = device)
        
    if not isinstance(x_test, Tensor):
        x_test = T.tensor(x_test, dtype = T.float32, device = device)
    
    if isinstance(y_train, ndarray):
        y_train = T.tensor(y_train, dtype = T.int64, device = x_train.device)

    if isinstance(y_test, Tensor):
        y_test = y_test.cpu().detach().numpy()

    if num_classes is None:
        num_classes = T.unique(y_train).size(0)

    # init args
    label_prefix = label_prefix if label_prefix is not None else ''
    dist_fn = sim_factory(dist_fn)
    k_vals = k_vals if isinstance(k_vals, list) else [k_vals]
    temp_vals = temp_vals if isinstance(temp_vals, list) else [temp_vals]
    weight_fn = weight_fn if isinstance(weight_fn, list) else [weight_fn]
    print(x_test.size)
    batch_size = x_test.size(0)

    dists = dist_fn(x_test, x_train) # Returns B x T, T is number of train samples
    results: List[dict] = []

    for k in k_vals:
        distances, indicies = dists.topk(k, largest = True, sorted = True) # get neighbours and their distances
        
        canidates = y_train.view(1, -1).expand(batch_size, -1)  # size is B x T. repeats y_train for each test sample
        retrieved_neighbours = T.gather(canidates, 1, indicies)  # B x k. Contains labels of nearest neighbours to each test sample
        
        retrieval_one_hot = T.zeros(k, num_classes).to(x_train.device)
        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()  # get zero tensor size B*k x C, C is number of classes
        retrieval_one_hot.scatter_(1, retrieved_neighbours.view(-1,1), 1)  # one hot encoder neighbour class for each neighbour for each sample, size is B*k x C

        for t_n, t in enumerate(temp_vals):
            for weight_fn_str in weight_fn:
                if weight_fn_str != 'dino' and t_n > 0: #FIXME remove dino hard coding
                    continue
                
                weight_fn_ = voting_factory(weight_fn_str)
                distances_transform = weight_fn_(distances.clone(), temp = t) if weight_fn is not None else distances.clone()
            
                # calculate scores as e^d/t, where d is distance and t is temp
                dists_ = T.mul(retrieval_one_hot.view(batch_size, -1, num_classes), distances_transform.view(batch_size, -1, 1),) # B x k x C, one hot distance of each neighbour
                probs = T.sum(dists_, 1) # B x C, summed distance of each test sample to each class
                y_pred = T.argmax(probs, dim = 1).cpu().detach().numpy()  # get class predictions as class with highest dist (similarity)

                results_dict = model_eval(
                        y_test,
                        y_pred,
                        label = label_prefix,
                        return_class_level= True,
                        return_detection_metrics=True
                )
                
                results_dict['weight_fn'] = weight_fn_str
                results_dict['k'] = k
                results_dict['temp'] = t
                results.append(results_dict)
    
    return results