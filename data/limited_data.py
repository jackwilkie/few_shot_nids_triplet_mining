"""
Functions for limited data evaluation

Created on Mon Jun 12 18:43:42 2023
"""

import numpy as np
from numpy import ndarray
import random 
from torch import Tensor
import copy
from typing import Optional, Tuple

#----------------------------- Limited Training Data Functions ----------------
def rng_choice(
    indicies, 
    n_samples, 
    generator = None,
    replace = False,
):
    rng = generator or random.Random()

    if replace:
        return rng.choices(indicies, k = n_samples)
    else:
        return rng.sample(indicies, k = n_samples)


def get_limited_train_set(
    x_data: ndarray, 
    y_data: ndarray, 
    benign_samples: int = None, 
    attack_samples: int = None, 
    seed: Optional[int] = None, 
    set_num: int= 0, 
    replacement: bool = False,
) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Get training and testing data sets to train and evaluate models under limited data conditions. Gets desired number of samples per class
    in training set with the remainder of data being used for testing.


    Parameters
    ----------
    x_data : ndarray
        Data to be split.
    y_data : ndarray
        Class labels of data to be split.
    benign_samples : Int, optional
        Specify number of benign samples required for train split, setting to less than 1 will use a ratio of benign samples each split.
    mal_samples : Int, optional
        Specify number of attack samples required for train split, setting to less than 1 will use a ratio of benign samples each split. The default is 0.
    seed : Int, optional
        Seed to use for random data sampling.
    set_num : Int, optional
        Number of traning set to return, allows for non-overlapping train sets to be samples (note: only last set is returned). The default is 0.
    
    Returns
    -------
    x_train : ndarray
        Training data set.
    y_train : ndarray
        Labels for training data set.
    x_test : ndarray
        Testing data set.
    y_test: ndarray
        Labels for test set.

    """

    # -- init function
    # return original data is sample numbers are not provided
    if attack_samples is None and benign_samples is None:
        return x_data, y_data, None, None
    
    # copy y_data to prevent change
    y_data_ = copy.deepcopy(y_data)
    
    # convert labels to numpy array if tensor
    if isinstance(y_data_, Tensor):
        y_data = y_data_.cpu().detach().numpy()
        
    #set seed for random number generators
    rng = random.Random(seed)

    #initalise pool of availble data (not yet used in a train set)
    i_pool = np.array([x for x in range(len(y_data))])
    y_pool = y_data
    
    # -- get number of samples needed from each class
    num_class_samples = {}
    
    for c in np.unique(y_data):
        class_size = np.count_nonzero(y_data == c)
        num_samples = benign_samples if c == 0 else attack_samples
        num_samples = num_samples or class_size  # get whole class if num samples is None
        num_samples = num_samples if num_samples > 1 else (class_size * num_samples) // 1 # use num samples as ratio if less than one
        num_class_samples[c] = num_samples  # store number of samples to samples from class
    
    # -- randomly sample indicies from each class        
    for i in range(set_num + 1):  
        
        #initalise train data for set
        set_indicies = []
        
        #iterate over each class in dataset
        for c in np.unique(y_data):
            
            class_indicies = np.argwhere(y_pool == c).flatten()  #get indicies of samples of current class
            class_size = len(class_indicies)
            
            if num_class_samples[c] > class_size:  # need more samples than class size
                if replacement:
                    extra_samples = num_class_samples[c] - class_size
                    resampled_indicies = rng_choice(class_indicies.tolist(), extra_samples, replace=True, generator=rng)
                    class_train_indicies = np.concatenate(
                        [class_indicies, np.array(resampled_indicies)]
                    )
                else:
                    raise ValueError('Required more samples than class size')
            else:  # enough class samples
                class_train_indicies = (
                    np.array(rng_choice(class_indicies.tolist(), num_class_samples[c], replace=False, generator=rng))
                    if class_size > num_class_samples[c]
                    else class_indicies
                )
            set_indicies.extend(class_train_indicies.flatten())
            if i != set_num and not replacement:
                #remove uesd samples from available data poule
                i_pool = np.delete(i_pool,class_train_indicies, axis = 0)
                y_pool = np.delete(y_pool,class_train_indicies, axis = 0)

        get_indicies = [i_pool[i] for i in set_indicies]
    
    
    # -- get samples from indicies
    x_train = x_data[get_indicies]
    y_train = y_data_[get_indicies]
    
    # test set is all data not used in chosen train set
    mask = np.ones(len(x_data), dtype=bool)  # Initially set all elements to True
    mask[get_indicies] = False
    x_test = x_data[mask]
    y_test = y_data_[mask]
                    
    return x_train, y_train, x_test, y_test  #reutrn train and test data


def iterate_limited_train_set(x_data, y_data, benign_samples = 0, attack_samples = 0, seed = 385929325, sets = 20):
    """
    Get training and testing data sets to train and evaluate models under limited data conditions. Gets desired number of samples per class
    in training set with the remainder of data being used for testing. 
    
    generator equivalent of get_limited_train_set generating all 20 train sets


    Parameters
    ----------
    x_data : Numpy Array
        Data to be split.
    y_data : Numpy Array
        Class labels of data to be split.
    benign_samples : Int, optional
        Specify number of benign samples required for train split, setting to zero will use a ratio of benign samples each split. The default is 0.
    mal_samples : Int, optional
        Specify number of attack samples required for train split, setting to zero will use a ratio of benign samples each split. The default is 0.
    ratio : Float, optional
        Ratio of data samples to use for each class in train split if fixed number is not specified. The default is 0.1.
    seed : Int, optional
        Seed to use for random data sampling. The default is 385929325.
    set_num : Int, optional
        Number of traning set to return, allows for non-overlapping train sets to be samples (note: only last set is returned). The default is 0.
    verbose : Bool, optional
        Prints progress messages if True. The default is False.

    Returns
    -------
    x_train : Numpy Array
        Training data set.
    y_train : Numpy Array
        Labels for training data set.
    x_test : Numpy Array
        Testing data set.
    y_test: Numpy Array
        Labels for test set.

    """
        
    #set seed for random number generators
    rng = random.Random(seed)

    #initalise pool of availble data (not yet used in a train set)
    i_pool = np.array([x for x in range(len(y_data))])
    y_pool = y_data
    
    #create dict of number of samples per set for each class
    num_class_samples = {}
    
    for c in np.unique(y_data):
        if c == 0:
            num_class_samples[c] = benign_samples
            
        else:
            num_class_samples[c]  = attack_samples
    
    #iteratively remove training data from available data pools until the desired set is sampled
    for i in range(sets):  

        #initalise train data for set
        set_indicies = []

        #iterate over each class in dataset
        for c in np.unique(y_data):

            class_indicies = np.argwhere(y_pool == c).flatten().tolist()  #get indicies of samples of current class
            
            #print(i)
            
            class_train_indicies = np.array(rng.sample(class_indicies, num_class_samples[c])) #sample class indicie for trianing set

            set_indicies.extend(class_train_indicies.flatten())

        get_indicies = [i_pool[i] for i in set_indicies]

        #get train set from train indicies
        x_train = np.array([x_data[i] for i in get_indicies])
        y_train = np.array([y_data[i] for i in get_indicies])

        # test set is all data not used in chosen train set
        x_test = np.delete(x_data,get_indicies, axis = 0)
        y_test = np.delete(y_data,get_indicies, axis = 0)
        
        i_pool = np.delete(i_pool,class_train_indicies, axis = 0)
        y_pool = np.delete(y_pool,class_train_indicies, axis = 0)
            
        yield x_train, y_train, x_test, y_test, i