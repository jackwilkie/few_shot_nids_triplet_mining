"""
Helper function for processing data batch

Created on Fri Jul 21 16:04:14 2023
"""

import torch as T
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Optional
from utils.model_eval import evaluation_function
from torch import Tensor

def process_batch(batch, device = 'cuda', mixed_precision = False, non_blocking = True):
    x,y = batch
    
    y = y.to(device, non_blocking = non_blocking)
    
    if isinstance(x, tuple) or isinstance(x, list):
        x = x[0]

    if x.device != device: 
        x = x.to(device, non_blocking = non_blocking)
    
    if y.device != device:
        y = y.to(device, non_blocking = non_blocking)

    if mixed_precision:
        x = x.half()
    
    return x, y


@evaluation_function
def extract_features(
    model: nn.Module,
    x_data = None,
    y_data = None,
    dl: DataLoader = None,
    mixed_precision: bool = False,
    non_blocking: bool = True,
    batch: bool = True,
    chunk_size: Optional[int] = 1024,
    move_to_cpu: bool = True,
) -> Tensor:
    
    if dl is not None:
        # use data provided in dataloader if provided
        if batch:
            features, labels = [], []
            for batch in dl:
                x,y = process_batch(batch, device = next(model.parameters()).device, mixed_precision = mixed_precision, non_blocking = non_blocking)
                z = model(x)
                features.append(z.cpu().detach())
                labels.append(y.cpu().detach())
            
            features = T.cat(features, dim = 0)
            labels = T.cat(labels, dim = 0)
            
        else:
            # use data extracted from dataset, kept for legacy reasons
            x,labels = dl.dataset.x_data, dl.dataset.y_data
            features = model(x)
    
    elif x_data is not None:
        if chunk_size is None:
            # use provided data 
            features = model(x_data).cpu().detach()
            labels = y_data.cpu().detach()
        else:
            features, labels = [], []
            
            n_samples = x_data.size(0)
            chunk_i =  0
            
            for idx in range(0, n_samples, chunk_size):
                chunk_i += 1

                # chunk features to compare to all test samples
                x_chunk = x_data[idx : min((idx + chunk_size), n_samples), :] # use rest of test data if not enough for chun
                y_chunk= y_data[idx : min((idx + chunk_size), n_samples)]
                x,y = process_batch((x_chunk,y_chunk), device = next(model.parameters()).device, mixed_precision = mixed_precision, non_blocking = non_blocking)
                z = model(x)
                
                z = z.cpu() if move_to_cpu else z
                y = y.cpu() if move_to_cpu else y
                
                features.append(z.detach())
                labels.append(y.detach())
            
            features = T.cat(features, dim = 0)
            labels = T.cat(labels, dim = 0)
            
                
    else:
        # raise error, either data or dl needed and both not provided
        raise ValueError('ERROR::: Need either data loader of x data')
    
    return features, labels
