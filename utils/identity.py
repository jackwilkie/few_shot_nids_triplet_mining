"""
Torch layer to pass through the input without modification. Useful for replacing layers in a model with an identity function.

Created on Fri Jul 21 16:59:46 2023
"""

import torch.nn as nn

def identity(x):
    return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args):
        
        if len(args) == 1:
            return args[0]
        else: 
            return args