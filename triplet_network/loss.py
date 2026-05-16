"""
Triplet losses in pytorch
Created on Mon Jun 12 17:27:26 2023
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from utils.base_losses import BaseLoss, SupervisedLoss
from utils.identity import Identity

# -------------------- Helper Functions for Triplet Mining --------------------

def euclidean_distance_matrix(embeddings, squared = True):
    '''
    Calculate the distance matric containng distances between all embeddings in tensor

    Parameters
    ----------
    embeddings : Pytorch Tensor
        Tensor containing embeddings for distance calculation
    squared : Bool, optional
        Return squared euclidean distances if true. The default is True.

    Returns
    -------
    d : Pytorch Tensor
        distance matrix where element 2,1 is the distance between the first and second embeddings.

    '''
    #find the dot product matrix for embeddings
    dot_product = T.matmul(embeddings,T.transpose(embeddings, 0, 1))

    #get digonals from dp matrix (squared sum of embedding row)
    sq_sum = T.diagonal(dot_product, 0)

    
    '''
    calculate the distance matrix using: 
        ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    
    shape is shape (batch_size, batch_size)
    '''
    
    d =  T.unsqueeze(sq_sum, 0) - 2.0 * dot_product + T.unsqueeze(sq_sum, 1)
    d = T.clamp(d, min = 0.0)  #prevent negatives due to floating point errors
    
    
    #sqrt distance matrix if squared distance not required
    if not squared:
        mask = T.eq(d, T.tensor(0.0)).float()  #find zero value s
        d = d + (mask * 1e-16)  #replace with small epsilon value

        d = T.sqrt(d)  #sqrt distance matrix
        
        d = d * (1.0 - mask)  #set epsilon values back to 0.0

    return d

def _get_triplet_mask(labels: T.Tensor) -> T.Tensor:
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    
    Parameters
    ----------
    labels : T.Tensor
        Batch labels tensor of shape (batch_size,).

    Returns
    -------
    T.Tensor
        3D boolean mask of shape (batch_size, batch_size, batch_size).
    """
    
    # Check that i, j and k are distinct
    I_eq = T.eye(labels.shape[0], dtype = T.bool, device = labels.device) #get matrix where indices are equal (BxB boolean identity matrix)
    I_neq = T.logical_not(I_eq)  #invert matrix to find where indices are not equal 
    
    z_neq_r = T.unsqueeze(I_neq, 2)  #find where z index does not equal row index
    z_neq_c =T.unsqueeze(I_neq, 1)  #find where z index does not equal column index
    r_neq_c = T.unsqueeze(I_neq, 0)  #find where row index does not equal column index
    z_neq_r, z_neq_c, r_neq_c = T.broadcast_tensors(z_neq_r, z_neq_c, r_neq_c)

    distinct_mask = T.logical_and(T.logical_and(z_neq_r, z_neq_c), r_neq_c)  #get mask showing true for triplets with 2 unique samples

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))  #get 2d matric of showing location of equal labels
    z_eq_r = T.unsqueeze(label_eq, 2)  #convert to column matrices
    z_eq_c = T.unsqueeze(label_eq, 1)  #convert to row matrices
    z_eq_c, z_eq_r = T.broadcast_tensors(z_eq_c, z_eq_r)

    label_mask = T.logical_and(z_eq_r, T.logical_not(z_eq_c))  #create 3d mask of triplets with valid label combinations
    
    # Combine the two masks
    mask = T.logical_and(distinct_mask, label_mask)
    
    return mask


def _get_anchor_positive_triplet_mask(labels: T.Tensor) -> T.Tensor:
    """
    Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    
    Parameters
    ----------
    labels : T.Tensor
        Batch labels tensor of shape (batch_size,).

    Returns
    -------
    T.Tensor
        2D boolean mask of shape (batch_size, batch_size).
    """
    # Check that i and j are distinct
    I_eq = T.eye(labels.shape[0], dtype = T.bool, device = labels.device) #get matrix where indices are equal (BxB boolean identity matrix)
    I_neq = T.logical_not(I_eq)  #invert matrix to find where indices are not equal 

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))

    # Combine the two masks
    mask = T.logical_and(I_neq, labels_eq)

    return mask


def _get_anchor_negative_triplet_mask(labels: T.Tensor) -> T.Tensor:
    """
    Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    
    Parameters
    ----------
    labels : T.Tensor
        Batch labels tensor of shape (batch_size,).

    Returns
    -------
    T.Tensor
        2D boolean mask of shape (batch_size, batch_size).
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))

    mask = T.logical_not(labels_eq)

    return mask


# ----------------------- Online Triplet Mining Losses ------------------------

class OnlineTripletMining(nn.Module):
    """
    Base class for online triplet mining loss functions.
    
    Provides common interface for triplet loss calculation with different mining strategies.
    """
    def __init__(
            self, 
            m: float = 1.0, 
            metric: nn.Module = euclidean_distance_matrix, 
            squared: bool = True
        ):
      
      '''
      Initialize OnlineTripletMining.

      Parameters
      ----------
      m : float, optional
          Margin hyperparameter for loss calculation. Default is 1.0.
      metric : Callable, optional
          Function which returns distance matrix for batch. Default is euclidean_distance_matrix.
      squared : bool, optional
          Square distance if True. Default is True.
      '''
       
      super().__init__()
      
      self.m = m  # margin or radius
      self.metric_matrix = metric
      self.squared = squared
      self.fraction_positive_triplets = 0

    def forward(self, embeddings: T.Tensor, target: T.Tensor):
        """
        Forward pass. Must be implemented by subclasses.

        Parameters
        ----------
        embeddings : T.Tensor
            Embedding tensor of shape (batch_size, embedding_dim).
        target : T.Tensor
            Labels tensor of shape (batch_size,).

        Raises
        ------
        NotImplementedError
            This method must be overridden by subclasses.
        """
        raise NotImplementedError('ERROR: PLEASE USE CHILD CLASS')
    
    def get_fraction_pos(self) -> float:
        """
        Get fraction of positive triplets.

        Returns
        -------
        float
            Fraction of positive triplets in the last forward pass.
        """
        return self.fraction_positive_triplets
        
class BatchAllTripletLoss(OnlineTripletMining):
    """
    Batch all triplet loss mining strategy.
    
    Generates all valid triplets and averages the loss over positive ones.
    """
    
    def get_class_hist(self):
        return self.class_hist
      
    def forward(self, embeddings: T.Tensor, target: T.Tensor):
        """Build the triplet loss over a batch of embeddings.

        We generate all the valid triplets and average the loss over the positive ones.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """

        # Get the pairwise distance matrix
        d_matrix = self.metric_matrix(embeddings, squared = self.squared )
        
        #compute BxBxB distance matrices for anchor to positve and anchor to negative samples
        d_ap_matrix = T.unsqueeze(d_matrix, 2) 
        d_an_matrix = T.unsqueeze(d_matrix, 1)
        
        #return d_matrix
        # Compute BxBxB Triplet loss matrix
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (B, B, 1)
        # and the 2nd (B, 1, B)
        triplet_loss = d_ap_matrix - d_an_matrix + self.m
        
        # zero loss for invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        
        mask = _get_triplet_mask(target).float()  #get mask for valid triplets
        
        triplet_loss = mask * triplet_loss  #apply mask to triplet loss matrix
        
        triplet_loss = T.clamp(triplet_loss, min=0.0)  #remove negative losses (easy triplets)

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = T.greater(triplet_loss,1e-16).float()
        num_positive_triplets = T.sum(valid_triplets)
        num_valid_triplets = T.sum(mask)
        self.fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = T.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss

    
#semi hard triplet mining
class BatchAllSemiHardTripletLoss(OnlineTripletMining):
    """
    Batch all semi-hard triplet loss mining strategy.
    
    Selects semi-hard triplets where negative distance is greater than positive distance.
    """
    def forward(self, embeddings: T.Tensor, target: T.Tensor):
        """Build the triplet loss over a batch of embeddings for semi hard triplets.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        d_matrix = self.metric_matrix(embeddings, squared= self.squared)
        
        mask_triplet = _get_triplet_mask(target).float()  #get mask for valid triplets
        
        d_ap = T.unsqueeze(d_matrix, 2)  # col matrices of anchor positive distances

        d_an = T.unsqueeze(d_matrix, 1)  # row matrices of anchor positive distances

        mask_sh = T.gt(d_an,d_ap).float()  #mask filtering out non semihard triplets

        #calculate triplet loss
        triplet_loss = d_ap - d_an + self.m  #find triplet loss for all triplets
        triplet_loss = triplet_loss * mask_triplet * mask_sh  #apply masks to filter invalid triplets
        triplet_loss = T.mean(T.clamp(triplet_loss, min=0.0))  #remove negative losses (easy triplets) and find mean loss     
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = T.greater(triplet_loss,1e-16).float()
        num_positive_triplets = T.sum(valid_triplets)
        
        num_valid_triplets = T.sum(mask_triplet)
        self.fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
        
        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = T.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        
        return triplet_loss  #return triplet loss

    def get_fraction_pos(self) -> float:
        """
        Get fraction of positive triplets.

        Returns
        -------
        float
            Fraction of positive triplets in the last forward pass.
        """
        return self.fraction_positive_triplets


#semi hard triplet mining
class BatchSemiHardTripletLoss(OnlineTripletMining):
    """
    Batch semi-hard triplet loss mining strategy.
    
    For each anchor, selects the hardest semi-hard triplet.
    """
    def forward(self, embeddings: T.Tensor, target: T.Tensor):
        """Build the triplet loss over a batch of embeddings for semi hard triplets.
        For each anchor, we get the positive and negative combination with the greatest loss where D_an > D_ap
        .
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        d_matrix = self.metric_matrix(embeddings, squared= self.squared)
        
        mask_triplet = _get_triplet_mask(target).float()  #get mask for valid triplets
        
        d_ap = T.unsqueeze(d_matrix, 2)  # col matrices of anchor positive distances
        d_an = T.unsqueeze(d_matrix, 1)  # row matrices of anchor positive distances
        mask_sh = T.gt(d_an,d_ap).float()  #mask filtering out non semihard triplets
        
        #calculate triplet loss
        triplet_loss = d_ap - d_an + self.m  #find triplet loss for all triplets
        triplet_loss = triplet_loss * mask_triplet * mask_sh  #apply masks to filter invalid triplets
        triplet_loss = T.amax(triplet_loss, dim=(1, 2))  #get hardest semihard triplet for each anchor
        triplet_loss = T.clamp(triplet_loss, min = 0.0)
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = T.greater(triplet_loss,1e-16).float()
        num_positive_triplets = T.sum(valid_triplets)
        self.fraction_positive_triplets = num_positive_triplets / (embeddings.size(0))
        
        triplet_loss = T.mean(triplet_loss)  #remove negative losses (easy triplets) and find mean loss

        return triplet_loss  #return triplet loss


#Online Triplet mining loss using batch hard strategy
class BatchHardTripletLoss(OnlineTripletMining):
    """
    Batch hard triplet loss mining strategy.
    
    For each anchor, selects the hardest positive and hardest negative samples.
    """
    def forward(self, embeddings: T.Tensor, target: T.Tensor):
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        # Get the pairwise distance matrix
        d_matrix = self.metric_matrix(embeddings, squared= self.squared)


        # Get hardest positive sample for each anchor
        mask_ap = _get_anchor_positive_triplet_mask(target).float()  #get mask for positive samples for each anchor
        
        #print(d_matrix)
        #print(mask_ap)
        d_ap = mask_ap * d_matrix  # zero distance for samples with labels which do not match anchor
        d_hard_p = T.max(d_ap, dim = 1, keepdim = True).values  # get maximum distance between anchor and positive samples, for each anchor
        
                            
        # Get hardest negative sample for each anchor
        mask_an = _get_anchor_negative_triplet_mask(target).float()
        d_max_an = T.max(d_matrix, dim=1, keepdim=True).values  #get max distance betweene each anchor and negative sample
        d_an = d_matrix + (d_max_an * (1.0 - mask_an))  #add max distance to invalid pairs to ensure they are not selected for triplet (cannot 0 as they would be selected)

        d_hard_n = T.min(d_an, dim = 1, keepdim = True).values

        triplet_loss = T.clamp(d_hard_p - d_hard_n + self.m, min = 0.0)  # calculate triplet loss using hardest samples for each anchor

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = T.greater(triplet_loss,1e-16).float()
        num_positive_triplets = T.sum(valid_triplets)
        self.fraction_positive_triplets = num_positive_triplets / (embeddings.size(0))
        
        triplet_loss = T.mean(triplet_loss)  # get final mean triplet loss

        return triplet_loss

class TripletLossWrapper(SupervisedLoss):
    """
    Wrapper for triplet loss with optional online probe head for classification.
    """
    def __init__(self, loss, *args, online_probe: bool = True, batch_aug = Identity(), **kwargs):
       self.online_probe = online_probe
       super().__init__(* args, loss = loss, batch_aug = batch_aug, **kwargs)

    def calc_metric(self, logits: T.Tensor, y_true: T.Tensor) -> float:
        """
        Calculate metric as 1 - fraction of positive triplets.

        Parameters
        ----------
        logits : T.Tensor
            Logits from model.
        y_true : T.Tensor
            True labels.

        Returns
        -------
        float
            Calculated metric.
        """
        return 1 - self.loss.get_fraction_pos()

    def feed_model(self, model, x: T.Tensor, y: T.Tensor, mixed_precision):
        """
        Extract features using model.

        Parameters
        ----------
        model
            The model.
        x : T.Tensor
            Input tensor.
        y : T.Tensor
            Labels tensor.
        mixed_precision
            Mixed precision setting.

        Returns
        -------
        tuple
            Features, labels, and true labels.
        """
        x, y, y_true = self.get_model_input(model, x, y, mixed_precision)
        x = model.forward_features(x)
        return x, y, y_true

    def forward(self, model, x: T.Tensor, y: T.Tensor, mixed_precision, training):
        """
        Forward pass with optional online probe.

        Parameters
        ----------
        model
            The model.
        x : T.Tensor
            Input tensor.
        y : T.Tensor
            Labels tensor.
        mixed_precision
            Mixed precision setting.
        training
            Training mode flag.

        Returns
        -------
        tuple
            Loss and metric.
        """
        logits, y, y_true = self.feed_model(model, x, y, mixed_precision)
        loss = self.loss(model.proj(logits), y)
        
        if self.online_probe:
            logits_ce = logits.clone().detach()
            logits_ce.requires_grad_(True)
            
            logits_ce = model.probe(logits_ce)
            ce_loss = F.cross_entropy(logits_ce, y) 
            loss = (loss, ce_loss)

            if self.cache_labels_:
                self.cache_labels(logits_ce, y_true.to(logits.device), training = model.training)

        return loss, self.calc_metric(logits.clone().detach(), y_true)

def batch_all_triplet_loss(*args, **kwargs):
   return TripletLossWrapper(BatchAllTripletLoss, *args, **kwargs)

def batch_all_semi_hard_triplet_loss(*args, **kwargs):
   return TripletLossWrapper(BatchAllSemiHardTripletLoss, *args, **kwargs)

def batch_semi_hard_triplet_loss(*args, **kwargs):
    return TripletLossWrapper(BatchSemiHardTripletLoss, *args, **kwargs)

def batch_hard_triplet_loss(*args, **kwargs):
   return TripletLossWrapper(BatchHardTripletLoss, *args, **kwargs)

# ----------------------- Offline Triplet Mining Losses ------------------------

class OfflineTripletLoss(nn.Module):
    """
    Offline triplet loss for pre-mined triplets.
    
    Expects triplet samples as input and computes pairwise distances.
    """
    def __init__(self, m: float = 1.0, squared: bool = True):
        """
        Initialize OfflineTripletLoss.

        Parameters
        ----------
        m : float, optional
            Margin for triplet loss. Default is 1.0.
        squared : bool, optional
            Whether to square distances. Default is True.
        """
        super().__init__()
        self.m = m
        self.squared = squared

    def forward(self, x1: T.Tensor, x2: T.Tensor, x3: T.Tensor) -> T.Tensor:
        """
        Compute offline triplet loss.

        Parameters
        ----------
        x1 : T.Tensor
            Anchor embeddings.
        x2 : T.Tensor
            Positive embeddings.
        x3 : T.Tensor
            Negative embeddings.

        Returns
        -------
        T.Tensor
            Triplet loss value.
        """
        dap = F.pairwise_distance(x1, x2)
        dan = F.pairwise_distance(x1, x3)

        if self.squared:
            dap = T.pow(dap, 2)
            dan = T.pow(dan, 2)

        return T.mean(F.relu(dap - dan + self.m))
    

class OfflineTripletLossWrapper(BaseLoss):
    """
    Wrapper for offline triplet loss.
    """
    def forward(self, model, x: T.Tensor, y: T.Tensor, mixed_precision, training):
        """
        Forward pass for offline triplet loss.

        Parameters
        ----------
        model
            The model.
        x : T.Tensor
            Triplet samples tensor of shape (batch_size, 3, embedding_dim).
        y : T.Tensor
            Labels tensor.
        mixed_precision
            Mixed precision setting.
        training
            Training mode flag.

        Returns
        -------
        tuple
            Loss and metric.
        """
        x1, x2, x3 = x[:,0,:], x[:,1,:], x[:,2,:]
        x1 = model(x1)
        x2 = model(x2)
        x3 = model(x3)

        return self.loss(x1, x2, x3), self.calc_metric(x1.clone().detach(), y)
    
def get_offline_triplet_loss(m: float, squared: bool = True):
    """
    Create offline triplet loss wrapper.

    Parameters
    ----------
    m : float
        Margin for triplet loss.
    squared : bool, optional
        Whether to square distances. Default is True.

    Returns
    -------
    OfflineTripletLossWrapper
        Offline triplet loss wrapper instance.
    """
    return OfflineTripletLossWrapper(OfflineTripletLoss, m = m, squared = squared)