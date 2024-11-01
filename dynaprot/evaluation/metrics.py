import torch
import numpy as np


def kl_divergence_mvn(mu1, sigma1, mu2, sigma2):
    """ KL Divergence between two multivariate Gaussian distributions (batch friendly)

    Args:
        mu1     (torch.Tensor): predicted mean vector of shape (*,d)         
        sigma1  (torch.Tensor): predicted covariance matrix of shape (*,d,d)
        mu2     (torch.Tensor): ground truth mean vector of shape (*,d)
        sigma2  (torch.Tensor): ground truth covariance matrix (*,d,d)
        
    Returns:
        mean of kl divergence over all residues in batch
    """
    d  = mu1.shape[-1]
    sigma1_det = torch.linalg.det(sigma1)
    sigma2_det = torch.linalg.det(sigma2)
    sigma2_inv = torch.linalg.inv(sigma2)
    batchtrace_s2inv_s1 = torch.einsum("nij,njk->n", sigma2_inv, sigma1)   # take trace of 3x3 cov matrices across all residues across all batches
    mean_diff = (mu2-mu1)
    squared_mahalanobis_term = torch.einsum("ni,nij,nj->n",mean_diff, sigma2_inv, mean_diff)
    kl_div = 0.5 * (torch.log(sigma2_det/sigma1_det + + 1e-6) - d + batchtrace_s2inv_s1 + squared_mahalanobis_term)
 
    return kl_div.mean()

