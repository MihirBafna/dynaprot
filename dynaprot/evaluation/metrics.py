import torch


# def kl_divergence_mvn(mu1, sigma1, mu2, sigma2):
#     """ KL Divergence between two multivariate Gaussian distributions (batch friendly)

#     Args:
#         mu1     (torch.Tensor): predicted mean vector of shape (*,d)         
#         sigma1  (torch.Tensor): predicted covariance matrix of shape (*,d,d)
#         mu2     (torch.Tensor): ground truth mean vector of shape (*,d)
#         sigma2  (torch.Tensor): ground truth covariance matrix (*,d,d)
        
#     Returns:
#         mean of kl divergence over all residues in batch
#     """
#     d  = mu1.shape[-1]
#     sigma1_det = torch.linalg.det(sigma1)
#     sigma2_det = torch.linalg.det(sigma2)
#     sigma2_inv = torch.linalg.inv(sigma2)
#     batchtrace_s2inv_s1 = torch.einsum("nij,njk->n", sigma2_inv, sigma1)   # take trace of 3x3 cov matrices across all residues across all batches
#     mean_diff = (mu2-mu1)
#     squared_mahalanobis_term = torch.einsum("ni,nij,nj->n",mean_diff, sigma2_inv, mean_diff)
#     kl_div = 0.5 * (torch.log(sigma2_det/sigma1_det + + 1e-6) - d + batchtrace_s2inv_s1 + squared_mahalanobis_term)

#     return kl_div.mean()


def kl_divergence_mvn(mu1, sigma1, mu2, sigma2):
    """
    KL Divergence between two multivariate Gaussian distributions (batch friendly).

    Args:
        mu1     (torch.Tensor): Predicted mean vector of shape (*, d)         
        sigma1  (torch.Tensor): Predicted covariance matrix of shape (*, d, d)
        mu2     (torch.Tensor): Ground truth mean vector of shape (*, d)
        sigma2  (torch.Tensor): Ground truth covariance matrix of shape (*, d, d)
        
    Returns:
        torch.Tensor: Mean KL divergence over all residues in batch.
    """
    d = mu1.shape[-1]

    # Log determinant (more numerically stable than torch.linalg.det)
    sigma1_logdet = torch.linalg.slogdet(sigma1)[1]  # Only the log-determinant
    sigma2_logdet = torch.linalg.slogdet(sigma2)[1]
    # print("Log determinants:", sigma1_logdet,sigma2_logdet)

    # Inverse of sigma2 (add regularization for numerical stability)
    eye = torch.eye(d, device=sigma2.device)
    sigma1_inv = torch.linalg.inv(sigma1 + 1e-6 * eye)

    # Trace term: tr(Sigma2_inv * Sigma1)
    batchtrace_s1inv_s2 = torch.einsum("nij,njk->n", sigma1_inv, sigma2)

    # Mahalanobis distance term: (mu2 - mu1)^T Sigma2_inv (mu2 - mu1)
    mean_diff = mu2 - mu1
    squared_mahalanobis_term = torch.einsum("ni,nij,nj->n", mean_diff, sigma1_inv, mean_diff) if mean_diff.sum() != 0 else 0

    # KL divergence formula KL(P2 || P1)
    kl_div = 0.5 * (
        sigma1_logdet - sigma2_logdet  # log(det(Sigma2) / det(Sigma1))
        - d                           # dimensionality term
        + batchtrace_s1inv_s2         # trace term
        + squared_mahalanobis_term    # Mahalanobis distance
    )
    
    return kl_div.mean()


def symmetric_kl(mu1, sigma1, mu2, sigma2):
    return 0.5 * (kl_divergence_mvn(mu1, sigma1, mu2, sigma2) + kl_divergence_mvn(mu2, sigma2, mu1, sigma1))


def condition_num_penalty(covars, max_condition=100.0, scale_factor=1e-3):
    """
    Penalize covariance matrices with high condition numbers.

    Args:
        covars (torch.Tensor): Covariance matrices of shape (batch_size, num_residues, 3, 3).
        max_condition (float): Maximum allowable condition number.
        scale_factor (float): Scaling factor for the penalty term.

    Returns:
        torch.Tensor: Condition number penalty term.
    """
    eigenvalues, _ = torch.linalg.eigh(covars)
    condition_numbers = eigenvalues[..., -1] / (eigenvalues[..., 0] + 1e-6)
    penalty = torch.clamp(condition_numbers - max_condition, min=0.0).pow(2).sum()

    return scale_factor * penalty


def eigenvalue_penalty(covars, lambda_min=1.0, lambda_max=20, scale_high=1e-3, scale_low=1e-3):
    """
    Penalize eigenvalues that are too small or too large.

    Args:
        covars (torch.Tensor): Covariance matrices of shape (batch_size, num_residues, 3, 3).
        lambda_min (float): Minimum allowable eigenvalue.
        lambda_max (float): Maximum allowable eigenvalue.
        scale_high (float): Scaling factor for the high eigenvalue penalty term.
        scale_low (float): Scaling factor for the low eigenvalue penalty term.

    Returns:
        torch.Tensor: Combined penalty for low and high eigenvalues.
    """
    eigenvalues, _ = torch.linalg.eigh(covars)
    low_eigen_penalty = torch.clamp(lambda_min - eigenvalues, min=0.0).pow(2).sum()
    high_eigen_penalty = torch.clamp(eigenvalues - lambda_max, min=0.0).pow(2).sum()
    return scale_low * low_eigen_penalty + scale_high * high_eigen_penalty


def frobenius_norm(A):
    return torch.sqrt(torch.einsum("nij,nji",A,A)).mean()


def log_frobenius_norm(sigma1, sigma2):
    """
    Compute the Log-Frobenius distance (a Riemannian metric) between two covariance matrices (batch-wise).
    
    Args:
        sigma1 (torch.Tensor): Predicted covariance matrices of shape (batch_size, d, d)
        sigma2 (torch.Tensor): Ground truth covariance matrices of shape (batch_size, d, d)
    
    Returns:
        torch.Tensor: Mean Log-Euclidean distance over the batch.
    """
        
    log_sigma1 = spd_matrix_log(sigma1)
    log_sigma2 = spd_matrix_log(sigma2)
    
    diff = log_sigma1 - log_sigma2

    frobenius_norm = torch.linalg.norm(diff, dim=(-2, -1))  # Norm along the last two dims
    
    return frobenius_norm.mean()


def spd_matrix_log(covariance_matrix):
    """
    Compute the matrix logarithm of a symmetric positive definite matrix.

    Args:
        covariance_matrix (torch.Tensor): Input matrix of shape (..., d, d).
                                        Must be symmetric positive definite.

    Returns:
        torch.Tensor: Logarithm of the input matrix of the same shape (..., d, d).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    log_eigenvalues = torch.log(eigenvalues)
    log_matrix = (eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-1, -2))
    
    return log_matrix


def affine_invariant_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the affine invariant distance distance between two SPD matrices A and B.

    Args:
        A (torch.Tensor): Predicted SPD matrix of shape (..., 3, 3).
        B (torch.Tensor): Ground truth SPD matrix of shape (..., 3, 3).

    Returns:
        torch.Tensor: Geodesic distance between A and B.
    """

    # Perform eigenvalue decomposition of A
    eigvals, eigvecs = torch.linalg.eigh(A)  # A = Q * Lambda * Q^T

    # Compute A^(-1/2) = Q * Lambda^(-1/2) * Q^T
    eigvals_sqrt_inv = torch.diag_embed(1.0 / torch.sqrt(eigvals))
    A_inv_sqrt = eigvecs @ eigvals_sqrt_inv @ eigvecs.transpose(-1, -2)

    # Transform B: A^(-1/2) * B * A^(-1/2)
    transformed_B = A_inv_sqrt @ B @ A_inv_sqrt

    log_transformed_B = spd_matrix_log(transformed_B)

    # Frobenius norm of log-transformed_B
    geodesic_dist = torch.norm(log_transformed_B, dim=(-2, -1))

    return geodesic_dist.mean()


def bures_distance(pred_cov, gt_cov):
    """
    Compute the squared 2-Wasserstein distance between two covariance matrices ignoring means (bures distance).

    Args:
        pred_cov (torch.Tensor): Predicted covariance matrix of shape (batch_size, 3, 3).
        gt_cov (torch.Tensor): Ground truth covariance matrix of shape (batch_size, 3, 3).
    
    Returns:
        torch.Tensor: Wasserstein distance between the covariance matrices.
    """
    
    def matrix_sqrt_eigen(matrix):
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        sqrt_matrix = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)
        return sqrt_matrix
    
    pred_sqrt = matrix_sqrt_eigen(pred_cov)
    
    cross_term = pred_sqrt @ gt_cov @ pred_sqrt.transpose(-1, -2)
    cross_sqrt = matrix_sqrt_eigen(cross_term)
    
    trace_pred = torch.diagonal(pred_cov, dim1=-2, dim2=-1).sum(-1)  # trace(Σ_P)
    trace_gt = torch.diagonal(gt_cov, dim1=-2, dim2=-1).sum(-1)      # trace(Σ_Q)
    trace_cross = torch.diagonal(cross_sqrt, dim1=-2, dim2=-1).sum(-1)  # trace((Σ_P^{1/2} Σ_Q Σ_P^{1/2})^{1/2})
    
    wasserstein_dist = trace_pred + trace_gt - 2 * trace_cross
    return wasserstein_dist.mean()