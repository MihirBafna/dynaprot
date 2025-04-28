import torch
from scipy.stats import pearsonr, spearmanr


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
        sigma1_logdet - sigma2_logdet  # log(det(Sigma1) / det(Sigma2))
        - d                           # dimensionality term
        + batchtrace_s1inv_s2         # trace term
        + squared_mahalanobis_term    # Mahalanobis distance
    )
    
    return kl_div.mean()


def symmetric_kl(mu1, sigma1, mu2, sigma2):
    return 0.5 * (kl_divergence_mvn(mu1, sigma1, mu2, sigma2) + kl_divergence_mvn(mu2, sigma2, mu1, sigma1))


def w2_distance(mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    """
    Compute squared Wasserstein-2 distance between two Gaussians (mu, cov).
    
    Args:
        mu1: Mean of first Gaussian, shape (d,)
        cov1: Covariance of first Gaussian, shape (d, d)
        mu2: Mean of second Gaussian, shape (d,)
        cov2: Covariance of second Gaussian, shape (d, d)

    Returns:
        Scalar tensor: W₂ distance (not squared).
    """
    # Mean term
    mean_diff = torch.linalg.norm(mu1 - mu2)

    # Covariance term
    sqrt_cov2 = matrix_sqrt_eigen(cov2)  # You already defined this

    # Middle term: sqrt( sqrt_cov2 @ cov1 @ sqrt_cov2 )
    middle = sqrt_cov2 @ cov1 @ sqrt_cov2
    sqrt_middle = matrix_sqrt_eigen(middle)

    cov_term = torch.trace(cov1 + cov2 - 2 * sqrt_middle)

    w2 = (mean_diff**2 + cov_term).sqrt()
    return w2


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


def spd_matrix_log(covariance_matrix, eps = 1e-6):
    """
    Compute the matrix logarithm of a symmetric positive definite matrix.

    Args:
        covariance_matrix (torch.Tensor): Input matrix of shape (..., d, d).
                                        Must be symmetric positive definite.

    Returns:
        torch.Tensor: Logarithm of the input matrix of the same shape (..., d, d).
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
    safe_eigenvalues = torch.clamp(eigenvalues, min=eps)

    if torch.any(eigenvalues < eps):
        print("[spd_matrix_log] Warning: Eigenvalues below threshold were clamped.")

    log_eigenvalues = torch.log(safe_eigenvalues)
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


def matrix_sqrt_eigen(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    sqrt_matrix = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(-1, -2)
    return sqrt_matrix
    

def bures_distance(pred_cov, gt_cov):
    """
    Compute the squared 2-Wasserstein distance between two covariance matrices ignoring means (bures distance).

    Args:
        pred_cov (torch.Tensor): Predicted covariance matrix of shape (batch_size, 3, 3).
        gt_cov (torch.Tensor): Ground truth covariance matrix of shape (batch_size, 3, 3).
    
    Returns:
        torch.Tensor: Wasserstein distance between the covariance matrices.
    """
    

    pred_sqrt = matrix_sqrt_eigen(pred_cov)
    
    cross_term = pred_sqrt @ gt_cov @ pred_sqrt.transpose(-1, -2)
    cross_sqrt = matrix_sqrt_eigen(cross_term)
    
    trace_pred = torch.diagonal(pred_cov, dim1=-2, dim2=-1).sum(-1)  # trace(Σ_P)
    trace_gt = torch.diagonal(gt_cov, dim1=-2, dim2=-1).sum(-1)      # trace(Σ_Q)
    trace_cross = torch.diagonal(cross_sqrt, dim1=-2, dim2=-1).sum(-1)  # trace((Σ_P^{1/2} Σ_Q Σ_P^{1/2})^{1/2})
    
    wasserstein_dist = trace_pred + trace_gt - 2 * trace_cross
    return wasserstein_dist.mean()


def bures_distance_ragged(pred_cov_list, gt_cov_list):
    """
    Compute the mean squared 2-Wasserstein (Bures) distance between a ragged list of covariance matrices.

    Args:
        pred_cov_list (list[torch.Tensor]): List of predicted covariance matrices (b, n_i, n_i).
        gt_cov_list (list[torch.Tensor]): List of ground truth covariance matrices (b, n_i, n_i).
    
    Returns:
        torch.Tensor: Mean Bures distance across all proteins in batch.
    """
    assert len(pred_cov_list) == len(gt_cov_list), "Batch size mismatch!"

    distances = torch.stack([bures_distance(pred_cov, gt_cov) for pred_cov, gt_cov in zip(pred_cov_list, gt_cov_list)])

    return distances.mean()


def log_frobenius_norm_ragged(pred_matrix_list, gt_matrix_list):
    """
    Compute the mean log frobenius norm between a ragged list of covariance matrices.

    Args:
        pred_cov_list (list[torch.Tensor]): List of predicted covariance matrices (b, n_i, n_i).
        gt_cov_list (list[torch.Tensor]): List of ground truth covariance matrices (b, n_i, n_i).
    
    Returns:
        torch.Tensor: Mean Bures distance across all proteins in batch.
    """
    assert len(pred_matrix_list) == len(gt_matrix_list), "Batch size mismatch"

    distances = []
    successful_count = 0
    for i, (pred_m, gt_m) in enumerate(zip(pred_matrix_list, gt_matrix_list)):
        try:
            dist = log_frobenius_norm(pred_m, gt_m)
            distances.append(dist)
            successful_count += 1
        except torch._C._LinAlgError as e:
            print(f"Skipping item {i} due to LinAlgError: {e}")
            # Optionally save matrices for debugging
            # torch.save(...)
            continue 

    if successful_count == 0:
        # Handle cases where the entire batch failed
        print("Warning: Entire batch failed log_frobenius_norm calculation.")
        # Return 0 loss but ensure it doesn't propagate gradients if desired
        # Or return a tensor indicating failure
        return torch.tensor(0.0, device=pred_matrix_list[0].device, requires_grad=False)

    mean_loss = torch.stack(distances).mean()
    return mean_loss
    # distances = torch.stack([log_frobenius_norm(pred_cov, gt_cov) for pred_cov, gt_cov in zip(pred_matrix_list, gt_matrix_list)])

    return distances.mean()



def mse_ragged(pred_cov_list, gt_cov_list):
    """
    Compute mse between a ragged list of covariance matrices.

    Args:
        pred_cov_list (list[torch.Tensor]): List of predicted covariance matrices (b, n_i, n_i).
        gt_cov_list (list[torch.Tensor]): List of ground truth covariance matrices (b, n_i, n_i).
    
    Returns:
        torch.Tensor: Mean Bures distance across all proteins in batch.
    """
    assert len(pred_cov_list) == len(gt_cov_list), "Batch size mismatch"

    distances = torch.stack([torch.nn.functional.mse_loss(pred_cov, gt_cov) for pred_cov, gt_cov in zip(pred_cov_list, gt_cov_list)])

    return distances.mean()


def diagonal_mse_loss(pred_covs, true_covs):
    pred_diag = pred_covs.diagonal(dim1=1, dim2=2)  # shape (N, 3)
    true_diag = true_covs.diagonal(dim1=1, dim2=2)  # shape (N, 3)
    return torch.nn.functional.mse_loss(pred_diag, true_diag)


def rmsf_correlation(true_rmsf, pred_rmsf, type="both"):
    if type=="spearman":
        corr, _ = spearmanr(true_rmsf, pred_rmsf)
        return corr
    elif type=="pearson":
        corr, _ = pearsonr(true_rmsf, pred_rmsf)
        return corr
    elif type=="both":
        sp, _ = spearmanr(true_rmsf, pred_rmsf)
        pcc, _ = pearsonr(true_rmsf, pred_rmsf)
        return pcc,sp

def compute_rmsf_from_covariances(cov_matrices):
    traces = cov_matrices.diagonal(dim1=1, dim2=2).sum(dim=1)
    rmsf = torch.sqrt(traces)
    return rmsf


def max_cosine_similarity_each_pc(cov_pred, cov_true, k_pred=4, k_true=4):
    eval_pred, evec_pred = torch.linalg.eigh(cov_pred)  # shape: (3N, 3N)
    eval_true, evec_true = torch.linalg.eigh(cov_true)

    top_pred = evec_pred[:, -k_pred:] 
    top_true = evec_true[:, -k_true:]  

    top_pred = torch.nn.functional.normalize(top_pred, dim=0)
    top_true = torch.nn.functional.normalize(top_true, dim=0)

    cosine_matrix = torch.abs(top_pred.T @ top_true) 

    max_similarities = torch.max(cosine_matrix, dim=1).values 

    return max_similarities



# Ensemble specific metrics

def compute_average_pairwise_rmsd(ensemble_coords: torch.Tensor, subsample: int = 250, seed: int = 137) -> float:
    """
    Compute average C-alpha RMSD across subsampled frames, matching AlphaFlow style.

    Args:
        ensemble_coords (torch.Tensor): (T, N, 3)
        subsample (int): Number of frames to subsample
        seed (int): Random seed

    Returns:
        float: Average pairwise RMSD
    """
    T, N, _ = ensemble_coords.shape

    flat_coords = ensemble_coords.reshape(T, -1)  # (T, 3N)

    np.random.seed(seed)
    idx1 = np.random.choice(T, subsample, replace=True)
    idx2 = np.random.choice(T, subsample, replace=True)

    coords1 = flat_coords[idx1]
    coords2 = flat_coords[idx2]

    dists = torch.cdist(coords1, coords2, p=2)  # (subsample, subsample)

    i, j = torch.triu_indices(subsample, subsample, offset=1)
    pairwise_rmsds = dists[i, j]

    avg_rmsd = (pairwise_rmsds.mean() / np.sqrt(N)) # *10 to get Angstroms

    return avg_rmsd.item()




def compute_validity(calpha_coords: torch.Tensor, clash_threshold: float = 3.0) -> float:
    """
    Compute the validity of an ensemble based on Cα clashes.
    
    Args:
        calpha_coords (torch.Tensor): (T, N, 3) tensor of sampled coordinates
        clash_threshold (float): Distance threshold (in Å) to consider a clash (default: 3.0 Å)

    Returns:
        float: Validity score (fraction of samples without any clashes)
    """
    T, N, _ = calpha_coords.shape

    diffs = calpha_coords.unsqueeze(2) - calpha_coords.unsqueeze(1)  # (T, N, N, 3)
    dists = torch.linalg.norm(diffs, dim=-1)  # (T, N, N)

    eye = torch.eye(N, device=calpha_coords.device).unsqueeze(0)
    dists = dists + eye * 1e6

    has_clash = (dists < clash_threshold).any(dim=(1, 2))  # (T,)

    num_valid = (~has_clash).sum()          # 1 - num steric clashes
    validity = num_valid.item() / T

    return validity


# import torch
# import numpy as np
# from sklearn.decomposition import PCA
# from scipy.optimize import linear_sum_assignment

# def compute_pca_metrics(gt_coords: torch.Tensor, pred_coords: torch.Tensor, k_pc_sim: int = 3, k_pca_proj: int = 2, n_samples: int = 250, seed: int = 137):
#     """
#     Compute PC similarity and PCA-based W2 distance between ground truth and predicted ensembles,
#     matching AlphaFlow evaluation style.
    
#     Args:
#         gt_coords (torch.Tensor): (T1, N, 3)
#         pred_coords (torch.Tensor): (T2, N, 3)
#         k_pc_sim (int): Number of PCs for cosine similarity
#         k_pca_proj (int): Number of PCs for PCA projection
#         n_samples (int): Number of frames to subsample from GT
#         seed (int): Random seed for subsampling
        
#     Returns:
#         Tuple: (pc_similarity_count, pca_wasserstein_distance)
#     """
#     T1, N, _ = gt_coords.shape
#     T2, _, _ = pred_coords.shape

#     device = gt_coords.device

#     # Flatten (T, N, 3) -> (T, 3N)
#     gt_flat = gt_coords.reshape(T1, -1).cpu().numpy()
#     pred_flat = pred_coords.reshape(T2, -1).cpu().numpy()

#     # Center
#     gt_flat -= gt_flat.mean(axis=0, keepdims=True)
#     pred_flat -= pred_flat.mean(axis=0, keepdims=True)

#     # Subsample GT frames
#     np.random.seed(seed)
#     idx = np.random.choice(T1, n_samples, replace=True)
#     gt_sampled = gt_flat[idx]

#     # PCA on GT sampled
#     pca = PCA(n_components=min(gt_sampled.shape))
#     pca.fit(gt_sampled)

#     pc_basis = pca.components_  # (k, 3N)

#     # Cosine similarity of top PCs
#     u_gt = pc_basis[:k_pc_sim]
#     pca_pred = PCA(n_components=min(pred_flat.shape))
#     pca_pred.fit(pred_flat)
#     u_pred = pca_pred.components_[:k_pc_sim]

#     sims = []
#     for i in range(k_pc_sim):
#         sim = np.abs(np.dot(u_gt[i], u_pred[i]) / (np.linalg.norm(u_gt[i]) * np.linalg.norm(u_pred[i])))
#         sims.append(sim)

#     sims = np.array(sims)

#     # Project ensembles onto GT PCA basis
#     proj_gt = gt_flat @ pc_basis[:k_pca_proj].T
#     proj_pred = pred_flat @ pc_basis[:k_pca_proj].T

#     # Compute Wasserstein distance manually using Hungarian matching
#     distmat = np.linalg.norm(proj_gt[:, None, :] - proj_pred[None, :, :], axis=-1)  # (T1, T2)
#     row_ind, col_ind = linear_sum_assignment(distmat)
#     wasserstein = distmat[row_ind, col_ind].mean()

#     # Rescale distances by sqrt(num_atoms)
#     wasserstein /= np.sqrt(N)

#     return (sims > 0.5).sum().item(), wasserstein
