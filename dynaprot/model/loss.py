import torch
import torch.nn.functional as F
from dynaprot.evaluation import metrics
from torch.distributions.multivariate_normal import MultivariateNormal


class DynaProtLoss(torch.nn.Module):
    """ 
    DynaProt Loss class for aggregating various loss functions/metrics
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def forward(self, preds, batch):
        """ 
        Calls each loss function based on cfg on the predicted and ground truth values.

        Args:
            preds (dict): model predictions for each dynamics type.
            batch (dict): input features and ground truth labels for each dynamics type.

        Returns:
            torch.Tensor: Weighted total loss.
            dict: Dictionary of individual loss components for analysis.
        """
        loss_dict = dict()
        loss_weights = self.cfg["eval_params"]["loss_weights"]

        mask = batch["resi_pad_mask"].bool()
        squaremask = batch["resi_pad_mask"].unsqueeze(1) * batch["resi_pad_mask"].unsqueeze(2) 
        num_res = batch["resi_pad_mask"].sum(dim=1).int()

        true_means = batch["dynamics_means"][mask]

        if "marginal_sqrt" in self.cfg["train_params"]["out_type"]:
            print("predicting sqrtm(marginal)")
            true_covars_sqrt = batch["dynamics_covars_local"][mask]
            predicted_covars_sqrt =  preds["marginal_covars"][mask]
            
            true_covars = batch["dynamics_covars_local"][mask].detach()
            predicted_covars = predicted_covars_sqrt.detach() @ predicted_covars_sqrt.detach()

            true_rmsfs = metrics.compute_rmsf_from_covariances(true_covars).cpu()
            pred_rmsfs = metrics.compute_rmsf_from_covariances(predicted_covars).cpu()
            
            loss_dict["resi_gaussians"] = dict(
                mse_covs=F.mse_loss(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["mse_covs"] is not None else None,
                kldiv=metrics.kl_divergence_mvn(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
                frob_norm=metrics.frobenius_norm_squared(predicted_covars_sqrt - true_covars_sqrt) if loss_weights["resi_gaussians"]["frob_norm"] is not None else None,
                
                log_frob_norm = metrics.log_frobenius_norm(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["log_frob_norm"] is not None else None,
                affine_invariant_dist = metrics.affine_invariant_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["affine_invariant_dist"] is not None else None,
                bures_dist = metrics.bures_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["bures_dist"] is not None else None,
            )
            
            loss_dict["resi_rmsf"] = dict(
                corr_sp = metrics.rmsf_correlation(true_rmsfs, pred_rmsfs, type="spearman") if loss_weights["resi_rmsf"]["corr_sp"] is not None else None,
                corr_pcc = metrics.rmsf_correlation(true_rmsfs, pred_rmsfs, type="pearson") if loss_weights["resi_rmsf"]["corr_pcc"] is not None else None,
            )
            
        elif "marginal" in self.cfg["train_params"]["out_type"]:
            # print("predicting regular marginals")
            true_covars = batch["dynamics_covars_local"][mask]
            predicted_covars =  preds["marginal_covars"][mask]
            true_rmsfs = metrics.compute_rmsf_from_covariances(true_covars.detach()).cpu()
            pred_rmsfs = metrics.compute_rmsf_from_covariances(predicted_covars.detach()).cpu()
            
            loss_dict["resi_gaussians"] = dict(
                # mse_means=F.mse_loss(predicted_means, true_means) if loss_weights["resi_gaussians"]["mse_means"] is not None else None,
                mse_covs=F.mse_loss(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["mse_covs"] is not None else None,
                kldiv=metrics.kl_divergence_mvn(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
                # kldiv=metrics.symmetric_kl(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
                # eigen_penalty = metrics.eigenvalue_penalty(predicted_covars) if loss_weights["resi_gaussians"]["eigen_penalty"] is not None else None,
                frob_norm=(metrics.frobenius_norm(predicted_covars - true_covars)/metrics.frobenius_norm(true_covars)) if loss_weights["resi_gaussians"]["frob_norm"] is not None else None,
                log_frob_norm = metrics.log_frobenius_norm(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["log_frob_norm"] is not None else None,
                affine_invariant_dist = metrics.affine_invariant_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["affine_invariant_dist"] is not None else None,
                bures_dist = metrics.bures_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["bures_dist"] is not None else None,
                # mse_diag= metrics.diagonal_mse_loss(predicted_covars,true_covars) if loss_weights["resi_gaussians"]["mse_diag"] is not None else None,
            )
            
            loss_dict["resi_rmsf"] = dict(
                corr_sp = metrics.rmsf_correlation(true_rmsfs, pred_rmsfs, type="spearman") if loss_weights["resi_rmsf"]["corr_sp"] is not None else None,
                corr_pcc = metrics.rmsf_correlation(true_rmsfs, pred_rmsfs, type="pearson") if loss_weights["resi_rmsf"]["corr_pcc"] is not None else None,
            )
            
        if "joint" in self.cfg["train_params"]["out_type"]:
            padded_true_corrs = batch["dynamics_correlations_nbyncovar"]
            # padded_true_corrs = batch["dynamics_correlations_sum"]
            padded_pred_corrs = preds["joint_covar"]
        
            true_corrs =  [padded_true_corrs[i, :num_res[i], :num_res[i]] for i in range(mask.shape[0])]
            predicted_corrs = [padded_pred_corrs[i, :num_res[i], :num_res[i]] for i in range(mask.shape[0])]
        

            loss_dict["resi_correlations"] = dict(
                log_frob_norm = metrics.log_frobenius_norm_ragged(predicted_corrs, true_corrs)  if loss_weights["resi_correlations"]["log_frob_norm"] is not None else None,
                mse=metrics.mse_ragged(predicted_corrs, true_corrs) if loss_weights["resi_correlations"]["mse"] is not None else None,
                # bures_dist = metrics.bures_distance_ragged(predicted_corrs, true_corrs) if loss_weights["resi_correlations"]["bures_dist"] is not None else None,
            )
 
        # Sum weighted losses
        total_loss = 0
        for dynamics_type in list(loss_dict):
            subdict = loss_dict[dynamics_type]
            subdict = {k: v for k, v in subdict.items() if v is not None}
            loss_dict[dynamics_type] = subdict

            for loss_name, loss in subdict.items():
                weight = loss_weights[dynamics_type][loss_name]
                total_loss += weight * loss
                subdict[loss_name] = loss.detach() if isinstance(loss, torch.Tensor) else loss

        return total_loss, loss_dict
