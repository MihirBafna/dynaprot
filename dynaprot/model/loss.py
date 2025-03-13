import torch
import torch.nn.functional as F
from dynaprot.evaluation import metrics


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
        
        mask = batch["resi_pad_mask"].bool()
        
        true_means = batch["dynamics_means"].float()[mask]
        # predicted_means = preds["means"][mask]

        # true_covars = batch["dynamics_covars"].float()[mask]  * torch.eye(3).to(preds["covars"])
        # predicted_covars =  preds["covars"][mask] * torch.eye(3).to(preds["covars"])
        true_covars = batch["dynamics_covars"].float()[mask]
        predicted_covars =  preds["covars"][mask]
        predicted_covars_clipped =  preds["covars_clipped"][mask]
        
        loss_weights = self.cfg["eval_params"]["loss_weights"]

        # loss_dict = dict(
        #     resi_gaussians=dict(
        #         # mse_means=F.mse_loss(predicted_means, true_means) if loss_weights["resi_gaussians"]["mse_means"] is not None else None,
        #         mse_covs=F.mse_loss(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["mse_covs"] is not None else None,
        #         kldiv=metrics.kl_divergence_mvn(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
        #         # kldiv=metrics.symmetric_kl(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
        #         # eigen_penalty = metrics.eigenvalue_penalty(predicted_covars) if loss_weights["resi_gaussians"]["eigen_penalty"] is not None else None,
        #         frob_norm=metrics.frobenius_norm(predicted_covars - true_covars)/metrics.frobenius_norm(true_covars) if loss_weights["resi_gaussians"]["frob_norm"] is not None else None,
        #         log_frob_norm = metrics.log_frobenius_norm(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["log_frob_norm"] is not None else None,
        #         affine_invariant_dist = metrics.affine_invariant_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["affine_invariant_dist"] is not None else None,
        #         bures_dist = metrics.bures_distance(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["bures_dist"] is not None else None,
        #     ),
        #     resi_rmsf=dict(
        #         # TODO: Implement RMSF loss calculation
        #     )
        # )
        
        loss_dict = dict(
            resi_gaussians=dict(
                # mse_means=F.mse_loss(predicted_means, true_means) if loss_weights["resi_gaussians"]["mse_means"] is not None else None,
                mse_covs=F.mse_loss(predicted_covars, true_covars) if loss_weights["resi_gaussians"]["mse_covs"] is not None else None,
                kldiv=metrics.kl_divergence_mvn(true_means, predicted_covars_clipped, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
                # kldiv=metrics.symmetric_kl(true_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None,
                # eigen_penalty = metrics.eigenvalue_penalty(predicted_covars) if loss_weights["resi_gaussians"]["eigen_penalty"] is not None else None,
                frob_norm=metrics.frobenius_norm(predicted_covars_clipped - true_covars)/metrics.frobenius_norm(true_covars) if loss_weights["resi_gaussians"]["frob_norm"] is not None else None,
                log_frob_norm = metrics.log_frobenius_norm(predicted_covars_clipped, true_covars) if loss_weights["resi_gaussians"]["log_frob_norm"] is not None else None,
                affine_invariant_dist = metrics.affine_invariant_distance(predicted_covars_clipped, true_covars) if loss_weights["resi_gaussians"]["affine_invariant_dist"] is not None else None,
                bures_dist = metrics.bures_distance(predicted_covars_clipped, true_covars) if loss_weights["resi_gaussians"]["bures_dist"] is not None else None,
            ),
            resi_rmsf=dict(
                # TODO: Implement RMSF loss calculation
            )
        )

        # Sum weighted losses
        total_loss = 0
        for dynamics_type, dynamics_specific_loss_dict in loss_dict.items():
            for loss_name, loss in dynamics_specific_loss_dict.items():
                if loss is None:
                    continue
                else:
                    # print(loss_name,loss)
                    weight = loss_weights[dynamics_type][loss_name]
                    total_loss += weight * loss
                    dynamics_specific_loss_dict[loss_name] = loss.item()

        return total_loss, loss_dict
