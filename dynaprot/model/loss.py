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
        true_means, true_covars = batch["dynamics_means"].float()[mask], batch["dynamics_covars"].float()[mask]
        predicted_means, predicted_covars = preds["means"][mask], preds["covars"][mask]

        loss_weights = self.cfg["eval_params"]["loss_weights"]

        loss_dict = dict(
            resi_gaussians=dict(
                mse_means=F.mse_loss(predicted_means, true_means) if loss_weights["resi_gaussians"]["mse_means"] is not None else None,
                kldiv=metrics.kl_divergence_mvn(predicted_means, predicted_covars, true_means, true_covars) if loss_weights["resi_gaussians"]["kldiv"] is not None else None
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
