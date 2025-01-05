import pytorch_lightning as pl
import torch

class EigenvalueLoggingCallback(pl.Callback):
    def __init__(self, log_on_step=True, log_on_epoch=False):
        """
        Callback to log eigenvalue statistics of covariance matrices.

        Args:
            log_on_step (bool): Whether to log eigenvalues during each step.
            log_on_epoch (bool): Whether to log eigenvalues at the end of each epoch.
        """
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch

    def log_eigenvalue_statistics(self, sigma, pl_module, stage):
        """
        Logs the mean of the min and max eigenvalues for a batch of covariance matrices.

        Args:
            sigma (torch.Tensor): Covariance matrices of shape (batch_size, d, d).
            pl_module (pl.LightningModule): The LightningModule to log metrics.
            stage (str): Training or validation stage.
        """
        eigenvalues = torch.linalg.eigvalsh(sigma)  # Shape: (batch_size, d)
        min_eigenvalues = eigenvalues.min(dim=-1)[0]  # Min eigenvalue per matrix
        max_eigenvalues = eigenvalues.max(dim=-1)[0]  # Max eigenvalue per matrix

        # Compute mean statistics
        mean_min_eigenvalue = min_eigenvalues.mean().item()
        mean_max_eigenvalue = max_eigenvalues.mean().item()
        eigenvalue_range = (max_eigenvalues - min_eigenvalues).mean().item()
        condition_numbers = (max_eigenvalues / (min_eigenvalues + 1e-6)).mean().item()

        # Log metrics
        log_key = f"train_losses/resi_gaussians/stability/"
        pl_module.logger.experiment[f"{log_key}mean_min_eigenvalue"].append(mean_min_eigenvalue)
        pl_module.logger.experiment[f"{log_key}mean_max_eigenvalue"].append(mean_max_eigenvalue)
        pl_module.logger.experiment[f"{log_key}eigenvalue_range"].append(eigenvalue_range)
        pl_module.logger.experiment[f"{log_key}condition_numbers"].append(condition_numbers)


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called at the end of each training batch.
        """
        mask = batch["resi_pad_mask"].bool()
        sigma_pred = outputs["covars"][mask]  # Predicted covariance matrices
        self.log_eigenvalue_statistics(sigma_pred, pl_module, stage="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called at the end of each validation batch.
        """
        mask = batch["resi_pad_mask"].bool()
        sigma_pred = outputs["covars"][mask]  # Predicted covariance matrices
        self.log_eigenvalue_statistics(sigma_pred, pl_module, stage="val")
