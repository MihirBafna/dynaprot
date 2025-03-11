import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from openfold.model.structure_module import InvariantPointAttention
from dynaprot.model.operators.invariant_point_attention import IPABlock
from openfold.utils.rigid_utils import  Rigid
from dynaprot.model.loss import DynaProtLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR


class DynaProt(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_residues = cfg["data_config"]["max_num_residues"]
        self.num_ipa_blocks = cfg["model_params"]["num_ipa_blocks"]
        self.d_model = cfg["model_params"]["d_model"]
        self.lr = cfg["train_params"]["learning_rate"]
        self.warmup_steps = cfg["train_params"]["warmup_steps"]
        self.total_steps = cfg["train_params"]["total_steps"]

        # Embedding layers for sequence and pairwise features
        self.sequence_embedding = nn.Embedding(21, self.d_model)  # 21 amino acid types
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_residues, self.d_model))

        # IPA layers
        self.ipa_blocks = nn.ModuleList([InvariantPointAttention(c_s=self.d_model,c_z=self.d_model,c_hidden=16,no_heads=4,no_qk_points=4,no_v_points=8) for _ in range(self.num_ipa_blocks)])
        # self.ipa_blocks = nn.ModuleList([IPABlock(dim=self.d_model, post_attn_dropout=0.2,post_ff_dropout=0.2, heads=4, point_key_dim = 4, point_value_dim = 8,require_pairwise_repr=False, post_norm=True) for _ in range(self.num_ipa_blocks)])

        self.dropout = nn.Dropout(0.2)

        self.covars_predictor = nn.Linear(self.d_model, 6)  # Predict lower diagonal matrix L (cholesky decomposition) to ensure symmetric psd Σ = LL^T
        
        # self.global_corr_predictor =
        
        # for stability
        self.epsilon = 1e-6      # regularization to ensure Σ is positive definite and other stability issues
        
        # Initialize DynaProtLoss which handles all separate loss functions
        self.loss = DynaProtLoss(self.cfg)
        
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
        warmup_steps = self.warmup_steps
        total_steps = self.total_steps 
        cosine_steps = total_steps - warmup_steps
        
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }


    def forward(self, sequence, frames, mask):  # what to do with mask at inference?
        seq_emb = self.sequence_embedding(sequence)  # Shape: (batch_size, num_residues, d_model)

        pos_emb = self.position_embedding.expand_as(seq_emb)  # Shape: (batch_size, num_residues, d_model)
        residue_features = seq_emb + pos_emb

        # Initialize pairwise embeddings (e.g., could be contact map or learned, right now is zero)
        pairwise_embeddings = self.init_pairwise_features(sequence).to(residue_features) 

        # IPA blocks
        for ipa_block in self.ipa_blocks:
            # residue_features = ipa_block(x=residue_features, rotations=frames.get_rots().get_rot_mats(),translations=frames.get_trans(), mask=mask.bool())
            residue_features = ipa_block(residue_features, pairwise_embeddings, frames, mask)
            residue_features = self.dropout(residue_features)

        preds = dict(
            # means = self.pred_mean(residue_features),      # Shape: (batch_size, num_residues, 3)
            covars = self.pred_covars(residue_features)    # Shape: (batch_size, num_residues, 3, 3)
        )

        return preds


    def init_pairwise_features(self, sequence):
        pairwise_features = torch.zeros(sequence.shape[0], self.num_residues, self.num_residues, self.d_model)
        return pairwise_features


    def pred_mean(self, residue_features):
        return  self.mean_predictor(residue_features)
    
    
    def pred_covars(self, residue_features, lambda_min=0.5, lambda_max=10, soft_clip=True):
        """
        Predict covariance matrices (by predicting cholesky factor).

        Args:
            residue_features (torch.Tensor): Input residue features of shape (batch_size, num_residues, feature_dim).
            lambda_min (float): Minimum eigenvalue for clipping.
            lambda_max (float): Maximum eigenvalue for clipping.

        Returns:
            torch.Tensor: Stabilized covariance matrices of shape (batch_size, num_residues, 3, 3).
        """
        
        L_entries = self.covars_predictor(residue_features) # Predict the 6 L entries

        L = torch.zeros(
            residue_features.shape[0], self.num_residues, 3, 3, device=L_entries.device
        )
        i = 0
        for c in range(3):
            for r in range(c, 3):
                if r == c:
                    L[:, :, r, c] = F.softplus(L_entries[:, :, i])  # Ensure positive variances
                else:
                    L[:, :, r, c] = L_entries[:, :, i]
                i += 1
        covars = L @ L.transpose(-1, -2)
        return covars
    
    
    def pred_covars_direct(self, residue_features, lambda_min=0.5, lambda_max=10, soft_clip=True):
        """
        Predict covariance matrices directly.

        Args:
            residue_features (torch.Tensor): Input residue features of shape (batch_size, num_residues, feature_dim).
            lambda_min (float): Minimum eigenvalue for clipping.
            lambda_max (float): Maximum eigenvalue for clipping.

        Returns:
            torch.Tensor: Stabilized covariance matrices of shape (batch_size, num_residues, 3, 3).
        """
        
        covar_entries = self.covars_predictor(residue_features) # Predict the 6 L entries

        covars = torch.zeros(
            residue_features.shape[0], self.num_residues, 3, 3, device=covar_entries.device
        )
        i = 0
        for c in range(3):
            for r in range(c, 3):
                if r == c:
                    covars[:, :, r, c] = F.softplus(covar_entries[:, :, i])  # Ensure positive variances
                else:
                    covars[:, :, r, c] = covar_entries[:, :, i]
                    covars[:, :, c, r] = covar_entries[:, :, i]      # preserve symmetry

                i += 1
        return covars
    
    
    # def pred_covars(self, residue_features, lambda_min=0.5, lambda_max=10, soft_clip=False):
    #     """
    #     Predict covariance matrices and apply eigenvalue clipping.

    #     Args:
    #         residue_features (torch.Tensor): Input residue features of shape (batch_size, num_residues, feature_dim).
    #         lambda_min (float): Minimum eigenvalue for clipping.
    #         lambda_max (float): Maximum eigenvalue for clipping.

    #     Returns:
    #         torch.Tensor: Stabilized covariance matrices of shape (batch_size, num_residues, 3, 3).
    #     """
        
    #     covar_entries = self.covars_predictor(residue_features) # Predict the 6 L entries (Var X, Var Y, Var Z, Corr XY, Corr XZ, Corr)

    #     covars = torch.zeros(
    #         residue_features.shape[0], self.num_residues, 3, 3, device=covar_entries.device
    #     )
    #     i = 0
    #     for c in range(3):
    #         for r in range(c, 3):
    #             if r == c:
    #                 covars[:, :, r, c] = F.softplus(covar_entries[:, :, i])  # Ensure positive variances
    #             else:
    #                 covars[:, :, r, c] = F.tanh(covar_entries[:, :, i])
    #                 covars[:, :, c, r] = F.tanh(covar_entries[:, :, i])

    #             i += 1

    #     # covars = covars + self.epsilon * torch.eye(3, device=covars.device)
    #     # Apply soft or hard clipping
    #     # eigenvalues, eigenvectors = torch.linalg.eigh(covars)  # Shape: (batch_size, num_residues, 3), (batch_size, num_residues, 3, 3)
    #     # if soft_clip:
    #     #     eigenvalues_clipped = torch.where(
    #     #         eigenvalues > lambda_max,
    #     #         lambda_max + torch.log(1 + eigenvalues - lambda_max),
    #     #         torch.where(
    #     #             eigenvalues < lambda_min,
    #     #             lambda_min - torch.log(1 + lambda_min - eigenvalues),
    #     #             eigenvalues,
    #     #         ),
    #     #     )
    #     # else:
    #     #     eigenvalues_clipped = torch.clamp(eigenvalues, min=lambda_min, max=lambda_max)

    #     # # Reconstruct covariance matrices with clipped eigenvalues
    #     # covars = (
    #     #     eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.transpose(-1, -2)
    #     # )
    #     return covars

    
    def on_before_optimizer_step(self, optimizer):
        parameters = self.parameters()
        clip_grad_norm_(parameters, self.cfg["train_params"]["grad_clip_norm"])
        
    
    
    def training_step(self, batch, batch_idx):
        
        preds = self(batch["aatype"].argmax(dim=-1), Rigid.from_tensor_4x4(batch["frames"]), batch["resi_pad_mask"])

        total_loss, loss_dict = self.loss(preds, batch)
        
        # if self.trainer.is_global_zero:
        for dynamics_type, losses in loss_dict.items():
            for loss_name, loss_value in losses.items():
                log_key = f"{dynamics_type}/{loss_name}"
                self.log(log_key,loss_value,on_epoch=False,on_step=True, sync_dist=True)
                # if self.logger is not None:
                # self.logger.experiment[log_key].append(loss_value, step=self.global_step)
        # Log the loss and return
        # self.log_dict({"train_losses/total_loss":total_loss})
        # if self.logger is not None:
        # self.logger.experiment["train_losses/total_loss"].append(total_loss, step=self.global_step)
        self.log("total_loss",total_loss,on_epoch=False,on_step=True, sync_dist=True)
        
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate",current_lr,on_epoch=False,on_step=True, sync_dist=True)
        # self.logger.experiment["train/learning_rate"].append(current_lr, step=self.global_step)
    
        return total_loss


    def validation_step(self, batch, batch_idx):
        preds = self(batch["aatype"].argmax(dim=-1), Rigid.from_tensor_4x4(batch["frames"]), batch["resi_pad_mask"])

        total_loss, loss_dict = self.loss(preds, batch)
        for dynamics_type, losses in loss_dict.items():
            for loss_name, loss_value in losses.items():
                log_key = f"validation/{dynamics_type}/{loss_name}"
                self.log(log_key,loss_value,on_epoch=True,on_step=False,sync_dist=True)
                # self.log_dict({log_key:loss_value})
                # if self.logger is not None:
                # self.logger.experiment[log_key].append(loss_value, step=self.global_step)
        # Log the loss and return
        # self.log_dict({"train_losses/total_loss":total_loss})
        # if self.logger is not None:
        # self.logger.experiment["val_losses/total_loss"].append(total_loss, step=self.global_step)
        self.log("validation/total_loss",total_loss,on_epoch=True,on_step=False,sync_dist=True)
        return total_loss