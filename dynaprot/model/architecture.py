import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from openfold.model.structure_module import InvariantPointAttention
from openfold.utils.rigid_utils import  Rigid
from dynaprot.model.loss import DynaProtLoss


class DynaProt(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_residues = cfg["data_config"]["max_num_residues"]
        self.num_ipa_blocks = cfg["model_params"]["num_ipa_blocks"]
        self.d_model = cfg["model_params"]["d_model"]
        self.lr = cfg["train_params"]["learning_rate"]
        # self.beta = beta  # Weighting factor for variance loss

        # Embedding layers for sequence and pairwise features
        self.sequence_embedding = nn.Embedding(21, self.d_model)  # 21 amino acid types
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_residues, self.d_model))

        # IPA layers
        self.ipa_blocks = nn.ModuleList([InvariantPointAttention(c_s=self.d_model,c_z=self.d_model,c_hidden=16,no_heads=4,no_qk_points=4,no_v_points=8) for _ in range(self.num_ipa_blocks)])

        # Dense layers for predicting means and variances
        self.mean_predictor = nn.Linear(self.d_model, 3)  # Predict (x, y, z) mean
        self.covars_predictor = nn.Linear(self.d_model, 6)  # Predict lower diagonal matrix L (cholesky decomposition) to ensure symmetric psd Σ = LL^T
        
        # for stability
        self.epsilon = 1e-6      # regularization to ensure Σ is positive definite and other stability issues
        
        # Initialize DynaProtLoss which handles all separate loss functions
        self.loss = DynaProtLoss(self.cfg)
        
        
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


    def forward(self, sequence, frames, mask):  # what to do with mask at inference?
        seq_emb = self.sequence_embedding(sequence)  # Shape: (batch_size, num_residues, d_model)

        pos_emb = self.position_embedding.expand_as(seq_emb)  # Shape: (batch_size, num_residues, d_model)
        residue_features = seq_emb + pos_emb

        # Initialize pairwise embeddings (e.g., could be contact map or learned, right now is zero)
        pairwise_embeddings = self.init_pairwise_features(sequence).to(residue_features) 

        # IPA blocks
        for ipa_block in self.ipa_blocks:
            residue_features = ipa_block(residue_features, pairwise_embeddings, frames, mask)

        preds = dict(
            means = self.pred_mean(residue_features),      # Shape: (batch_size, num_residues, 3)
            covars = self.pred_covars(residue_features)    # Shape: (batch_size, num_residues, 3, 3)
        )

        return preds


    def init_pairwise_features(self, sequence):
        pairwise_features = torch.zeros(sequence.shape[0], self.num_residues, self.num_residues, self.d_model)
        return pairwise_features


    def pred_mean(self, residue_features):
        return  self.mean_predictor(residue_features)
    
    
    def pred_covars(self, residue_features):
        L_entries  = self.covars_predictor(residue_features) # predicts the 6 L entries
                
        # Populate the lower triangular part of L using preds
        L = torch.zeros(residue_features.shape[0],self.num_residues,3,3).to(L_entries)  # put on same device

        i = 0
        for c in range(3):
            for r in range(c,3):
                if r == c:
                    L[:,:,r,c] = F.softplus(L_entries[:,:,i])     # Ensure nonnegativity for the variances (when r==c)
                else:
                    L[:,:,r,c] = L_entries[:,:,i] 
                i+=1

        covars = L @ L.transpose(-1, -2) + self.epsilon * torch.eye(3).to(L)  # Σ = LL^T 
        
        return covars
    
    
    def training_step(self, batch, batch_idx):
        
        preds = self(batch["aatype"].argmax(dim=-1), Rigid.from_tensor_4x4(batch["frames"]), batch["resi_pad_mask"] )

        total_loss, loss_dict = self.loss(preds, batch)
        
        for dynamics_type, losses in loss_dict.items():
            for loss_name, loss_value in losses.items():
                log_key = f"train_losses/{dynamics_type}/{loss_name}"
                self.logger.experiment[log_key].append(loss_value)
        # Log the loss and return
        # self.log_dict({"train_total_loss":total_loss})
        self.logger.experiment["train_losses/total_loss"].append(loss_value)
        return total_loss


    # def validation_step(self, batch, batch_idx):
    #     sequence, true_means, true_variances, initial_coords = batch
    #     predicted_means, predicted_variances = self(sequence, initial_coords)

    #     mean_loss = F.mse_loss(predicted_means, true_means)
    #     variance_loss = self.kl_divergence_loss(predicted_means, predicted_variances, true_means, true_variances)
    #     total_loss = mean_loss + self.beta * variance_loss

    #     # Log validation loss
    #     self.log('val_loss', total_loss)
    #     return total_loss
