import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from openfold.model.structure_module import InvariantPointAttention
from openfold.utils.rigid_utils import  Rigid


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


    def forward(self, sequence, frames, mask):  # what to do with mask at inference?

        batch_size = sequence.shape[0]
        seq_emb = self.sequence_embedding(sequence)  # Shape: (batch_size, num_residues, d_model)

        pos_emb = self.position_embedding.expand_as(seq_emb)  # Shape: (batch_size, num_residues, d_model)
        residue_features = seq_emb + pos_emb

        # Initialize pairwise embeddings (e.g., could be contact map or learned, right now is zero)
        pairwise_embeddings = self.init_pairwise_features(sequence)

        # IPA blocks
        for ipa_block in self.ipa_blocks:
            residue_features = ipa_block(residue_features, pairwise_embeddings, frames, mask)

        # Predict means and variances
        means = self.mean_predictor(residue_features)
        L_entries  = self.covars_predictor(residue_features) # predicts the 6 L entries
                
        # Populate the lower triangular part of L using preds
        L = torch.zeros(batch_size,self.num_residues,3,3)

        i = 0
        for c in range(3):
            for r in range(c,3):
                L[:,:,r,c] = L_entries[:,:,i] 
                if r == c:
                    L[:,:,r,c] = F.softplus(L[:,:,r,c])     # Ensure nonnegativity for the variances (when r==c)
                i+=1

        covars = L @ L.transpose(-1, -2)   # Calculate Sigma = L @ L^T to get the covariance matrix
        
        preds = dict(
            means = means,      # Shape: (batch_size, num_residues, 3)
            covars = covars     # Shape: (batch_size, num_residues, 3, 3)
        )

        return preds


    def init_pairwise_features(self, sequence):
        # Dummy initialization of pairwise features, can be enhanced with structural data
        batch_size = sequence.size(0)
        pairwise_features = torch.zeros(batch_size, self.num_residues, self.num_residues, self.d_model)
        return pairwise_features


    def training_step(self, batch, batch_idx):
        sequence, frames, mask, true_means, true_covars = batch["aatype"].argmax(dim=-1), Rigid.from_tensor_4x4(batch["frames"]), batch["resi_pad_mask"], batch["dynamics_means"], batch["dynamics_covars"]

        preds = self(sequence, frames, mask)
        predicted_means, predicted_covars = preds["means"],preds["covars"]

        mean_loss = F.mse_loss(predicted_means, true_means)
        covars_loss = self.kl_divergence_loss(predicted_means, predicted_covars, true_means, true_covars)

        total_loss = mean_loss + covars_loss

        # Log the loss and return
        self.log('train_loss', total_loss)
        return total_loss


    def kl_divergence_loss(self, pred_means, pred_vars, true_means, true_vars):
        # KL Divergence between two Gaussian distributions
        # pred_means, pred_vars are predicted, true_means, true_vars are ground truth
        kl_loss = (true_vars / pred_vars).sum() + ((pred_means - true_means) ** 2 / pred_vars).sum()
        kl_loss -= self.num_residues  # Adjustment for dimensions
        kl_loss *= 0.5
        return kl_loss


    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # def validation_step(self, batch, batch_idx):
    #     sequence, true_means, true_variances, initial_coords = batch
    #     predicted_means, predicted_variances = self(sequence, initial_coords)

    #     mean_loss = F.mse_loss(predicted_means, true_means)
    #     variance_loss = self.kl_divergence_loss(predicted_means, predicted_variances, true_means, true_variances)
    #     total_loss = mean_loss + self.beta * variance_loss

    #     # Log validation loss
    #     self.log('val_loss', total_loss)
    #     return total_loss
