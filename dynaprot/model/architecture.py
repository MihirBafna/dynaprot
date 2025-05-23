import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from dynaprot.model.operators.lr_ipa import IPABlock as LRIPABlock
from dynaprot.model.operators.lr_ipa import InvariantPointAttention as LRIPA
from dynaprot.model.operators.lr_ipa import FeedForward as FF
from dynaprot.model.operators.lowrank import LowRankDiagonalReadout
from dynaprot.model.operators.lr_pairattention import PairwiseAttentionBlock
from openfold.utils.rigid_utils import  Rigid
from dynaprot.model.loss import DynaProtLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.checkpoint import checkpoint_sequential


class DynaProt(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.num_residues = cfg["data_config"]["max_num_residues"]
        self.num_ipa_blocks = cfg["model_params"]["num_ipa_blocks"]
        self.d_model = cfg["model_params"]["d_model"]
        self.lr = cfg["train_params"]["learning_rate"]
        self.warmup_steps = cfg["train_params"]["warmup_steps"]
        self.total_steps = cfg["train_params"]["total_steps"]
        self.grad_clip_val = cfg["train_params"]["grad_clip_norm"]


        self.sequence_embedding = nn.Embedding(21, self.d_model)  # 21 amino acid types
        self.use_sinusoidal = cfg["model_params"].get("use_sinusoidal_pos_emb", False)
        if self.use_sinusoidal:
            self.pos_encoder = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=10000)
        # else:
        #     self.position_embedding = nn.Parameter(torch.zeros(1, self.num_residues, self.d_model))

        self.automatic_optimization = False
        self.grad_accum_batches = cfg["train_params"].get("accumulate_grad_batches", 1)
        self._grad_accum_counter = 0
        self._opt_step_counter = 0
        self._loss_accumulator = 0.0
        self._loss_steps = 0

        self.out_type = self.cfg["train_params"]["out_type"]
        # IPA layers
        # self.ipa_blocks = nn.ModuleList([OpenFoldIPA(c_s=self.d_model,c_z=self.d_model,c_hidden=16,no_heads=4,no_qk_points=4,no_v_points=8) for _ in range(self.num_ipa_blocks)])
        # self.ipa_blocks = nn.ModuleList([LRIPABlock(dim=self.d_model, require_pairwise_repr=False) for _ in range(self.num_ipa_blocks)])
        
        self.ipa_blocks = nn.ModuleList([LRIPA(dim=self.d_model,require_pairwise_repr=False) for _ in range(self.num_ipa_blocks)])

        self.dropout = nn.Dropout(0.2)

        if "marginal" in self.out_type:
            self.covars_predictor = FF(
                dim=self.d_model,
                mult=self.cfg["model_params"].get("marginal_readout_mult", 1),
                num_layers=self.cfg["model_params"].get("marginal_readout_layers", 1),
                act=nn.ReLU,
                output_dim=6
            ) # Predict lower diagonal matrix L (cholesky decomposition) to ensure symmetric psd Σ = LL^T
            
            if self.cfg["model_params"].get("marginal_layernorm", False):
                self.ln = nn.LayerNorm(self.d_model)
                
        if "joint" in self.out_type:
            # if  "cholesky" in self.out_type:
            #     self.global_corr_predictor = self.get_corr_predictor(
            #         input_dim=1 + 2 * self.d_model,
            #         hidden_dim=self.cfg["model_params"].get("readout_hidden_dim", 256),
            #         num_layers=self.cfg["model_params"].get("readout_layers", 4),
            #         dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
            #         use_layernorm=self.cfg["model_params"].get("use_layernorm", False)
            #     )
            # elif "lowrank" in self.out_type:
            #     self.global_corr_predictor = LowRankDiagonalReadout(
            #             d_model=self.d_model,
            #             rank=self.cfg["model_params"].get("readout_rank", 8),
            #             hidden_dim=self.cfg["model_params"].get("readout_hidden_dim", 256),
            #             num_layers=self.cfg["model_params"].get("readout_layers", 3),
            #             dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
            #             # use_layernorm=self.cfg["model_params"].get("readout_use_layernorm", False)
            #         )
            if self.out_type=="joint_pairattention_choleskycombiner":
                self.pair_divisor = self.cfg["model_params"].get("pair_divisor", 4)
                self.pairwise_projector = nn.Linear(2 * self.d_model, self.d_model // self.pair_divisor)
                self.pair_blocks = nn.Sequential(*[
                    PairwiseAttentionBlock(
                        dim=self.d_model // self.pair_divisor,
                        heads=self.cfg["model_params"].get("pair_heads", 2),
                        dim_head=self.cfg["model_params"].get("pair_dim_head", 8),
                        dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
                        global_column_attn=False
                    )
                    for _ in range(self.cfg["model_params"].get("pair_blocks", 1))
                ])
                
                self.combiner = FF(
                    dim=self.d_model // self.pair_divisor,
                    mult=1.0,                           
                    num_layers=3,
                    act=nn.ReLU,
                    output_dim=1
                )
  
            elif self.out_type=="joint_pairattention_lowrank":
                self.pair_divisor = self.cfg["model_params"].get("pair_divisor", 4)
                self.pairwise_projector = nn.Linear(2 * self.d_model, self.d_model // self.pair_divisor)
                self.pair_blocks = nn.Sequential(*[
                    PairwiseAttentionBlock(
                        dim=self.d_model // self.pair_divisor,
                        heads=self.cfg["model_params"].get("pair_heads", 2),
                        dim_head=self.cfg["model_params"].get("pair_dim_head", 8),
                        dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
                        global_column_attn=False
                    )
                    for _ in range(self.cfg["model_params"].get("pair_blocks", 1))
                ])
                
                self.global_corr_predictor = LowRankDiagonalReadout(
                    input_dim=self.d_model // self.pair_divisor,
                    hidden_dim=self.cfg["model_params"].get("readout_hidden_dim"),
                    num_layers=self.cfg["model_params"].get("readout_layers", 4),
                    dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
                    # use_layernorm=self.cfg["model_params"].get("use_layernorm", False)
                )
            elif self.out_type=="joint_pairattention":
                self.pair_divisor = self.cfg["model_params"].get("pair_divisor", 4)
                self.pairwise_projector = nn.Linear(2 * self.d_model, self.d_model // self.pair_divisor)
                self.pair_blocks = nn.Sequential(*[
                    PairwiseAttentionBlock(
                        dim=self.d_model // self.pair_divisor,
                        heads=self.cfg["model_params"].get("pair_heads", 2),
                        dim_head=self.cfg["model_params"].get("pair_dim_head", 8),
                        dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
                        global_column_attn=False
                    )
                    for _ in range(self.cfg["model_params"].get("pair_blocks", 1))
                ])
                
                self.global_corr_predictor = self.get_corr_predictor(
                    input_dim=self.d_model // self.pair_divisor,
                    hidden_dim=self.cfg["model_params"].get("readout_hidden_dim"),
                    num_layers=self.cfg["model_params"].get("readout_layers", 4),
                    dropout=self.cfg["model_params"].get("readout_dropout", 0.1),
                    use_layernorm=self.cfg["model_params"].get("use_layernorm", False)
                )

                
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


    
    def get_corr_predictor(self, input_dim, hidden_dim, num_layers=3, dropout=0.1, use_layernorm=False):
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)
    
    

    def forward(self, sequence, frames, mask):  # mask is just seq length at inference time
        seq_emb = self.sequence_embedding(sequence)  # Shape: (batch_size, num_residues, d_model)

        if self.use_sinusoidal:
            residue_features = self.pos_encoder(seq_emb)
        else:
            # residue_features = seq_emb + self.position_embedding[:, :seq_emb.shape[1], :]
            residue_features = seq_emb


        rots = frames[..., :3, :3]
        trans = frames[..., :3, 3]
        
        for i,ipa in enumerate(self.ipa_blocks):
            residue_features, attn =  ipa(single_repr=residue_features, rotations=rots,translations=trans, mask=mask.bool(), return_attn= (i == len(self.ipa_blocks) - 1))
            residue_features = self.dropout(residue_features)
            
        if self.cfg["model_params"].get("marginal_layernorm", False):
            residue_features = self.ln(residue_features)
        
        preds = dict()
        if "marginal" in self.out_type:
            preds["marginal_covars"] = self.pred_marginals(residue_features)    # Shape: (batch_size, num_residues, 3, 3)
        if "joint" in self.out_type:
            preds["joint_covar"] = self.pred_joint(residue_features, attn)   # (batch_size, num_residues, num_residues)
        return preds



    def pred_mean(self, residue_features):
        return  self.mean_predictor(residue_features)
    
    
    def pred_marginals(self, residue_features, lambda_min=0.5, lambda_max=10, soft_clip=True):
        """
        Predict marginal covariance matrices (by predicting cholesky factor).

        Args:
            residue_features (torch.Tensor): Input residue features of shape (batch_size, num_residues, feature_dim).
            lambda_min (float): Minimum eigenvalue for clipping.
            lambda_max (float): Maximum eigenvalue for clipping.

        Returns:
            torch.Tensor: Stabilized covariance matrices of shape (batch_size, num_residues, 3, 3).
        """
        b, n, _ = residue_features.shape
        
        L_entries = self.covars_predictor(residue_features) # predict the 6 L entries

        L = torch.zeros(
            b, n, 3, 3, device=L_entries.device
        )
        i = 0
        for c in range(3):
            for r in range(c, 3):
                if r == c:
                    L[:, :, r, c] = F.softplus(L_entries[:, :, i])  # ensure positive variances
                else:
                    L[:, :, r, c] = L_entries[:, :, i]
                i += 1
        covars = L @ L.transpose(-1, -2)
        return covars
    
    
    def pred_covars_direct(self, residue_features, lambda_min=0.1):
        """
        Predict covariance matrices directly.

        Args:
            residue_features (torch.Tensor): Input residue features of shape (batch_size, num_residues, feature_dim).
            lambda_min (float): Minimum eigenvalue for clipping.
            lambda_max (float): Maximum eigenvalue for clipping.

        Returns:
            torch.Tensor: Stabilized covariance matrices of shape (batch_size, num_residues, 3, 3).
        """
        b, n, _ = residue_features.shape
        
        covar_entries = self.covars_predictor(residue_features) # Predict the 6 L entries

        covars = torch.zeros(
            b, n, 3, 3, device=covar_entries.device
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
                
        # return covars, None
        covars_detached = covars.detach()
        eigenvalues, eigenvectors = torch.linalg.eigh(covars_detached)  # Shape: (batch_size, num_residues, 3), (batch_size, num_residues, 3, 3)
        eigenvalues_clipped = torch.clamp(eigenvalues, min=lambda_min)

        covars_clipped = (
            eigenvectors @ torch.diag_embed(eigenvalues_clipped) @ eigenvectors.transpose(-1, -2)
        )
        return covars, covars_clipped
    

    def pred_joint(self, residue_features, attention):
        # if  "cholesky" in self.out_type:

        #     b, n, _ = residue_features.shape
        #     device = attention.device
        #     num_cholesky_entries = int(n*(n+1)/2)
            
        #     tril_indices = torch.tril_indices(row=n, col=n, offset=0, device=device)
        #     flat_attn_lower_tri = attention[:, tril_indices[0], tril_indices[1]] #  (b, n*(n+1)/2)

        #     features_i = residue_features[:, tril_indices[0], :] # (b, n*(n+1)/2, d_model)
        #     features_j = residue_features[:, tril_indices[1], :] # (b, n*(n+1)/2, d_model)

        #     combined_features = torch.cat(  
        #             (flat_attn_lower_tri.unsqueeze(-1),     # (b, n*(n+1)/2, 1)
        #             features_i,                            # (b, n*(n+1)/2, d_model)
        #             features_j),                           # (b, n*(n+1)/2, d_model)
        #             dim=-1
        #         )                                           # resulting shape (b, n*(n+1)/2, 2*d_model + 1)
            
        #     features_reshaped = combined_features.view(-1, 2*self.d_model + 1)
        #     flat_L_entries = self.global_corr_predictor(features_reshaped).view(b, num_cholesky_entries)
            
        #     L = torch.zeros(b, n, n, device=device)
        #     L[:, tril_indices[0], tril_indices[1]] = flat_L_entries

        #     diag_indices = torch.arange(n, device=device)
        #     L_diag_old = L[:, diag_indices, diag_indices]
        #     L_diag_new = F.softplus(L_diag_old) + self.epsilon                              # ensuring positivity on the diagonal of L
        #     L = L - torch.diag_embed(L_diag_old) + torch.diag_embed(L_diag_new)

        #     covars = L @ L.transpose(-1, -2) # Shape: (b, n, n)
        #     return covars
        
        # elif "lowrank" in self.out_type:
            
        #     return self.global_corr_predictor(residue_features, attention)
        
        if  self.out_type=="joint_pairattention":
            B, N, D = residue_features.shape
            h_i = residue_features.unsqueeze(2).expand(B, N, N, D)
            h_j = residue_features.unsqueeze(1).expand(B, N, N, D)
            pairwise_input = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2D)
            del h_i, h_j
            
            pairwise_input = self.pairwise_projector(pairwise_input)  # (B, N, N, D)
            
            pairwise_output = self.pair_blocks(pairwise_input)  # (B, N, N, 2D)
            # pairwise_output = checkpoint_sequential(self.pair_blocks, len(self.pair_blocks), pairwise_input, use_reentrant=True)  # (B, N, N, 2D)

            tril = torch.tril_indices(N, N, 0, device=residue_features.device)
            z_ij = pairwise_output[:, tril[0], tril[1]]  # (B, num_pairs, 2D)

            # Now run through the original global_corr_predictor
            flat_L_entries = self.global_corr_predictor(z_ij).view(B, -1)

            # Reconstruct L from flat entries
            L = torch.zeros(B, N, N, device=residue_features.device)
            L[:, tril[0], tril[1]] = flat_L_entries

            diag_indices = torch.arange(N, device=residue_features.device)
            L_diag_old = L[:, diag_indices, diag_indices]
            L_diag_new = F.softplus(L_diag_old) + self.epsilon
            L = L - torch.diag_embed(L_diag_old) + torch.diag_embed(L_diag_new)

            covars = L @ L.transpose(-1, -2)
            return covars
        elif "pairattention_choleskycombiner" in self.out_type:
            # breakpoint()
            B, N, D = residue_features.shape
            h_i = residue_features.unsqueeze(2)  # (B, N, 1, D)
            h_j = residue_features.unsqueeze(1)  # (B, 1, N, D)
            pairwise_input = torch.cat([h_i.expand(-1, -1, N, -1), h_j.expand(-1, N, -1, -1)], dim=-1)  # (B, N, N, 2D)
            del h_i, h_j
            
            pairwise_input = self.pairwise_projector(pairwise_input)  # (B, N, N, D)
            pairwise_output_basis = self.pair_blocks(pairwise_input)  # (B, N, N, D)
            
            logits = self.combiner(pairwise_output_basis).squeeze(-1) # (B, N, N)
            
            tril = torch.tril_indices(N, N, 0, device=residue_features.device)
            L = torch.zeros(B, N, N, device=residue_features.device)
            L[:, tril[0], tril[1]] = logits[:, tril[0], tril[1]]

            diag = torch.arange(N, device=residue_features.device)
            L_diag = L[:, diag, diag]
            L = L - torch.diag_embed(L_diag) + torch.diag_embed(F.softplus(L_diag) + self.epsilon)

            covars = L @ L.transpose(-1, -2)  # (B, N, N)
            return covars
        
        elif "pairattention_lowrank" in self.out_type:
            B, N, D = residue_features.shape
            h_i = residue_features.unsqueeze(2).expand(B, N, N, D)
            h_j = residue_features.unsqueeze(1).expand(B, N, N, D)
            pairwise_input = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2D)
            del h_i, h_j
            pairwise_input = self.pairwise_projector(pairwise_input)  # (B, N, N, D)
            pairwise_output = self.pair_blocks(pairwise_input)  # (B, N, N, D)
            return self.global_corr_predictor(residue_features,pairwise_output)
            
            

            
    def on_before_batch_transfer(self, batch, dataloader_idx):
        typ = next(self.parameters()).dtype
        if typ  == torch.float64:
            for k, v in batch.items():
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    batch[k] = v.double()
        elif typ == torch.float32:
           for k, v in batch.items():
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    batch[k] = v.float()
        return batch
    
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        preds = self(batch["aatype"].argmax(dim=-1), batch["frames"], batch["resi_pad_mask"])
        total_loss, loss_dict = self.loss(preds, batch)
        self.manual_backward(total_loss)
        
        self._grad_accum_counter += 1
        self._loss_steps += 1
        self._loss_accumulator += total_loss.detach()
        
        if self._grad_accum_counter % self.grad_accum_batches == 0:
            
            avg_loss = self._loss_accumulator / self._loss_steps

            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm"
            )
            
            optimizer.step()
            optimizer.zero_grad()

            if self.global_step % 10 == 0 or self.grad_accum_batches != 1:
                
                for dynamics_type, losses in loss_dict.items():
                    for loss_name, loss_value in losses.items():
                        log_key = f"train/{dynamics_type}/{loss_name}"
                        loss_all_ranks = self.all_gather(loss_value).mean()
                        if self.trainer.is_global_zero:
                            self.logger.experiment[log_key].append(loss_all_ranks, step=self.global_step * self.grad_accum_batches)
                            
                total_loss_all_ranks = self.all_gather(avg_loss).mean()
                if self.trainer.is_global_zero:
                    self.logger.experiment["train/total_loss"].append(total_loss_all_ranks.item(), step=self.global_step * self.grad_accum_batches)
                    self._loss_accumulator = 0.0
                    self._loss_steps = 0
                                    
                lr = self.optimizers().param_groups[0]["lr"]
                self.logger.experiment["train/learning_rate"].append(lr, step=self.global_step * self.grad_accum_batches)

        return total_loss
    
    
    def validation_step(self, batch, batch_idx):    # called every epoch
        # preds = self(batch["aatype"].argmax(dim=-1), Rigid.from_tensor_4x4(batch["frames"]), batch["resi_pad_mask"])
        preds = self(batch["aatype"].argmax(dim=-1), batch["frames"], batch["resi_pad_mask"])
        total_loss, loss_dict = self.loss(preds, batch)

        for dynamics_type, losses in loss_dict.items():
            for loss_name, loss_value in losses.items():
                log_key = f"val/{dynamics_type}/{loss_name}"
                loss_all_ranks = self.all_gather(loss_value).mean()
                if self.trainer.is_global_zero:
                    self.logger.experiment[log_key].append(loss_all_ranks, step=self.global_step * self.grad_accum_batches) 
                           
        total_loss_all_ranks = self.all_gather(total_loss.detach()).mean()
        if self.trainer.is_global_zero:                   
            self.logger.experiment["val/total_loss"].append(total_loss_all_ranks, step=self.global_step * self.grad_accum_batches)
        return total_loss
    


import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)  # Automatically moves with model to cuda/cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0).to(x.dtype)
