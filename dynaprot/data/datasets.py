import torch
import os
from torch.utils.data import Dataset
from openfold.data import data_transforms, feature_pipeline
from openfold.np.protein import from_pdb_string, Protein
import numpy as np
import pandas as pd
from dynaprot.data.transforms import make_fixed_size
from dynaprot.data.utils import dict_multimap


class DynaProtDataset(Dataset):
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.label_dir = cfg["dynam_data_dir"]                                                                         # directory for protein dynamics labels
        self.data_dir = cfg["struc_data_dir"]                                                                          # directory of protein structures (npz/pdb/mmcif files)
        self.protein_list = [prot for prot in os.listdir(self.label_dir) if os.path.isdir(os.path.join(self.label_dir, prot))]  # dynamics dataset protein list

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        protein_id = self.protein_list[idx]
        # matching_npz =  self.protein_chains[self.protein_chains["name"] == protein_id]

        try:
            path = f"{self.data_dir}/{protein_id}.npz"
            mmcif_feats = dict(np.load(path, allow_pickle=True))
        except:
            path = f"{self.data_dir}/{protein_id[1:3]}/{protein_id}.npz"
            mmcif_feats = dict(np.load(path, allow_pickle=True))
            
        feats = mmcif_feats 
        feats.pop("sequence")       
        feats.pop("release_date")
        feats.pop("domain_name")
        
        feats["resi_pad_mask"] = np.ones(feats["seq_length"][0])                                     # mask will be padded with zeros later on

        # adding dynamics labels to escape padding
        feats["dynamics_means"]= np.load(os.path.join(self.label_dir, f"{protein_id}/dynamics_labels/means.npy"))
        feats["dynamics_covars"] = np.load(os.path.join(self.label_dir, f"{protein_id}/dynamics_labels/covariances.npy"))
        
        
        feats = feature_pipeline.np_to_tensor_dict(feats, feats.keys())
        
        feats = data_transforms.atom37_to_frames(feats)             # Getting true backbone frames (num_res, 4, 4)
        feats = data_transforms.get_backbone_frames(feats)


        selected_feats = {k:feats[k] for k in ["aatype","residue_index","all_atom_positions","all_atom_mask", "resi_pad_mask","dynamics_means","dynamics_covars"]}
        selected_feats["frames"] = feats["backbone_rigid_tensor"][torch.arange(feats["backbone_rigid_tensor"].shape[0]),feats['aatype'].argmax(dim=1)]     # only selecting frame for relevant residue type
                
        shape_schema = {}
        for k in selected_feats.keys():
            schema = list(selected_feats[k].size())
            schema[0] = "NUM_RES"   # to be infilled by padding function
            shape_schema[k] = schema

        padded_selected_feats = make_fixed_size(selected_feats,shape_schema, num_residues = self.cfg["max_num_residues"])      
        
        return padded_selected_feats
    
    # def get_dataloaders(self):
    #     train_dataloader = torch.utils.data.DataLoader(
    #         self,
    #         batch_size=self.cfg["train"],
    #         collate_fn=OpenFoldBatchCollator(),
    #         num_workers=12,
    #         shuffle=False,
    #     )
        
    
class OpenFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = lambda x: torch.stack(x, dim=0) if isinstance(x[0], torch.Tensor) else x
        return dict_multimap(stack_fn, prots) 



if __name__ == "__main__":
    dataset = DynaProtDataset({"max_seq_len":300},"./atlas_proteins.npy","/data/cb/scratch/datasets/pdb_npz/","/data/cb/scratch/datasets/atlas_gaussians/")

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            collate_fn=OpenFoldBatchCollator(),
            num_workers=12,
            shuffle=False,
        )

    batch_prots = next(iter(dataloader))

    for k in batch_prots.keys():
        # print(k,batch_prots[k])
        print(f"{k}:{batch_prots[k].shape}")
    # print(batch_prots["resi_pad_mask"])
    # print(batch_prots["dynamics_means"])