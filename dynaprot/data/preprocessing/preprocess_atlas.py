import mdtraj as md
import pandas as pd
import numpy as np
import os
import multiprocessing
from rich.progress import Progress
from rich.console import Console
from dynaprot.data.datasets import DynaProtDataset
from openfold.data import data_transforms, feature_pipeline
from dynaprot.data.utils import from_pdb_string, align_one_protein
import torch
import argparse
from functools import partial
from Bio import PDB



def parse_args():
    parser = argparse.ArgumentParser(description='Process MD trajectories and compute Gaussian parameters per residue.')
    parser.add_argument('--inpath', type=str, required=False, help='Input directory containing folders (protein names) that have trajectories')
    parser.add_argument('--outpath', type=str, required=False, help='Output directory to save the results.')
    
    return parser.parse_args()


args = parse_args()
inpath = args.inpath
outpath = args.outpath
console =Console()


def preprocess_atlas():
    # proteins = [prot for prot in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, prot))]
    proteins = np.load("dynaprot/data/preprocessing/protein_lists/atlas_proteins.npy")
    total_chains = len(proteins)
    with Progress() as progress:
        task = progress.add_task(f"[cyan]dynaprot preprocessing of ATLAS chains (0/{total_chains})...", total=total_chains)
        with multiprocessing.Pool(processes=40) as pool:
            for i,protein in enumerate(pool.imap(process_one_trajectory, proteins)):
                progress.update(task, advance=1,description=f"[cyan]dynaprot preprocessing of ATLAS chains ({i}/{total_chains})... completed {protein}")



def process_one_trajectory(prot):   # just taking R1 traj and topology file as pdb. Is this valid??
    name,chain = prot.split("_")
    
    traj_path = os.path.join(inpath, prot, prot+"_prod_R1_fit.xtc")
    pdb_path = os.path.join(inpath,prot, prot+".pdb")
    
    # generate feats and process them into dicts
    feats = from_pdb_string(open(pdb_path, 'r').read())
    feats = feature_pipeline.np_to_tensor_dict(feats, feats.keys()) # converting to tensor dict
    feats = data_transforms.atom37_to_frames(feats)                 # Getting true backbone frames (num_res, 4, 4)
    feats = data_transforms.get_backbone_frames(feats)
    selected_feats = {k:feats[k] for k in ["aatype","residue_index","all_atom_positions","all_atom_mask"]}
    selected_feats["frames"] = feats["backbone_rigid_tensor"] 

    
    # compute all dynamics here
    traj = md.load(traj_path,top=pdb_path)
    ref = md.load(pdb_path)     
    traj.superpose(ref)         # superpose to our mmcifs
                                                
    selected_feats["dynamics_means"], selected_feats["dynamics_covars"] = compute_gaussians_per_residue(traj)
    
    selected_feats["dynamics_covars"] = align_one_protein(selected_feats)
    
    # print(selected_feats)
    if not os.path.exists(outpath+f"/{prot}/"):
        os.mkdir(outpath+f"/{prot}/")
    
    torch.save(selected_feats,os.path.join(outpath,prot,f"{prot}.pt"))
        
    return prot
    
# Function to compute mean and variance per residue
def compute_gaussians_per_residue(traj):
    num_residues = traj.topology.n_residues
    means = np.zeros((num_residues, 3))       # Shape (n_residues, 3) for (x, y, z)
    variances = np.zeros((num_residues, 3,3))   # Shape (n_residues, 3) for (x, y, z)

    # Loop over all residues
    for i, residue in enumerate(traj.topology.residues):
        # Get atom indices for this residue
        atom_indices = [atom.index for atom in residue.atoms]
        
        # Extract xyz coordinates for all atoms in the residue across all frames
        # scale nanometers to angstroms (x10)
        xyz = np.mean(traj.xyz[:, atom_indices, :],axis=1) * 10 # shape (n_frames, 3)  frames by residue i's position (mean pos of atoms) 

        # Compute mean and variance across all frames for each atom
        means[i] = np.mean(xyz, axis=0)  # shape (1, 3)

        centered_data = xyz - means[i]
        
        variances[i] = centered_data.T @ centered_data /(centered_data.shape[0] - 1)  # shape (3, 3) 

    

    return torch.from_numpy(means), torch.from_numpy(variances)


# def align_to_local_frames():
#     cfg = {
#         "dynam_data_dir":args.outpath,
#         "struc_data_dir":"/data/cb/scratch/datasets/pdb_npz",
#     }
#     data = DynaProtDataset(cfg)

#     with Progress() as progress:
#         task = progress.add_task("[cyan]Locally aligning proteins...", total=len(data))
#         with multiprocessing.Pool(processes=40) as pool:
#             results = []
#             align_function = partial(align_one_protein, data=data)
#             for i, result in enumerate(pool.imap(align_function, range(len(data)))):
#                 results.append(result)
#                 progress.update(task, advance=1)
                
#             np.save("aligned_proteins.npy",np.array(results))
            


# process_one_trajectory("2c9i_D")

preprocess_atlas()

