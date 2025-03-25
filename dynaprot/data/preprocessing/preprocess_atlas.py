import mdtraj as md
import pandas as pd
import numpy as np
import os
import multiprocessing
from rich.progress import Progress
from rich.console import Console
from dynaprot.data.datasets import DynaProtDataset
from openfold.data import data_transforms, feature_pipeline
from dynaprot.data.utils import from_pdb_string, map_one_protein_local_frame
import torch
import argparse
from functools import partial
from Bio import PDB
import random
import io
import time
import tempfile

def parse_args():
    parser = argparse.ArgumentParser(description='Process MD trajectories and compute Gaussian parameters per residue.')
    parser.add_argument('--inpath', type=str, required=False, help='Input directory containing folders (protein names) that have trajectories')
    parser.add_argument('--outpath', type=str, required=False, help='Output directory to save the results.')
    parser.add_argument('--calpha', action='store_true', help='Use Cα atoms instead of centroids.')

    
    return parser.parse_args()


args = parse_args()
inpath = args.inpath        # /data/cb/scratch/datasets/atlas
outpath = args.outpath      # /data/cb/scratch/datasets/atlas_dynamics_labels
console =Console()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def preprocess_atlas():
    # proteins = [prot for prot in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, prot))]
    tic = time.time()
    proteins = np.load("dynaprot/data/preprocessing/protein_lists/atlas_proteins.npy")
    total_chains = len(proteins)
    with Progress() as progress:
        task = progress.add_task(f"[cyan]dynaprot preprocessing of ATLAS chains (0/{total_chains})...", total=total_chains)
        with multiprocessing.Pool(processes=40) as pool:
            for i,protein in enumerate(pool.imap(process_one_trajectory_atlas, proteins)):
                progress.update(task, advance=1,description=f"[cyan]dynaprot preprocessing of ATLAS chains ({i}/{total_chains})... completed {protein}")

        tim = round((time.time() - tic)/60,2)
        progress.update(task, advance=1,description=f"[cyan]completed dynaprot preprocessing of ATLAS chains ({total_chains}/{total_chains}) in {tim} min")


def save_separate_frames(prot, superposed_traj, frame_stride=20):
    frame_dir_path = outpath+f"/{prot}/"+"frames/"
    if not os.path.exists(frame_dir_path):
        os.mkdir(frame_dir_path)
        
    traj = superposed_traj
    for i in range(0, len(traj), frame_stride):
        frame_path = os.path.join(frame_dir_path, f"frame_{i:05d}.pt")  # Save as frame_00000.pt, frame_00005.pt, etc.
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmp_pdb:
            traj[i].save_pdb(tmp_pdb.name)  # Save PDB to temp file
            pdb_string = tmp_pdb.read().decode()
            # print(pdb_string)
        feats = from_pdb_string(pdb_string)
        feats = feature_pipeline.np_to_tensor_dict(feats, feats.keys())
        feats = data_transforms.atom37_to_frames(feats)
        feats = data_transforms.get_backbone_frames(feats)
        torch.save({"frames":feats["backbone_rigid_tensor"]}, frame_path)
        
    return prot
        

def process_one_trajectory_atlas(prot):  
    name,chain = prot.split("_")
    if not os.path.exists(outpath+f"/{prot}/"):
        os.mkdir(outpath+f"/{prot}/")

    pt_path = os.path.join(outpath,prot,f"{prot}.pt")
    
    pdb_path = os.path.join(inpath,prot, prot+".pdb")
    ref = md.load(pdb_path)     

    replicates = []
    for i in range(3):  # assuming 3 replicates
        traj_path = os.path.join(inpath, prot, f"{prot}_prod_R{i+1}_fit.xtc")
        traj = md.load(traj_path, top=pdb_path)
        replicates.append(traj)
    
    traj = md.join(replicates)
    traj.superpose(ref)         # superpose to our reference
    
    # if os.path.exists(pt_path):
    #     selected_feats = torch.load(pt_path)
    #     selected_feats["dynamics_rmsf"] = compute_rmsf_from_covariances(selected_feats["dynamics_covars_local"])
    #     torch.save(selected_feats,pt_path)
    #     return prot
    
    save_separate_frames(prot, traj, 100)
    
    # generate feats and process them into dicts
    feats = from_pdb_string(open(pdb_path, 'r').read())
    feats = feature_pipeline.np_to_tensor_dict(feats, feats.keys()) # converting to tensor dict
    feats = data_transforms.atom37_to_frames(feats)                 # Getting true backbone frames (num_res, 4, 4)
    feats = data_transforms.get_backbone_frames(feats)
    selected_feats = {k:feats[k] for k in ["aatype","residue_index","all_atom_positions","all_atom_mask"]}
    selected_feats["frames"] = feats["backbone_rigid_tensor"] 

    # compute all dynamics here                 
    selected_feats["dynamics_means"], selected_feats["dynamics_covars_global"],selected_feats["dynamics_fullcovar_global"] = compute_gaussians_per_residue(traj, args.calpha)
    # selected_feats["dynamics_fullcovar"] = compute_full_covariance(traj, args.calpha)

    selected_feats["dynamics_covars_local"],selected_feats["dynamics_fullcovar_local"] = map_one_protein_local_frame( selected_feats["frames"].double(),  selected_feats["dynamics_covars_global"].double(),selected_feats["dynamics_fullcovar_global"].double())
    selected_feats["dynamics_rmsf"] = compute_rmsf_from_covariances(selected_feats["dynamics_covars_local"])

    selected_feats["dynamics_correlations"] = compute_residue_correlations(selected_feats["dynamics_fullcovar_local"])
    
    torch.save(selected_feats,pt_path)
    
    return prot



def compute_rmsf_from_covariances(cov_matrices):
    trace = cov_matrices.diagonal(dim1=1, dim2=2).sum(dim=1)
    rmsf = torch.sqrt(trace)
    return rmsf
    
    
def compute_gaussians_per_residue(traj, calpha: bool):
    num_residues = traj.topology.n_residues
    means = np.zeros((num_residues, 3),dtype=np.float64)       # Shape (n_residues, 3) for (x, y, z)
    covariances = np.zeros((num_residues, 3,3),dtype=np.float64)   # Shape (n_residues, 3) for (x, y, z)
    residuecoords = []

    for i, residue in enumerate(traj.topology.residues):
        use_calpha = calpha
        if use_calpha:
            ca_atom = [atom.index for atom in residue.atoms if atom.name == 'CA']
            if ca_atom:
                xyz = traj.xyz[:, ca_atom[0], :].astype(np.float64) * 10           # shape (T, 3)
            else:
                use_calpha = False  # calpha wasnt found
        
        if not use_calpha:
            atom_indices = [atom.index for atom in residue.atoms]   # Extract xyz coordinates for all atoms in the residue across all frames
            # scale nanometers to angstroms (x10)
            xyz = np.mean(traj.xyz[:, atom_indices, :].astype(np.float64),axis=1) * 10 # shape (T, 3)  frames by residue i's position (mean pos of atoms) 

        mean_xyz = np.mean(xyz, axis=0).astype(np.float64)  # shape (1, 3)
        centered_xyz = xyz-mean_xyz # shape (T,3)
        residuecoords.append(centered_xyz)

        # Compute mean and variance across all frames for each residue
        means[i] = mean_xyz
        covariances[i] = (centered_xyz.T @ centered_xyz /(centered_xyz.shape[0] - 1)).astype(np.float64)  # shape (3, 3) 

    X_centered = np.stack(residuecoords, axis=2).astype(np.float64) # shape (T, 3, n_residues) 
    T, _, N = X_centered.shape  # T: time frames, 3: coordinates, N: residues
    X_centered = X_centered.reshape((T,3*N))    # shape (T, 3 * num_residues)
    
    covariance_full = (X_centered.T @ X_centered / (T-1)).astype(np.float64)
    
    return torch.from_numpy(means), torch.from_numpy(covariances), torch.from_numpy(covariance_full)



def compute_residue_correlations(full_cov, mode="trace"):  # using trace or fro based correlation. Also, should probably run after alignment

    N = int(full_cov.shape[0]/3)
    C = np.zeros((N, N))
    if mode=="trace":
        corr_func = lambda x : np.trace(x)
    elif mode=="frobenius" or mode =="fro":
        corr_func = lambda x : np.linalg.norm(x,'fro')
    
    for i in range(N):
        Sigma_ii = full_cov[3*i:3*i+3, 3*i:3*i+3]
        corr_ii = corr_func(Sigma_ii)
        for j in range(N):
            Sigma_jj = full_cov[3*j:3*j+3, 3*j:3*j+3]
            corr_jj = corr_func(Sigma_jj)
            Sigma_ij = full_cov[3*i:3*i+3, 3*j:3*j+3]
            C[i, j] = corr_func(Sigma_ij) / np.sqrt(corr_ii * corr_jj)
    
    return torch.from_numpy(C)




# process_one_trajectory_atlas("1bq8_A")

preprocess_atlas()

