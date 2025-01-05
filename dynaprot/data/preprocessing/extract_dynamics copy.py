import mdtraj as md
import pandas as pd
import numpy as np
import os
import multiprocessing
from rich.progress import Progress
from rich.console import Console
from dynaprot.data.datasets import DynaProtDataset
from openfold.utils.rigid_utils import  Rigid
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
    proteins = np.load(outpath+"/atlas_proteins.npy")
    print(proteins, proteins.shape)
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Preprocessing ATLAS dataset for dynaprot...", total=len(proteins))
        with multiprocessing.Pool(processes=40) as pool:
            for protein in pool.imap(process_one_trajectory, proteins):
                progress.update(task, advance=1,description=f"[cyan]Processed {protein}")



def process_one_trajectory(prot):   # just taking R1 traj and topology file as pdb. Is this valid??
    name,chain = prot.split("_")
    traj = md.load(os.path.join(inpath, prot, prot+"_prod_R1_fit.xtc"),top=os.path.join(inpath,prot, prot+".pdb"))
    # superpose to our mmcifs
    ref = md.load(f"/data/cb/scratch/datasets/pdb_chains/{prot[1:3]}/{prot}.cif")
    traj.superpose(ref)
    
    means, vars = compute_gaussians_per_residue(traj)
    if not os.path.exists(outpath+f"/{prot}/"):
        os.mkdir(outpath+f"/{prot}/")
    if not os.path.exists(outpath+f"/{prot}/dynamics_labels/"):
        os.mkdir(outpath+f"/{prot}/dynamics_labels/")
    np.save(outpath+f"/{prot}/dynamics_labels/means.npy",  means)
    np.save(outpath+f"/{prot}/dynamics_labels/covariances.npy",  vars)
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

    print(means,variances)
    return means, variances


def align_to_local_frames():
    cfg = {
        "dynam_data_dir":args.outpath,
        "struc_data_dir":"/data/cb/scratch/datasets/pdb_npz",
    }
    data = DynaProtDataset(cfg)

    with Progress() as progress:
        task = progress.add_task("[cyan]Locally aligning proteins...", total=len(data))
        with multiprocessing.Pool(processes=40) as pool:
            results = []
            align_function = partial(align_one_protein, data=data)
            for i, result in enumerate(pool.imap(align_function, range(len(data)))):
                results.append(result)
                progress.update(task, advance=1)
                
            np.save("aligned_proteins.npy",np.array(results))
            
                
def align_one_protein(i,data):
    """Aligns the covariance matrices of a single protein to its local frames."""
    name = data.protein_list[i]
    feats = data.get_feats(name)
    prot_frames, prot_covars = feats["frames"].double(), feats["dynamics_covars"].double()
    
    assert prot_frames.shape[0] != prot_covars.shape[0]
        
    rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices

    if not os.path.exists(args.outpath+f"/{name}/dynamics_labels/covariances_local.npy"):
    # Align covariance matrices using the rotation matrices
        # try:
        aligned_covars = torch.einsum("nij,njk,nlk->nil", rotations, prot_covars, rotations)
        np.save(args.outpath+f"/{name}/dynamics_labels/covariances_local.npy",  aligned_covars.numpy())
        # except:
        #     console.print(f"[cyan]Error {name}, frames shape {prot_frames.shape[0]}, covars shape {prot_covars.shape[0]}")    
        #     return name+"_error"
        
    return name


process_one_trajectory("1m5q_I")
# extract_residue_gaussians_atlas()
# align_to_local_frames()

