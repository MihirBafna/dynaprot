import mdtraj as md
import pandas as pd
import numpy as np
import os
import multiprocessing
from rich.progress import Progress
from rich.console import Console

import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='Process MD trajectories and compute Gaussian parameters per residue.')
    parser.add_argument('--inpath', type=str, required=True, help='Input directory containing folders (protein names) that have trajectories')
    parser.add_argument('--outpath', type=str, required=True, help='Output directory to save the results.')
    return parser.parse_args()


args = parse_args()
inpath = args.inpath
outpath = args.outpath
console =Console()


def extract_residue_gaussians_atlas():
    # proteins = [prot for prot in os.listdir(inpath) if os.path.isdir(os.path.join(inpath, prot))]
    proteins = np.load(outpath+"/atlas_proteins.npy")
    print(proteins, proteins.shape)
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Extracting residue gaussians from ATLAS dataset...", total=len(proteins))
        with multiprocessing.Pool(processes=40) as pool:
            for protein in pool.imap(process_one_trajectory, proteins):
                progress.advance(task_id)
                console.print(f"[cyan]Processed {protein}")


def process_one_trajectory(prot):   # just taking R1 traj and topology file as pdb. Is this valid??
    traj = md.load(os.path.join(inpath, prot, prot+"_prod_R1_fit.xtc"),top=os.path.join(inpath,prot, prot+".pdb"))
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
        xyz = np.mean(traj.xyz[:, atom_indices, :],axis=1)  # shape (n_frames, 3)  frames by residue i's position (mean pos of atoms) 

        # Compute mean and variance across all frames for each atom
        means[i] = np.mean(xyz, axis=0)  # shape (1, 3)

        variances[i] = np.cov(xyz.T, rowvar=True)    # shape (n_atoms_in_residue, 3, 3)



    return means, variances


# process_one_trajectory("2pmb_C")
extract_residue_gaussians_atlas()
