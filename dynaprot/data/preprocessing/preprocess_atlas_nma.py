import mdtraj as md
import pandas as pd
import numpy as np
import os
import multiprocessing
from rich.progress import Progress
from rich.console import Console
import argparse
import random
import time
from prody import ANM, calcCovariance
from contextlib import redirect_stdout, redirect_stderr
import contextlib
import io



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
np.random.seed(seed)
random.seed(seed)

def atlas_nma():
    tic = time.time()
    proteins = np.load("dynaprot/data/preprocessing/protein_lists/atlas_proteins.npy")

    total_chains = len(proteins)
    with Progress() as progress:
        task = progress.add_task(f"[cyan]computing NMA for ATLAS chains (0/{total_chains})...", total=total_chains)
        with multiprocessing.Pool(processes=40) as pool:
            for i,protein in enumerate(pool.imap(compute_anm_covariance, proteins)):
                progress.update(task, advance=1,description=f"[cyan]computing NMA for ATLAS chains ({i}/{total_chains})... completed {protein}")

        tim = round((time.time() - tic)/60,2)
        progress.update(task, advance=1,description=f"[cyan]completed computing NMA for ATLAS chains ({total_chains}/{total_chains}) in {tim} min")


def compute_anm_covariance(prot, n_modes=None):
    """
    Compute 3N x 3N covariance matrix from a PDB using ANM via ProDy.

    Args:
        pdb_path (str): Path to the input PDB file.
        n_modes (int or None): Number of non-zero modes to include in covariance (default: all non-zero modes).

    Returns:
        np.ndarray: 3N x 3N covariance matrix (numpy array).
    """
    try:
        pdb_path = os.path.join(inpath, prot, f"{prot}.pdb")
        save_dir = os.path.join(outpath, prot)
        # os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{prot}_nma.npy")

        if os.path.exists(save_path):
            return prot
        
        traj = md.load_pdb(pdb_path)
        ca_indices = traj.topology.select("name CA")
        ca_coords = traj.xyz[0, ca_indices, :] * 10  # shape: (N, 3) in Å

        with contextlib.redirect_stdout(io.StringIO()):
            anm = ANM()
            anm.buildHessian(ca_coords)
            anm.calcModes(n_modes="all")

        if n_modes is None:
            n_modes = anm.numModes()

        cov = calcCovariance(anm[:n_modes])
        np.save(save_path, cov)

        return prot
    except Exception as e:
        return f"{prot} - ERROR: {e}"
    
    
atlas_nma()
# compute_anm_covariance("1fxo_G")
