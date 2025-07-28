import torch
from openfold.np.protein import Protein
from openfold.utils.rigid_utils import  Rigid
from typing import Optional
import dataclasses
import io
from typing import Any, Sequence, Mapping, Optional
import re
import string
from openfold.np import residue_constants
from Bio.PDB import PDBParser
import numpy as np
import mdtraj as md
from torch.distributions.multivariate_normal import MultivariateNormal
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.PDBIO import PDBIO
from .utils_secondary import assign_secondary_structures
import os
import shutil

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
assert(PDB_MAX_CHAINS == 62)


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)
            
    return new_dict

def find_rot_trans(a, b, weights=None):
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return C.mT,a_mean,b_mean

#https://github.com/scipy/scipy/blob/main/scipy/spatial/transform/_rotation.pyx
def rmsdalign(a, b, weights=None): # alignes B to A  # [*, N, 3]
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    weights = weights.unsqueeze(-1)
    a_mean = (a * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    a = a - a_mean
    b_mean = (b * weights).sum(-2, keepdims=True) / weights.sum(-2, keepdims=True)
    b = b - b_mean
    B = torch.einsum('...ji,...jk->...ik', weights * a, b)
    u, s, vh = torch.linalg.svd(B)

    # Correct improper rotation if necessary (as in Kabsch algorithm)
    '''
    if torch.linalg.det(u @ vh) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]
    '''
    sgn = torch.sign(torch.linalg.det(u @ vh))
    s[...,-1] *= sgn
    u[...,:,-1] *= sgn.unsqueeze(-1)
    C = u @ vh # c rotates B to A
    return b @ C.mT + a_mean
    
def kabsch_rmsd(a, b, weights=None):    # from AlphaFlow
    B = a.shape[:-2]
    N = a.shape[-2]
    if weights == None:
        weights = a.new_ones(*B, N)
    b_aligned = rmsdalign(a, b, weights)
    out = torch.square(b_aligned - a).sum(-1)
    out = (out * weights).sum(-1) / weights.sum(-1)
    return torch.sqrt(out)


def sample_ensemble(mean_3n, cov_3n, num_samples=10):
    device = cov_3n.device
    dtype = cov_3n.dtype
    N = cov_3n.shape[0] // 3
    print(mean_3n.shape, cov_3n.shape)
    if mean_3n is None:
        mean_3n = torch.zeros(3 * N, device=device, dtype=dtype)

    mvn = MultivariateNormal(loc=mean_3n, covariance_matrix=cov_3n)
    samples = mvn.rsample((num_samples,))
    samples = samples.view(num_samples, N, 3)
    return samples



def save_ensemble_with_ss(
    coord_ensemble: torch.Tensor,
    ref_pdb_path: str,
    out_pdb_path: str,
    assign_ss_fn = assign_secondary_structures,  # Your `assign_secondary_structures` function
):
    """
    Save an ensemble of structures using only Cα atoms and annotate secondary structure using a custom assignment function.

    Args:
        coord_ensemble (torch.Tensor): Tensor of shape (num_samples, N, 3), in Ångström.
        ref_pdb_path (str): Path to a reference PDB file with correct topology.
        out_pdb_path (str): Output path for the ensemble PDB.
        assign_ss_fn (function): Function to assign secondary structure from coordinates.
    """
    coord_ensemble = coord_ensemble.cpu().numpy()
    num_models, N, _ = coord_ensemble.shape

    ref = md.load_pdb(ref_pdb_path)
    ss_assignments =list( md.compute_dssp(ref)[0])
    print(ss_assignments, len(ss_assignments))

    # Only use Cα atoms
    ca_indices = ref.topology.select("name CA")
    ref_calpha = ref.atom_slice(ca_indices)
    residues = list(ref_calpha.topology.residues)

    if len(ca_indices) != coord_ensemble.shape[1]:
        raise ValueError(f"Reference has {len(ca_indices)} Cα atoms, but ensemble has {coord_ensemble.shape[1]} residues")

    lines = []

    # Write HELIX/SHEET records manually for model 1
    current_ss = None
    start = None

    for i, ss in enumerate(ss_assignments + ["END"]):  # Add sentinel
        if ss != current_ss:
            if current_ss in ["H", "E"]:
                record_type = "HELIX" if current_ss == "H" else "SHEET"
                end = i
                res_start = residues[start].resSeq
                res_end   = residues[end - 1].resSeq
                chain_id = string.ascii_uppercase[residues[start].chain.index] 
                lines.append(f"{record_type:<6}    1 {chain_id} {res_start:>4}    {chain_id} {res_end:>4}")
            current_ss = ss
            start = i

    # Write each model
    atom_line_format = (
        "ATOM  {atom_idx:5d}  CA  {resname:>3} {chain_id}{resid:4d}    "
        "{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
    )

    resnames  = [res.name for res in residues]
    resids    = [res.resSeq for res in residues]
    chain_ids = [string.ascii_uppercase[res.chain.index]   for res in residues]
    
    for model_idx in range(num_models):
        lines.append(f"MODEL     {model_idx + 1}")
        for i in range(N):
            coord = coord_ensemble[model_idx, i]
            lines.append(atom_line_format.format(
                atom_idx=i + 1,
                resname=resnames[i],
                chain_id=chain_ids[i],
                resid=resids[i],
                x=coord[0].item(),
                y=coord[1].item(),
                z=coord[2].item()
            ))
        lines.append("ENDMDL")

    lines.append("END")

    with open(out_pdb_path, "w") as f:
        f.write("\n".join(lines) + "\n")


import os
import subprocess
from pathlib import Path
import mdtraj as md

def split_models(input_pdb, temp_dir):
    """Split multi-model PDB into individual models."""
    with open(input_pdb, 'r') as f:
        lines = f.readlines()

    models = []
    current_model = []
    for line in lines:
        if line.startswith("MODEL"):
            current_model = []
        elif line.startswith("ENDMDL"):
            models.append(current_model[:])
        else:
            current_model.append(line)

    paths = []
    for i, model in enumerate(models):
        model_path = temp_dir / f"model_{i+1}.pdb"
        with open(model_path, 'w') as f:
            f.write("".join(model) + "END\n")
        paths.append(model_path)

    return paths

def run_pulchra(pdb_path):
    """Run Pulchra on a single Cα-only PDB."""
    result = subprocess.run(
        ["pulchra", str(pdb_path)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pulchra failed on {pdb_path}: {result.stderr}")
    return pdb_path.with_suffix(".rebuilt.pdb")

def merge_models(pdb_paths, out_path):
    """Merge single-model PDBs into a multi-model PDB."""
    with open(out_path, 'w') as out_f:
        for i, pdb in enumerate(pdb_paths):
            with open(pdb, 'r') as f:
                atoms = [l for l in f if l.startswith("ATOM")]
            out_f.write(f"MODEL     {i + 1}\n")
            out_f.writelines(atoms)
            out_f.write("ENDMDL\n")
        out_f.write("END\n")

def reconstruct_ensemble_with_pulchra(input_pdb, output_pdb, save_individual_models=False, individual_out_dir=None):
    temp_dir = Path("tmp_pulchra")
    temp_dir.mkdir(exist_ok=True)

    if save_individual_models:
        individual_out_dir = Path(individual_out_dir or output_pdb).with_suffix('')  # default to output_pdb stem
        individual_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] Splitting ensemble: {input_pdb}")
        model_paths = split_models(input_pdb, temp_dir)

        print(f"[INFO] Running Pulchra on {len(model_paths)} models...")
        rebuilt_paths = []
        for i, p in enumerate(model_paths):
            rebuilt = run_pulchra(p)
            rebuilt_paths.append(rebuilt)

            if save_individual_models:
                dest = individual_out_dir / f"{Path(input_pdb).stem}_model_{i+1:03d}.pdb"
                shutil.copyfile(rebuilt, dest)

        print(f"[INFO] Merging rebuilt models into: {output_pdb}")
        merge_models(rebuilt_paths, output_pdb)

    finally:
        print("[INFO] Cleaning up temp files...")
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()


# def save_ensemble(
#     coord_ensemble: torch.Tensor,
#     ref_pdb_path: str,
#     out_pdb_path: str,
#     transfer_secondary_struc: bool = True,
# ):
#     """
#     Save an ensemble of structures using only Cα atoms.

#     Args:
#         coord_ensemble (torch.Tensor): Tensor of shape (num_samples, N, 3), in Ångström.
#         ref_pdb_path (str): Path to a reference PDB file with correct topology.
#         out_pdb_path (str): Output path for the ensemble PDB.
#         transfer_secondary_struc (bool): Whether to assign secondary structure using DSSP from ref PDB.
#     """
#     coord_ensemble = coord_ensemble.cpu().numpy()
#     ref = md.load_pdb(ref_pdb_path)

#     ca_indices = ref.topology.select("name CA")
#     ref_calpha = ref.atom_slice(ca_indices)

#     if len(ca_indices) != coord_ensemble.shape[1]:
#         raise ValueError(f"Reference has {len(ca_indices)} Cα atoms, but ensemble has {coord_ensemble.shape[1]} residues")

#     # Save coordinates (first model only) with SS annotation
#     coords = coord_ensemble[0]

#     ss_annotation = assign_secondary_structures(torch.from_numpy(ref.xyz*10.0), return_encodings=False)  
#     print(ss_annotation)
#     # Build HELIX and SHEET lines
#     header_lines = []
#     if transfer_secondary_struc and ss_annotation:
#         def find_regions(label):
#             regions = []
#             start = None
#             for i, c in enumerate(ss_annotation):
#                 if c == label and start is None:
#                     start = i
#                 elif c != label and start is not None:
#                     if i - start >= 3:
#                         regions.append((start, i - 1))
#                     start = None
#             if start is not None and len(ss_annotation) - start >= 3:
#                 regions.append((start, len(ss_annotation) - 1))
#             return regions

#         helix_regions = find_regions('H')
#         sheet_regions = find_regions('E')

#         for i, (start, end) in enumerate(helix_regions, 1):
#             res_start = ref_calpha.topology.residue(start)
#             res_end = ref_calpha.topology.residue(end)
#             header_lines.append(
#                 f"HELIX  {i:>3}  H{i:>3} {res_start.name:>3} A{res_start.resSeq:>4}  {res_end.name:>3} A{res_end.resSeq:>4}  1 {end-start+1:>30}"
#             )

#         for i, (start, end) in enumerate(sheet_regions, 1):
#             res_start = ref_calpha.topology.residue(start)
#             res_end = ref_calpha.topology.residue(end)
#             header_lines.append(
#                 f"SHEET  {i:>3}  S{i:>3} 1 {res_start.name:>3} A{res_start.resSeq:>4}  {res_end.name:>3} A{res_end.resSeq:>4}  0"
#             )

#     # Save annotated PDB manually
#     os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)
#     with open(out_pdb_path, "w") as f:
#         for line in header_lines:
#             f.write(line + "\n")

#         for i, atom in enumerate(ref_calpha.topology.atoms):
#             x, y, z = coords[i]
#             f.write(
#                 f"ATOM  {i+1:5d}  CA  {atom.residue.name:>3} A{atom.residue.resSeq:>4}    "
#                 f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
#             )
#         f.write("END\n")
        

    
def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> dict:
    """Takes a single chain PDB string and constructs a dictionary of protein chain features.
        Modifies the original method from openfold.np.protein

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the whole pdb file is parsed. If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      dictionary of protein features
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if(chain_id is not None and chain.id != chain_id):
            continue

        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[
                    residue_constants.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue

            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # parents = None
    # parents_chain_index = None
    # if("PARENT" in pdb_str):
    #     parents = []
    #     parents_chain_index = []
    #     chain_id = 0
    #     for l in pdb_str.split("\n"):
    #         if("PARENT" in l):
    #             if(not "N/A" in l):
    #                 parent_names = l.split()[1:]
    #                 parents.extend(parent_names)
    #                 parents_chain_index.extend([
    #                     chain_id for _ in parent_names
    #                 ])
    #             chain_id += 1

    # unique_chain_ids = np.unique(chain_ids)
    # chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    # chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return dict(
        aatype= np.array(aatype),
        all_atom_positions= np.array(atom_positions),
        all_atom_mask=np.array(atom_mask),
        residue_index=np.array(residue_index),
        # chain_index=chain_index,
        b_factors=np.array(b_factors),
        # parents=parents,
        # parents_chain_index=parents_chain_index,
    )
    
    
def map_one_protein_local_frame(prot_frames, prot_covars, prot_fullcovar = None):
    """Maps the per residue covariance matrices of a single protein to its local frames."""
    # prot_frames, prot_covars, prot_fullcovar = feats["frames"].double(), feats["dynamics_covars"].double(), feats["dynamics_fullcovar"].double()
    
    # print(prot_frames.shape[0], prot_covars.shape[0])
    assert prot_frames.shape[0] == prot_covars.shape[0]
    # rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices
    rotations  =  prot_frames[..., :3, :3]
    local_covars = torch.einsum("nij,njk,nlk->nil", rotations, prot_covars, rotations)
    # feats["dynamics_covars"] = local_covars

    # align full covar
    local_fullcovar = None
    if prot_fullcovar is not None:
        R_block = torch.block_diag(*[rotations[i] for i in range(rotations.shape[0])])
        local_fullcovar = R_block @ prot_fullcovar @ R_block.transpose(-1, -2)
    # feats["dynamics_fullcovar"] = local_fullcovar

    # return feats
    return local_covars, local_fullcovar


def map_one_protein_global_frame(prot_frames, prot_covars, prot_fullcovar = None):
    """Maps the per residue covariance matrices of a single protein to its global frames."""
    # prot_frames, prot_covars, prot_fullcovar = feats["frames"].double(), feats["dynamics_covars"].double(), feats["dynamics_fullcovar"].double()
    
    # print(prot_frames.shape[0], prot_covars.shape[0])
    assert prot_frames.shape[0] == prot_covars.shape[0]
    # rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices
    rotations  =  prot_frames[..., :3, :3]
    global_covars = torch.einsum("nji,njk,nkl->nil", rotations, prot_covars, rotations)
    # feats["dynamics_covars"] = global_covars

    # align full covar
    global_fullcovar = None
    if prot_fullcovar is not None:
        R_block = torch.block_diag(*[rotations[i] for i in range(rotations.shape[0])])
        global_fullcovar = R_block.transpose(-1, -2) @ prot_fullcovar @ R_block
    # feats["dynamics_fullcovar"] = global_fullcovar
    # return feats
    return global_covars, global_fullcovar

