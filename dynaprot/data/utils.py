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
import modelcif
import modelcif.model
import modelcif.dumper
import modelcif.reference
import modelcif.protocol
import modelcif.alignment
import modelcif.qa_metric


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
    
    
def align_one_protein(feats):
    """Aligns the covariance matrices of a single protein to its local frames."""
    prot_frames, prot_covars = feats["frames"].double(), feats["dynamics_covars"].double()
    
    # print(prot_frames.shape[0], prot_covars.shape[0])
    assert prot_frames.shape[0] == prot_covars.shape[0]
    rotations = Rigid.from_tensor_4x4(prot_frames).get_rots().get_rot_mats().double()  # Extract rotation matrices
    aligned_covars = torch.einsum("nij,njk,nlk->nil", rotations, prot_covars, rotations)

    return aligned_covars
   