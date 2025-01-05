from Bio import PDB
import os
from rich.progress import Progress
import multiprocessing
from rich.console import Console
import numpy as np


console =Console()


mmcif_path = "/data/cb/scratch/datasets/pdb_mmcif"
chain_path = "/data/cb/scratch/datasets/pdb_chains"



proteins = np.load("/data/cb/mihirb14/projects/dynaprot/dynaprot/data/preprocessing/protein_lists/atlas_proteins.npy")    # atlas
# proteins = np.load("/data/cb/mihirb14/projects/dynaprot/dynaprot/data/preprocessing/protein_lists/pdb_proteins.npy")    # atlas
total_chains = len(proteins)


def parse_one_protein(protein_id):
    if "_" in protein_id:
        protein_name,chain = protein_id.split("_")
    else:
        protein_name = protein_id
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', os.path.join(mmcif_path,f"{protein_name[1:3]}/{protein_name}.cif"))  # Replace with your CIF file path
    # Create an instance of PDBIO to save chains separately if needed
    pdb_io = PDB.PDBIO()
    mmcif_io = PDB.MMCIFIO()
    # print(structure)

    # Iterate through models and chains
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            # print(f"Processing chain {chain_id}...")

            new_structure = PDB.Structure.Structure(chain_id)
            new_model = PDB.Model.Model(0)
            new_model.add(chain.copy())
            new_structure.add(new_model)

            pdb_io.set_structure(new_structure)
            pdb_io.save(os.path.join(chain_path,f"{protein_id[1:3]}/{protein_name}_{chain_id}.pdb"))
            mmcif_io.set_structure(new_structure)
            mmcif_io.save(os.path.join(chain_path,f"{protein_id[1:3]}/{protein_name}_{chain_id}.cif"))
    
    return protein_id


with Progress() as progress:
    # task = progress.add_task("[cyan]Parsing proteins...", total=len(proteins))
    task = progress.add_task(f"[cyan]Processing chains (0/{total_chains})...", total=total_chains)
    with multiprocessing.Pool(processes=40) as pool:
        for i, protein in enumerate(pool.imap(parse_one_protein, proteins)):
            progress.update(
                task, 
                advance=1, 
                description=f"[cyan]Processing chains ({i}/{total_chains})... Completed {protein}"
            )