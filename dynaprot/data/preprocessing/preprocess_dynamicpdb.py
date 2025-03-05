import os
import glob
import shutil
import subprocess
import multiprocessing
import numpy as np


# Paths
raw_path = "/data/cb/scratch/datasets/dynamicPDB_raw"
out_path = "/data/cb/mihirb14/projects/dynamicPDB"

# Get all protein directories
proteins = [file for file in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, file))]

def extract_combine_one_protein(prot):
    """
    Extracts compressed files, merges segmented DCD files, and generates a single DCD trajectory.
    All temporary files and extracted content are placed in `out_path`, keeping `raw_path` clean.

    Args:
        prot (str): The protein ID (e.g., "1a62_A").

    Returns:
        str: Path to the final merged DCD file, or None if no DCD files were found.
    """
    prot_path = os.path.join(raw_path, prot)
    output_prot_dir = os.path.join(out_path, prot) 
    os.makedirs(output_prot_dir, exist_ok=True)
    try:

        # # Step 1: Merge split-volume .tar.gz files
        tar_parts = sorted(glob.glob(os.path.join(prot_path, f"{prot}.tar.gz.part*")))
        tar_gz_path = os.path.join(prot_path, f"{prot}.tar.gz")  # store combined .tar.gz

        if tar_parts:
            print(f"Merging {len(tar_parts)} parts into {tar_gz_path}...")
            subprocess.run(f"cat {' '.join(tar_parts)} > {tar_gz_path}", shell=True, check=True)

        # # Step 2: Extract the .tar.gz archive
        print(f"Extracting {tar_gz_path} to {output_prot_dir}...")
        subprocess.run([
            "tar", "-zxvf", tar_gz_path, "-C", output_prot_dir,
            f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}.pdb",
            f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}_T.dcd"
        ], check=True)
        # except subprocess.CalledProcessError as e:
            # print(f"Error during extraction: {e}")
        # return None
        
        os.remove(tar_gz_path)  
        return prot
    except:
        os.rmdir(output_prot_dir)
        return None

with multiprocessing.Pool(10) as pool:
    results = pool.map(extract_combine_one_protein, proteins)

completed = [prot for prot in results if prot is not None]
failed = [prot for prot in results if prot is None]

print(completed)
print(failed)
np.save(os.path.join(out_path,"completed"),np.array(completed))
np.save(os.path.join(out_path,"failed"),np.array(failed))