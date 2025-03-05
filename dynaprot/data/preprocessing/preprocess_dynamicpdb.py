# import os
# import glob
# import shutil
# import subprocess
# import multiprocessing
# import numpy as np


# # Paths
# raw_path = "/data/cb/scratch/datasets/dynamicPDB_raw"
# out_path = "/data/cb/mihirb14/projects/dynamicPDB"

# # Get all protein directories
# proteins = [file for file in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, file))]

# def extract_combine_one_protein(prot):
#     """
#     Extracts compressed files, merges segmented DCD files, and generates a single DCD trajectory.
#     All temporary files and extracted content are placed in `out_path`, keeping `raw_path` clean.

#     Args:
#         prot (str): The protein ID (e.g., "1a62_A").

#     Returns:
#         str: Path to the final merged DCD file, or None if no DCD files were found.
#     """
#     prot_path = os.path.join(raw_path, prot)
#     output_prot_dir = os.path.join(out_path, prot) 
#     os.makedirs(output_prot_dir, exist_ok=True)
#     try:

#         print(f"🔄 Pulling {prot} from Git LFS...")
#         subprocess.run(["git", "lfs", "pull", "--include", f"{prot}/*"], cwd=raw_path, check=True)

#         tar_parts = sorted(glob.glob(os.path.join(prot_path, f"{prot}.tar.gz.part*")))
#         tar_gz_path = os.path.join(prot_path, f"{prot}.tar.gz")  # store combined .tar.gz

#         if tar_parts:
#             print(f"Merging {len(tar_parts)} parts into {tar_gz_path}...")
#             subprocess.run(f"cat {' '.join(tar_parts)} > {tar_gz_path}", shell=True, check=True)

#         # # Step 2: Extract the .tar.gz archive
#         print(f"Extracting {tar_gz_path} to {output_prot_dir}...")
#         subprocess.run([
#             "tar", "-zxvf", tar_gz_path, "-C", output_prot_dir,
#             f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}.pdb",
#             f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}_T.dcd"
#         ], check=True)
#         # except subprocess.CalledProcessError as e:
#             # print(f"Error during extraction: {e}")
#         # return None
        
#         os.remove(tar_gz_path)  
#         return prot
#     except:
#         os.rmdir(output_prot_dir)
#         return None

# with multiprocessing.Pool(10) as pool:
#     results = pool.map(extract_combine_one_protein, proteins)

# completed = [prot for prot in results if prot is not None]
# failed = [prot for prot in results if prot is None]

# print(completed)
# print(failed)
# np.save(os.path.join(out_path,"completed"),np.array(completed))
# np.save(os.path.join(out_path,"failed"),np.array(failed))

import os
import glob
import shutil
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Paths
raw_path = "/data/cb/scratch/datasets/dynamicPDB_raw"
out_path = "/data/cb/mihirb14/projects/dynamicPDB"

# Get all protein directories
# proteins = [file for file in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, file))]

df = pd.read_csv(os.path.join(raw_path, "PDBList.csv"))
proteins = df.iloc[:, 0].astype(str).tolist()[:5]  # 

# Log file paths
completed_log = os.path.join(out_path, "completed_proteins.txt")
failed_log = os.path.join(out_path, "failed_proteins.txt")

def extract_combine_one_protein(prot):
    """
    Pulls the specific protein using Git LFS, extracts compressed files, merges segmented DCD files,
    and generates a single DCD trajectory. Cleans up failed extractions.
    
    Args:
        prot (str): The protein ID (e.g., "1a62_A").
    
    Returns:
        str: Protein ID if successfully processed, None if extraction failed.
    """
    prot_path = os.path.join(raw_path, prot)
    output_prot_dir = os.path.join(out_path, prot)  
    os.makedirs(output_prot_dir, exist_ok=True)

    relative_pdb_path = f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}.pdb"
    relative_dcd_path = f"simulate/raw/{prot}_npt100000.0_ts0.001/{prot}_T.dcd"
    tar_gz_path = os.path.join(prot_path, f"{prot}.tar.gz")

    # if extracted files already exist.
    if os.path.exists(os.path.join(output_prot_dir,relative_dcd_path)) and os.path.exists(os.path.join(output_prot_dir,relative_pdb_path)):
        with open(completed_log, "a") as f:
            f.write(prot + "\n")
        return prot

    try:
        # Step 1: Run `git lfs pull` for the specific protein
        if not os.path.exists(tar_gz_path):
            print(f"git lfs pull {prot}")
            subprocess.run(["git", "lfs", "pull", "--include", f"{prot}/*"], cwd=raw_path, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Step 2: Merge split-volume .tar.gz files
            tar_parts = sorted(glob.glob(os.path.join(prot_path, f"{prot}.tar.gz.part*")))
            print(f"cat separate .tar.gz {prot} into {tar_gz_path}")
            if tar_parts:
                subprocess.run(f"cat {' '.join(tar_parts)} > {tar_gz_path}", shell=True, check=True)
                # for part in tar_parts:
                #     os.remove(part)

        # Step 3: Extract the .tar.gz archive
        print(f"extracting {prot} dcd and pdb files")
        subprocess.run([
            "tar", "-zxvf", tar_gz_path, "-C", output_prot_dir,
            f"{prot}_npt100000.0_ts0.001/{prot}.pdb",
            f"{prot}_npt100000.0_ts0.001/{prot}_T.dcd",
            relative_pdb_path,
            relative_dcd_path
        ], check=True)

        # Step 4: Remove the tar.gz file after extraction to save space
        # os.remove(tar_gz_path)

        # Log completion
        with open(completed_log, "a") as f:
            f.write(prot + "\n")

        return prot

    except subprocess.CalledProcessError:
        # Log failure
        with open(failed_log, "a") as f:
            f.write(prot + "\n")
        os.rmdir(output_prot_dir)
        # shutil.rmtree(output_prot_dir, ignore_errors=True)  # Remove partially extracted files
        return None

# Parallel execution with `rich` progress bar
def parallel_extract():
    num_workers = min(10, multiprocessing.cpu_count())  # Limit parallel extractions
    completed = []
    failed = []

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        transient=False,
    ) as progress:

        task = progress.add_task("[cyan]Extracting Proteins...", total=len(proteins))

        with multiprocessing.Pool(num_workers) as pool:
            for result in pool.imap_unordered(extract_combine_one_protein, proteins):
                progress.update(task, advance=1)
                if result:
                    completed.append(result)
                else:
                    failed.append(result)

    print(f"Completed {len(completed)} proteins.")
    print(f"Failed {len(failed)} proteins. Check logs for details.")

    # Save results as numpy arrays
    np.save(os.path.join(out_path, "completed.npy"), np.array(completed))
    np.save(os.path.join(out_path, "failed.npy"), np.array(failed))

if __name__ == "__main__":
    # print(proteins)
    # parallel_extract()
    extract_combine_one_protein("1aol_A")
    # extract_combine_one_protein("2j6a_A")
    
