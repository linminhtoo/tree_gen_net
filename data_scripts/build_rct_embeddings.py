import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import smi_to_bit_fp

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv")
    parser.add_argument("--path_fps", type=Path, default="data/rct_fps.npz")
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--output_radius", type=int, default=2)
    args = parser.parse_args()

    # load valid building blocks
    df_matched = pd.read_csv(args.path_csv_matched_rcts)
    rct_smis = df_matched.SMILES.tolist()

    bit_fps = []
    for rct_smi in tqdm(rct_smis):
        bit_fp = smi_to_bit_fp(rct_smi, radius=args.output_radius, fp_size=args.output_dim)
        bit_fp = sparse.csr_matrix(bit_fp, dtype="int32")
        bit_fps.append(bit_fp)

    bit_fps = sparse.vstack(bit_fps)
    sparse.save_npz(args.path_fps, bit_fps)

    print(f"verify length of bit_fps: {bit_fps.shape[0]}")
    print(f"number of building blocks: {len(rct_smis)}")