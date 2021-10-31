import argparse
import os
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np
from rdkit import DataStructs, RDLogger
from tqdm import tqdm

from data_scripts.utils import smi_to_bit_fp_raw

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def tanimoto_similarity(smi_a, smi_b, radius, fp_size):
    if smi_a is None or smi_b is None:
        return 0

    fp_a = smi_to_bit_fp_raw(smi_a, radius=radius, fp_size=fp_size)
    fp_b = smi_to_bit_fp_raw(smi_b, radius=radius, fp_size=fp_size)

    return DataStructs.FingerprintSimilarity(
        fp_a, fp_b, metric=DataStructs.TanimotoSimilarity
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_target_smis",
        type=Path,
        default="data/split/trees_test.pickle",
        help="path to .csv or .txt containing list of SMILES or .pickle containing trees",
    )
    parser.add_argument(
        "--path_decoded_trees", type=Path, default="data/decoded_trees.pickle"
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=4096)
    parser.add_argument("--ncpu", type=int, default=8)
    args = parser.parse_args()

    print(args)

    # load the target product SMILES - can be a list of SMILES (.csv/.txt) or tree (.pickle)
    target_ext = args.path_target_smis.name.split(".")[-1]
    if target_ext == "pickle":
        # target_smi are in trees
        with open(args.path_target_smis, "rb") as f:
            target_trees = pickle.load(f)
        target_smis = [tree.molecules[-1].smi for tree in target_trees]
    elif target_ext == "txt":
        # target_smi are in text file
        with open(args.path_target_smis, "r") as f:
            target_smis = [l.strip() for l in f.readlines()]
    elif target_ext == "csv":
        # target_smi are in dataframe
        df_target = pd.read_csv(args.path_target_smis)
        target_smis = df_target.SMILES.tolist()
    else:
        raise ValueError(f"unrecognized extension of --path_target_smis: {target_ext}")

    # load decoded trees & get decoded SMILES
    with open(args.path_decoded_trees, "rb") as f:
        trees_decoded = pickle.load(f)
    decoded_smis = [tree.molecules[-1].smi if tree else None for tree in trees_decoded]

    # assert len(decoded_smis) == len(target_smis)

    # calculate target-generated pairwise similarities
    p = Pool(args.ncpu)

    tanimoto_similarity_ = partial(
        tanimoto_similarity, radius=args.radius, fp_size=args.fp_size
    )
    simils = []
    for simil in tqdm(
        p.starmap(tanimoto_similarity_, zip(target_smis, decoded_smis)),
        total=len(decoded_smis),
        desc="calculating tanimoto similarities",
    ):
        simils.append(simil)

    simils = np.array(simils)
    print(
        f"average tanimoto similarity (including failed decoding): {simils.mean():.4f} (+-{simils.std():.4f})"
    )
    exact = sum(simils == 1) / len(simils)
    print(f"exact match (including failed decoding): {exact * 100:.2f}")

    success_idxs = np.array(decoded_smis) != None
    simils = simils[success_idxs]
    exact = sum(simils == 1) / len(simils)
    print(f"exact match (excluding failed decoding): {exact * 100:.2f}")
    print(
        f"average tanimoto similarity (excluding failed decoding): {simils.mean():.4f} (+-{simils.std():.4f})"
    )
