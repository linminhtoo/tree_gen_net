import os
import pickle
import random
import sys
from pathlib import Path

from tdc import Oracle
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import seed_everything

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_trees", type=Path, default="data/trees.pickle")
    parser.add_argument(
        "--path_trees_filtered", type=Path, default="data/trees_filtered.pickle"
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    with open(args.path_trees, "rb") as f:
        trees = pickle.load(f)

    qed = Oracle(name="QED")
    # https://github.com/mims-harvard/TDC/blob/main/tutorials/TDC_105_Oracle.ipynb

    trees_keep = []

    seed_everything(args.seed)

    for tree in tqdm(trees):
        target_smi = tree.molecules[-1].smi
        qed_val = qed(target_smi)

        if qed_val > 0.5:
            trees_keep.append(tree)

        elif random.random() < qed_val / 0.5:
            trees_keep.append(tree)

    print(f"num trees original: {len(trees)}")
    print(f"num trees kept: {len(trees_keep)}")

    with open(args.path_trees_filtered, "wb") as f:
        pickle.dump(trees_keep, f)
