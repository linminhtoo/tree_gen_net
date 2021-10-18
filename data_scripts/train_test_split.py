import os
import pickle
import random
import sys
from pathlib import Path

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import seed_everything

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_trees", type=Path, default="data/trees_filtered.pickle")
    parser.add_argument("--path_train", type=Path, default="data/trees_train.pickle")
    parser.add_argument("--path_val", type=Path, default="data/trees_val.pickle")
    parser.add_argument("--path_test", type=Path, default="data/trees_test.pickle")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--pct_train", type=float, default=0.6)
    parser.add_argument("--pct_val", type=float, default=0.2)
    parser.add_argument("--pct_test", type=float, default=0.2)
    args = parser.parse_args()

    assert args.pct_train + args.pct_val + args.pct_test == 1, \
        "please ensure args.pct_train/val/test add up to 1"

    # load filtered trees
    with open(args.path_trees, 'rb') as f:
        trees = pickle.load(f)

    # seed & shuffle
    seed_everything(args.seed)
    random.shuffle(trees)

    # split
    TOTAL = len(trees)
    trees_train = trees[:int(TOTAL * args.pct_train)]
    trees_val = trees[int(TOTAL * args.pct_train): int(TOTAL * (args.pct_train + args.pct_val))]
    trees_test = trees[int(TOTAL * (args.pct_train + args.pct_val)):]

    print(f"total: {TOTAL}")
    print(f"num train: {len(trees_train)}")
    print(f"num val: {len(trees_val)}")
    print(f"num test: {len(trees_test)}")

    # save
    with open(args.path_train, 'wb') as f:
        pickle.dump(trees_train, f)
    with open(args.path_val, 'wb') as f:
        pickle.dump(trees_val, f)
    with open(args.path_test, 'wb') as f:
        pickle.dump(trees_test, f)