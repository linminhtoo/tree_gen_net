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

    assert (
        args.pct_train + args.pct_val + args.pct_test == 1
    ), "please ensure args.pct_train/val/test add up to 1"

    # load filtered trees
    with open(args.path_trees, "rb") as f:
        trees = pickle.load(f)

    # seed & shuffle
    seed_everything(args.seed)
    random.shuffle(trees)

    # split
    TOTAL = len(trees)
    trees_train = trees[: int(TOTAL * args.pct_train)]
    trees_val = trees[
        int(TOTAL * args.pct_train) : int(TOTAL * (args.pct_train + args.pct_val))
    ]
    trees_test = trees[int(TOTAL * (args.pct_train + args.pct_val)) :]

    print(f"total: {TOTAL}")
    print("#" * 20, f"initial random shuffle", "#" * 20)
    print(f"num train: {len(trees_train)}")
    print(f"num val: {len(trees_val)}")
    print(f"num test: {len(trees_test)}")

    # check how many final product molecules in test trees overlap with any molecule in train / val trees
    smis_seen = set()
    for tree_list in [trees_val, trees_train]:
        for tree in tree_list:
            for mol_node in tree.molecules:
                smis_seen.add(mol_node.smi)
    print(
        f"no. of unique molecules at any part of train + val trees --> {len(smis_seen)}"
    )

    # the duplicate rate should be ~0.4%
    dup_cnt = 0
    trees_test_dedup = []
    for tree in trees_test:
        for mol_node in [tree.molecules[-1]]:
            if mol_node.smi in smis_seen:
                dup_cnt += 1
                # put in validation set instetad
                trees_val.append(tree)
            else:
                trees_test_dedup.append(tree)
    print(
        f"no. of test trees with final product already seen in a train or val tree --> {dup_cnt}"
    )
    if dup_cnt / len(trees_test) > 0.05:
        raise ValueError(
            f"ERROR: too many test trees (>5%) have final product seen in train or val trees. \
            Please recheck!"
        )

    random.shuffle(trees_val)
    print("#" * 20, f"after de-duplication", "#" * 20)
    print(f"num train: {len(trees_train)}")
    print(f"num val: {len(trees_val)}")
    print(f"num test: {len(trees_test_dedup)}")

    # save
    (args.path_train.parent).mkdir(exist_ok=True, parents=True)
    with open(args.path_train, "wb") as f:
        pickle.dump(trees_train, f)

    (args.path_val.parent).mkdir(exist_ok=True, parents=True)
    with open(args.path_val, "wb") as f:
        pickle.dump(trees_val, f)

    (args.path_test.parent).mkdir(exist_ok=True, parents=True)
    with open(args.path_test, "wb") as f:
        pickle.dump(trees_test_dedup, f)
