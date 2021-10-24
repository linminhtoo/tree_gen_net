import argparse
import os
import pickle
import sys
from pathlib import Path

import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

import nmslib
import pandas as pd
import yaml
from rdkit import RDLogger
from scipy import sparse
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from decode_trees import load_models, decode_synth_tree

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv"
)
parser.add_argument("--path_templates", type=Path, default="data/templates_cleaned.txt")
parser.add_argument(
    "--path_rct_to_temps", type=Path, default="data/rct_to_temps_cleaned.pickle"
)
parser.add_argument(
    "--path_temp_to_rcts", type=Path, default="data/temp_to_rcts_cleaned.pickle"
)
parser.add_argument("--path_fps", type=Path, default="data/rct_fps.npz")
parser.add_argument("--path_index", type=Path, default="data/knn_rct_fps.index")
parser.add_argument(
    "--path_target_smis",
    type=Path,
    default="data/split/trees_test.pickle",
    help="path to .csv or .txt containing list of SMILES or .pickle containing trees",
)
parser.add_argument("--path_model_config", type=Path, default="config/models.yaml")
parser.add_argument(
    "--path_save_decoded_trees", type=Path, default="data/decoded_trees.pickle"
)
parser.add_argument("--max_steps", type=int, default=10)
parser.add_argument("--checkpoint_every", type=int, default=5000)
parser.add_argument("--ncpu", type=int, default=8)
args = parser.parse_args()

print(args)

# load valid building blocks
df_matched = pd.read_csv(args.path_csv_matched_rcts)
smis = df_matched.SMILES.tolist()

# load templates
with open(args.path_templates, "r") as f:
    template_strs = [l.strip().split("|")[1] for l in f.readlines()]

# NOTE: this has limited utility, once we start making new molecules, this dict cannot be used
with open(args.path_rct_to_temps, "rb") as f:
    rct_to_temps = pickle.load(f)

with open(args.path_temp_to_rcts, "rb") as f:
    temp_to_rcts = pickle.load(f)

# load building block embeddings (fingerprints)
mol_fps = sparse.load_npz(args.path_fps)
mol_fps = mol_fps.toarray()

# load building block kNN search index
index_all_mols = nmslib.init(method="hnsw", space="cosinesimil")
index_all_mols.loadIndex(str(args.path_index), load_data=True)

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

with open(args.path_model_config, "r") as stream:
    model_config = yaml.safe_load(stream)

# load 4 trained models from checkpoints
f_act, f_rt1, f_rt2, f_rxn = load_models(model_config)
f_act.share_memory()
f_rt1.share_memory()
f_rt2.share_memory()
f_rxn.share_memory()
print(f"finished loading 4 models from checkpoints")


def decode_one_tree(target_smi):
    tree = decode_synth_tree(
        f_act=f_act,
        f_rt1=f_rt1,
        f_rt2=f_rt2,
        f_rxn=f_rxn,
        target_smi=target_smi,
        target_z=None,
        mol_fps=mol_fps,
        smis=smis,
        index_all_mols=index_all_mols,
        template_strs=template_strs,
        temp_to_rcts=temp_to_rcts,
        rct_to_temps=rct_to_temps,
        input_dim=model_config["input_fp_dim"],
        radius=model_config["radius"],
        t_max=args.max_steps,
    )
    return tree


if __name__ == "__main__":
    # pool must be guarded by __main__
    p = Pool(args.ncpu)

    # run the decoding, multiprocess
    trees = []
    cnt_success, cnt_fail = 0, 0
    for tree in tqdm(p.imap(decode_one_tree, target_smis), total=len(target_smis)):
        if tree:
            cnt_success += 1
            trees.append(tree)

            if cnt_success > 0 and cnt_success % args.checkpoint_every == 0:
                # checkpoint trees
                with open(args.path_save_decoded_trees, "wb") as f:
                    pickle.dump(trees, f)

        else:
            cnt_fail += 1
            trees.append(
                None
            )  # needed to maintain pairing of target_smi to decoded tree!!

    print(f"num targets: {len(target_smis)}")
    print(f"num success: {cnt_success} ({cnt_success / len(target_smis) * 100:.2f}%)")
    print(f"num fail: {cnt_fail} ({cnt_fail / len(target_smis) * 100:.2f}%)")

    # save decoded trees for evaluation & metric calculation
    with open(args.path_save_decoded_trees, "wb") as f:
        pickle.dump(trees, f)
