import argparse
import pickle
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
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


def tree_to_steps(tree, rxn2idx, rct2idx, template_strs, in_dim, out_dim):
    states, steps = [], []
    tree_state = deque(maxlen=2)

    target_smi = tree.molecules[-1].smi
    z_target = embed_input(target_smi)

    for t, action in enumerate(tree.actions):
        if action != 3:
            rxn_node = tree.reactions[t]
            rxn_str = rxn_node.template_str
            rxn_idx = rxn2idx[rxn_str]
            # get one-hot encoding of a_rxn (aka rxn_idx)
            z_rxn = np.zeros(len(template_strs))
            z_rxn[rxn_idx] = 1

            rct1_node = rxn_node.left
            rct1_smi = rct1_node.smi
            a_rt1 = embed_output(rct1_smi)
            z_rt1 = embed_input(rct1_smi)

            # NOTE: rct1_idx is only retrieved if action == 0,
            # when we explicitly sample from building blocks
            # with other actions, we already know rct1
            if action == 0:
                rct1_idx = rct2idx[rct1_smi]
            else:
                rct1_idx = -1 # mask out

            if rxn_node.right: # bi-molecular reaction
                rct2_node = rxn_node.right
                rct2_smi = rct2_node.smi
                a_rt2 = embed_output(rct2_smi)

                # NOTE: rct2_idx is only retrieved if
                # 1) action == 0 and bi-molecular, OR
                # 2) action == 1 and bi-molecular
                # when merging (action == 2), rct2 is already known
                if action == 0 or action == 1:
                    rct2_idx = rct2idx[rct2_smi]
                else:
                    rct2_idx = -1 # mask out

            else:
                a_rt2 = np.zeros(out_dim)
                rct2_idx = -1 # mask out

        else: # need to predict "END", only has valid data for f_act
            # so, when training the other networks, need to mask out X's which are all 0's
            a_rt1 = np.zeros(out_dim)
            z_rt1 = np.zeros(in_dim)
            z_rxn = np.zeros(len(template_strs))
            a_rt2 = np.zeros(out_dim)

        # construct z_state based on tree_state
        if t == 0:
            z_state = np.zeros(in_dim * 2)
        else:
            # use tree_state to retrieve smi and then embed into z_state
            if len(tree_state) == 1:
                z_state = np.hstack(
                    (embed_input(tree_state[0].smi),
                    np.zeros(in_dim))
                )
            elif len(tree_state) == 2:
                z_state = np.hstack(
                    (embed_input(tree_state[0].smi),
                    embed_input(tree_state[1].smi))
                )
            else:
                raise ValueError(f"invalid len(state) of {len(tree_state)}")

        action_ohe = np.zeros(4)
        action_ohe[action] = 1

        state = np.hstack((z_state, z_target, z_rt1, z_rxn))
        step = np.hstack((action_ohe, a_rt1, z_rxn, a_rt2, rct1_idx, rct2_idx))

        states.append(state)
        steps.append(step)

        if action != 3:
            # tree_state depends on action, need to update it at each timestep
            prod_node = rxn_node.head
            if action == 0:
                tree_state.appendleft(prod_node)

            elif action == 1:
                tree_state.popleft()
                tree_state.appendleft(prod_node)

            elif action == 2:
                tree_state.appendleft(prod_node)
                tree_state.pop()

    return states, steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv")
    parser.add_argument("--path_templates", type=Path, default="data/templates_cleaned.txt")
    parser.add_argument("--path_trees", type=Path, default="data/trees_filtered.pickle")
    parser.add_argument("--path_steps", type=Path, default="data/steps.npz")
    parser.add_argument("--path_states", type=Path, default="data/states.npz")
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--output_dim", type=int, default=256)
    parser.add_argument("--input_radius", type=int, default=2)
    parser.add_argument("--output_radius", type=int, default=2)
    args = parser.parse_args()

    # prepare the embed functions with user-defined fingerprint size
    def embed_input(smi):
        # embeds the bit fingerprint that will be model's input
        return smi_to_bit_fp(smi, radius=args.input_radius, fp_size=args.input_dim)

    def embed_output(smi):
        # embeds the bit fingerprint that will be model's output (groundtruth)
        return smi_to_bit_fp(smi, radius=args.output_radius, fp_size=args.output_dim)

    # load templates
    with open(args.path_templates, 'r') as f:
        template_strs = [l.strip().split('|')[1] for l in f.readlines()]

    # build a helper dict to map template string to template index
    rxn2idx = {}
    for temp_idx, temp_str in enumerate(template_strs):
        rxn2idx[temp_str] = temp_idx

    # load valid building blocks
    df_matched = pd.read_csv(args.path_csv_matched_rcts)
    rct_smis = df_matched.SMILES.tolist()

    # build a helper dict to map building block SMILES to index
    rct2idx = {}
    for rct_idx, rct_smi in enumerate(rct_smis):
        rct2idx[rct_smi] = rct_idx

    # load filtered trees
    with open(args.path_trees, 'rb') as f:
        trees = pickle.load(f)

    # later we will select specific columns depending on the network to train
    states_all = [] # we will get our "X" from here
    steps_all = [] # we will get our "y" from here

    error_cnt = 0
    for tree in tqdm(trees):
        try:
            states, steps = tree_to_steps(tree, rxn2idx, rct2idx, template_strs,
                                        args.input_dim, args.output_dim)
            states_all.extend(states)
            steps_all.extend(steps)
        except:
            # a few trees have product SMILES that somehow cannot be parsed into Chem.Mol
            # from debugging, seems that they are very complex molecules
            error_cnt += 1
    print(f'error_cnt: {error_cnt}')

    # make into sparse matrix
    # csc matrix for fast column slicing operations
    # (later, we will select appropriate columns as input & groundtruth data depending on network)
    states_sparse = sparse.csc_matrix(states_all)
    steps_sparse = sparse.csc_matrix(steps_all)

    (args.path_states.parent).mkdir(parents=True, exist_ok=True)
    (args.path_steps.parent).mkdir(parents=True, exist_ok=True)

    sparse.save_npz(args.path_states, states_sparse)
    sparse.save_npz(args.path_steps, steps_sparse)

    print(f'length of states: {states_sparse.shape[0]}')
    print(f'length of steps: {steps_sparse.shape[0]}')