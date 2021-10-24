import argparse
import os
import sys
from pathlib import Path

from scipy import sparse

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_templates", type=Path, default="data/templates_cleaned.txt"
    )
    parser.add_argument("--path_steps", type=Path, default="data/steps.npz")
    parser.add_argument("--path_states", type=Path, default="data/states.npz")
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--output_dim", type=int, default=256)
    args = parser.parse_args()

    IN_DIM = args.input_dim
    OUT_DIM = args.output_dim

    # load templates
    with open(args.path_templates, "r") as f:
        template_strs = [l.strip().split("|")[1] for l in f.readlines()]

    # load the sparse csc matrices
    states = sparse.load_npz(args.path_states)
    steps = sparse.load_npz(args.path_steps)

    # prepare output filename prefixes
    state_prefix = str(args.path_states).replace(".npz", "")
    step_prefix = str(args.path_steps).replace(".npz", "")

    ############################################
    # f_act (action selection network)
    # input: z_state + z_target
    # groundtruth: action

    z_state = states[:, : IN_DIM * 2]
    z_target = states[:, IN_DIM * 2 : IN_DIM * 3]
    X = sparse.hstack([z_state, z_target])

    # one-hot encoded
    action = steps[:, :4]
    y = action
    # for f_act, all rows are useful, no need to mask anything

    sparse.save_npz(f"{state_prefix}_f_act.npz", X)
    sparse.save_npz(f"{step_prefix}_f_act.npz", y)
    print(f"saved X f_act: {state_prefix}_f_act.npz --> length: {X.shape[0]}")
    print(f"saved y f_act: {step_prefix}_f_act.npz --> length: {y.shape[0]}")

    ############################################
    # f_rt1 (reactant #1 prediction network)
    # input: z_state + z_target
    # groundtruth: a_rt1

    z_state = states[:, : IN_DIM * 2]
    z_target = states[:, IN_DIM * 2 : IN_DIM * 3]
    X = sparse.hstack([z_state, z_target])

    a_rt1 = steps[:, 4 : 4 + OUT_DIM]
    rct1_idx = steps[:, -2]
    y = sparse.hstack([a_rt1, rct1_idx])

    # for f_rt1, we just want rows with action == 0 (ADD)
    keep = action[:, 0].getnnz(1) > 0
    y = y[keep]
    X = X[keep]

    sparse.save_npz(f"{state_prefix}_f_rt1.npz", X)
    sparse.save_npz(f"{step_prefix}_f_rt1.npz", y)
    print(f"saved X f_rt1: {state_prefix}_f_rt1.npz --> length: {X.shape[0]}")
    print(f"saved y f_rt1: {step_prefix}_f_rt1.npz --> length: {y.shape[0]}")

    ############################################
    # f_rxn (reaction selection network)
    # input: z_state + z_target + z_rt1
    # groundtruth: a_rxn

    z_state = states[:, : IN_DIM * 2]
    z_target = states[:, IN_DIM * 2 : IN_DIM * 3]
    z_rt1 = states[:, IN_DIM * 3 : IN_DIM * 4]
    X = sparse.hstack([z_state, z_target, z_rt1])

    # one-hot encoded
    a_rxn = steps[:, 4 + OUT_DIM : 4 + OUT_DIM + len(template_strs)]
    y = a_rxn

    # remove rows where all(y) == 0 (aka action == 3, END)
    keep = y.getnnz(1) > 0
    y = y[keep]
    X = X[keep]

    sparse.save_npz(f"{state_prefix}_f_rxn.npz", X)
    sparse.save_npz(f"{step_prefix}_f_rxn.npz", y)
    print(f"saved X f_rxn: {state_prefix}_f_rxn.npz --> length: {X.shape[0]}")
    print(f"saved y f_rxn: {step_prefix}_f_rxn.npz --> length: {y.shape[0]}")

    ############################################
    # f_rt2 (reactant #2 prediction network)
    # input: z_state + z_target + z_rt1 + z_rxn
    # groundtruth: a_rt2

    # showing the logic in comments, but essentially just take the whole of states
    # z_state = states[:, :IN_DIM * 2]
    # z_target = states[:, IN_DIM * 2:IN_DIM * 3]
    # z_rt1 = states[:, IN_DIM * 3:IN_DIM * 4]
    # z_rxn = states[:, IN_DIM * 4:]
    # X = sparse.hstack([z_state, z_target, z_rt1, z_rxn])
    X = states

    a_rt2 = steps[:, 4 + OUT_DIM + len(template_strs) : -2]
    rct2_idx = steps[:, -1]
    y = sparse.hstack([a_rt2, rct2_idx])

    # for f_rt2, we just want rows with action == 0 (ADD) or action == 1 (EXPAND)
    keep = action[:, [0, 1]].getnnz(1) > 0
    y = y[keep]
    X = X[keep]

    # then remove rows with all(y) == 0 (means uni-molecular)
    keep = y.getnnz(1) > 0
    y = y[keep]
    X = X[keep]

    sparse.save_npz(f"{state_prefix}_f_rt2.npz", X)
    sparse.save_npz(f"{step_prefix}_f_rt2.npz", y)
    print(f"saved X f_rt2: {state_prefix}_f_rt2.npz --> length: {X.shape[0]}")
    print(f"saved y f_rt2: {step_prefix}_f_rt2.npz --> length: {y.shape[0]}")
