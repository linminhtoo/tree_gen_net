import argparse
import os
import pickle
import sys
from pathlib import Path

import nmslib
import pandas as pd
import torch
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdChemReactions
from scipy import sparse
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.build_knn_index import create_knn_index
from data_scripts.build_random_trees import (get_mask_action, get_mask_rct2,
                                             get_mask_reaction_merge,
                                             get_mask_rxn_and_order)
from data_scripts.synthesis_tree import SynthesisTree
from data_scripts.utils import knn_search, smi_to_bit_fp
from model.basic import BasicFeedforward

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def embed_state(state, dim=4096, radius=2):
    '''
    embeds state into fingerprints
    '''

    z_state = torch.zeros((1, dim * 2))

    if len(state) == 0:
        return z_state

    elif len(state) == 1:
        fp = smi_to_bit_fp(state[0], radius=radius, fp_size=dim)
        z_state[0, :dim] = torch.from_numpy(fp)
        return z_state

    elif len(state) == 2:
        fp_1 = smi_to_bit_fp(state[0], radius=radius, fp_size=dim)
        fp_2 = smi_to_bit_fp(state[1], radius=radius, fp_size=dim)
        z_state[0, :dim] = torch.from_numpy(fp_1)
        z_state[0, dim:] = torch.from_numpy(fp_2)
        return z_state

    else:
        raise ValueError(f"invalid length of state: {len(state)}")


def decode_synth_tree(
        f_act, f_rt1, f_rt2, f_rxn,
        target_smi, target_z,
        mol_fps, smis, index_all_mols,
        template_strs, temp_to_rcts, rct_to_temps,
        input_dim=4096, radius=2,
        t_max=10
    ):

    tree = SynthesisTree()
    most_recent_mol_smi = None

    # assume f_act will be on same device as other 3 models
    # assume all parameters of f_act are on same device
    device = next(f_act.parameters()).device

    if target_smi:
        z_target = torch.from_numpy(
            smi_to_bit_fp(target_smi, fp_size=input_dim, radius=radius)
        ).unsqueeze(0)
    elif target_z:
        z_target = torch.from_numpy(target_z).unsqueeze(0)
    else:
        raise ValueError("either target_smi or target_z must be provided")
    z_target = z_target.to(device)

    try:
        for t in range(t_max):
            state = tree.eval_state()

            z_state = embed_state(state, dim=input_dim, radius=radius)
            z_state = z_state.to(device)

            # sample a random, but valid, action
            # 0 --> "ADD", 1 --> "EXPAND", 2 --> "MERGE", 3 --> "END"
            probs_action = f_act(torch.cat([z_state, z_target], dim=1))
            mask_action = get_mask_action(state, template_strs)
            probs_action_masked = probs_action.cpu() * mask_action
            action = int(torch.argmax(probs_action_masked).item())

            if action == 3: # END
                break

            elif action == 0: # ADD
                pred_rt1 = f_rt1(torch.cat([z_state, z_target], dim=1))
                rct1_idx, _ = knn_search(pred_rt1.detach().cpu().numpy(), index_all_mols, k=1)
                rct1_smi = smis[rct1_idx[0]]

            else: # EXPAND or MERGE
                rct1_smi = most_recent_mol_smi

            z_rt1 = torch.from_numpy(
                smi_to_bit_fp(rct1_smi, fp_size=input_dim, radius=radius)
            ).unsqueeze(0)
            z_rt1 = z_rt1.to(device)

            # sample a valid reaction template (that rct1_smi can undergo)
            probs_rxn = f_rxn(torch.cat([z_state, z_target, z_rt1], dim=1))

            if action == 2:
                # if merge, the reaction must fit both sub-tree root molecules
                # thus, reaction mask has to be specially calculated
                mask_rxn = get_mask_reaction_merge(state, template_strs)
            else: # if not merge, reaction just has to fit rct1_smi, we can sample rct2_smi later
                mask_rxn, rct1_temps, _ = get_mask_rxn_and_order(rct1_smi,
                                                                template_strs,
                                                                rct_to_temps)
            probs_rxn_masked = probs_rxn.cpu() * mask_rxn
            rxn_idx = int(torch.argmax(probs_rxn_masked).item())
            rxn_str = template_strs[rxn_idx]

            z_rxn = torch.zeros((1, len(template_strs)))
            z_rxn[0, rxn_idx] = 1
            z_rxn = z_rxn.to(device)

            if sum(mask_rxn) == 0:
                # no rxns in our template library can be validly applied to rct1_smi
                if len(state) == 1:
                    # only 1 sub-tree, we can force the action to be END
                    action == 3
                    break
                else:
                    # there are two sub-trees, we cannot stop generation here (need to merge)
                    # so, this tree has an error
                    break

            rxn = AllChem.ReactionFromSmarts(rxn_str)
            rdChemReactions.ChemicalReaction.Initialize(rxn)

            # check num reactants --> uni- or bi-molecular
            if rxn.GetNumReactantTemplates() > 1:
                if action == 2: # MERGE
                    rct2_smi = set(state) - set([rct1_smi])
                    rct2_smi = rct2_smi.pop() # get element from set
                    rct2_is_second = True

                else: # ADD or EXPAND
                    # determine order of rct1 & rct2
                    if rxn_idx in rct1_temps: # first reactant is reactant #1
                        rct2_is_second = True # second reactant must be reactant #2
                    else: # first reactant is reactant #2
                        rct2_is_second = False # second reactant must be reactant #1

                    mask_rct2 = get_mask_rct2(rxn_idx, rct2_is_second,
                                            template_strs, temp_to_rcts, len(smis))
                    if sum(mask_rct2) == 0:
                        # no building block can match as reactant #2 of template
                        break

                    valid_rct2_fps = mol_fps[mask_rct2.numpy().astype(bool)]
                    masked_rt2_knn_index = create_knn_index(valid_rct2_fps)

                    pred_rt2 = f_rt2(torch.cat([z_state, z_target, z_rt1, z_rxn], dim=1))
                    rct2_idx, _ = knn_search(pred_rt2.detach().cpu().numpy(), masked_rt2_knn_index, k=1)
                    rct2_smi = smis[torch.nonzero(mask_rct2)[rct2_idx[0]].item()]

                # run the bi-molecular reaction
                rct1_mol = Chem.MolFromSmiles(rct1_smi)
                rct2_mol = Chem.MolFromSmiles(rct2_smi)

                if rct2_is_second:
                    prod_mol = rxn.RunReactants((rct1_mol, rct2_mol))[0][0] # output is tuple of tuple
                else:
                    prod_mol = rxn.RunReactants((rct2_mol, rct1_mol))[0][0]

                prod_smi = Chem.MolToSmiles(prod_mol)

            else:
                # run the uni-molecular reaction
                rct1_mol = Chem.MolFromSmiles(rct1_smi)
                prod_mol = rxn.RunReactants((rct1_mol, ))[0][0] # [0] # output is tuple of tuple
                prod_smi = Chem.MolToSmiles(prod_mol)
                rct2_smi = None

            # update the tree
            tree.execute_action(
                action, rxn_str, rct1_smi, rct2_smi, prod_smi
            )
            most_recent_mol_smi = prod_smi
    except Exception as e:
        # something wrong happened
        print(e)
        # raise e
        action = -1
        tree = None

    if t == t_max - 1 and tree:
        if len(tree.eval_state()) == 1:
            action = 3

    if action == 3:
        tree.execute_action(
            3,
            template_str=None, rct1_smi=None, rct2_smi=None,
            prod_smi=None
        )
        return tree
    else:
        # something wrong happened
        # print('error')
        return None

def load_models(config):
    if torch.cuda.is_available():
        print('GPU is available')
        cuda_available = True
    else:
        print('no GPU')
        cuda_available = False

    # action selection network
    f_act = BasicFeedforward(
        input_size=config['input_fp_dim'] * 3,
        act_fn="ReLU",
        hidden_sizes=config['f_act']['hidden_sizes'],
        output_size=4,
        dropout=config['f_act']['dropout'],
        final_act_fn="softmax"
    )
    f_act_ckpt = torch.load(config['f_act']['path_ckpt'])
    f_act.load_state_dict(f_act_ckpt['state_dict'])
    f_act = f_act.eval()
    if cuda_available:
        f_act = f_act.cuda()

    # reactant1 prediction network
    f_rt1 = BasicFeedforward(
        input_size=config['input_fp_dim'] * 3,
        act_fn="ReLU",
        hidden_sizes=config['f_rt1']['hidden_sizes'],
        output_size=config['output_fp_dim'],
        dropout=config['f_rt1']['dropout'],
        final_act_fn=None # linear activation
    )
    f_rt1_ckpt = torch.load(config['f_rt1']['path_ckpt'])
    f_rt1.load_state_dict(f_rt1_ckpt['state_dict'])
    f_rt1 = f_rt1.eval()
    if cuda_available:
        f_rt1 = f_rt1.cuda()

    # reaction selection network
    f_rxn = BasicFeedforward(
        input_size=config['input_fp_dim'] * 4,
        act_fn="ReLU",
        hidden_sizes=config['f_rxn']['hidden_sizes'],
        output_size=config['num_templates'],
        dropout=config['f_rxn']['dropout'],
        final_act_fn="softmax" # linear activation
    )
    f_rxn_ckpt = torch.load(config['f_rxn']['path_ckpt'])
    f_rxn.load_state_dict(f_rxn_ckpt['state_dict'])
    f_rxn = f_rxn.eval()
    if cuda_available:
        f_rxn = f_rxn.cuda()

    # reactant2 prediction network
    f_rt2 = BasicFeedforward(
        input_size=config['input_fp_dim'] * 4 + config['num_templates'],
        act_fn="ReLU",
        hidden_sizes=config['f_rt2']['hidden_sizes'],
        output_size=config['output_fp_dim'],
        dropout=config['f_rt2']['dropout'],
        final_act_fn=None # linear activation
    )
    f_rt2_ckpt = torch.load(config['f_rt2']['path_ckpt'])
    f_rt2.load_state_dict(f_rt2_ckpt['state_dict'])
    f_rt2 = f_rt2.eval()
    if cuda_available:
        f_rt2 = f_rt2.cuda()

    return f_act, f_rt1, f_rt2, f_rxn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv")
    parser.add_argument("--path_templates", type=Path, default="data/templates_cleaned.txt")
    parser.add_argument("--path_rct_to_temps", type=Path, default="data/rct_to_temps_cleaned.pickle")
    parser.add_argument("--path_temp_to_rcts", type=Path, default="data/temp_to_rcts_cleaned.pickle")
    parser.add_argument("--path_fps", type=Path, default="data/rct_fps.npz")
    parser.add_argument("--path_index", type=Path, default="data/knn_rct_fps.index")
    parser.add_argument("--path_target_smis", type=Path, default="data/split/trees_test.pickle",
                        help="path to .csv or .txt containing list of SMILES or .pickle containing trees")
    parser.add_argument("--path_model_config", type=Path, default="config/models.yaml")
    parser.add_argument("--path_save_decoded_trees", type=Path, default="data/decoded_trees.pickle")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=25000)
    args = parser.parse_args()

    # load valid building blocks
    df_matched = pd.read_csv(args.path_csv_matched_rcts)
    smis = df_matched.SMILES.tolist()

    # load templates
    with open(args.path_templates, 'r') as f:
        template_strs = [l.strip().split('|')[1] for l in f.readlines()]

    # NOTE: this has limited utility, once we start making new molecules, this dict cannot be used
    with open(args.path_rct_to_temps, 'rb') as f:
        rct_to_temps = pickle.load(f)

    with open(args.path_temp_to_rcts, 'rb') as f:
        temp_to_rcts = pickle.load(f)

    # load building block embeddings (fingerprints)
    mol_fps = sparse.load_npz(args.path_fps)
    mol_fps = mol_fps.toarray()

    # load building block kNN search index
    index_all_mols = nmslib.init(method='hnsw', space='cosinesimil')
    index_all_mols.loadIndex(str(args.path_index), load_data=True)

    # load the target product SMILES - can be a list of SMILES (.csv/.txt) or tree (.pickle)
    target_ext = args.path_target_smis.name.split('.')[-1]
    if target_ext == "pickle":
        # target_smi are in trees
        with open(args.path_target_smis, 'rb') as f:
            target_trees = pickle.load(f)
        target_smis = [tree.molecules[-1].smi for tree in target_trees]
    elif target_ext == "txt":
        # target_smi are in text file
        with open(args.path_target_smis, 'r') as f:
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
    print(f"finished loading 4 models from checkpoints")

    # run the decoding on single process
    trees = []
    cnt_success, cnt_fail = 0, 0
    for target_smi in tqdm(target_smis):
        tree = decode_synth_tree(
                f_act=f_act, f_rt1=f_rt1, f_rt2=f_rt2, f_rxn=f_rxn,
                target_smi=target_smi, target_z=None,
                mol_fps=mol_fps, smis=smis, index_all_mols=index_all_mols,
                template_strs=template_strs, temp_to_rcts=temp_to_rcts, rct_to_temps=rct_to_temps,
                input_dim=model_config['input_fp_dim'], radius=model_config['radius'],
                t_max=args.max_steps
            )
        if tree:
            cnt_success += 1
            trees.append(tree)

            if cnt_success > 0 and cnt_success % args.checkpoint_every == 0:
                # checkpoint trees
                with open(args.path_save_decoded_trees, 'wb') as f:
                    pickle.dump(trees, f)

        else:
            cnt_fail += 1

    print(f"num targets: {len(target_smis)}")
    print(f"num success: {cnt_success} ({cnt_success / len(target_smis) * 100:.2f}%)")
    print(f"num fail: {cnt_fail} ({cnt_fail / len(target_smis) * 100:.2f}%)")

    # save decoded trees for evaluation & metric calculation
    with open(args.path_save_decoded_trees, 'wb') as f:
        pickle.dump(trees, f)
