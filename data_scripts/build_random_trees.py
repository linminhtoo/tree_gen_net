import argparse
import pickle
import os
import sys
import random
from multiprocessing import Pool
from pathlib import Path

import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdChemReactions
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.synthesis_tree import SynthesisTree
from data_scripts.utils import is_valid_reactant, seed_everything

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def get_mask_rct2(rxn_idx, rct2_is_second,
                template_strs, temp_to_rcts, num_rcts):
    '''
    args
        rct2_is_second : bool, whether second reactant is reactant #2
    '''

    # collate all possible choices for second reactant
    rct2_idxs = set()

    temp_str = template_strs[rxn_idx]
    valid_rct2_idxs = temp_to_rcts[temp_str][int(rct2_is_second)]
    rct2_idxs.update(valid_rct2_idxs)

    # build the mask
    mask_rct2 = [0 for _ in range(num_rcts)]
    for i in rct2_idxs:
        mask_rct2[i] = 1

    return torch.Tensor(mask_rct2)

def get_mask_rxn_and_order(rct1_smi, template_strs, rct_to_temps):
    '''
    args
        rct1_smi : SMILES strㅕng of first reactant

    returns
        mask_rxn : mask of reactions where smi is a valid reactant
        valid_rct1_temps: reaction idxs where first reactant is reactant #1
        valid_rct2_temps: reaction idxs where first reactant is reactant #2

    NOTE #1:
        to increase the number of valid templates, we need to allow that
        the first reactant can be either reactant #1 OR reactant #2 of template (not just reactant #1)
        as a result, we also need to be careful in building the mask for second reactant (rct2),
        since if first reactant is reactant #2, then second reactant must be reactant #1

    NOTE #2:
        attempts to first check if rct1_smi exists in rct_to_temps (only if it is a building block),
        if not, then manually uses RDKit subgraph matching against list of templates
    '''

    if rct1_smi in rct_to_temps:
        valid_rct1_temps, valid_rct2_temps = rct_to_temps[rct1_smi]

    else:
        # need to check manually
        rct1_mol = Chem.MolFromSmiles(rct1_smi)

        valid_rct1_temps, valid_rct2_temps = set(), set()
        for temp_idx, temp_str in enumerate(template_strs):
            valid_rct1, valid_rct2 = is_valid_reactant(rct1_mol, temp_str)

            if valid_rct1:
                valid_rct1_temps.add(temp_idx)

            if valid_rct2:
                valid_rct2_temps.add(temp_idx)

    # make the reaction mask
    mask_rxn = [0 for _ in range(len(template_strs))]
    for i in (valid_rct1_temps | valid_rct2_temps):
        mask_rxn[i] = 1

    return torch.Tensor(mask_rxn), valid_rct1_temps, valid_rct2_temps

def get_mask_reaction_merge(state, template_strs, just_checking=False):
    rct1_mol = Chem.MolFromSmiles(state[0])
    rct2_mol = Chem.MolFromSmiles(state[1])

    mask_rxn = [0 for _ in range(len(template_strs))]

    for i, template_str in enumerate(template_strs):
        rxn = AllChem.ReactionFromSmarts(template_str)
        rdChemReactions.ChemicalReaction.Initialize(rxn)

        if rxn.GetNumReactantTemplates() == 2:
            # check if a valid product molecule can be obtained
            prod_mol = rxn.RunReactants((rct1_mol, rct2_mol))
            if prod_mol:
                if just_checking:
                    # if we just want to check if merge is possible,
                    # we can return just True once the first match is found
                    # to reduce redundant calculation
                    return True
                else:
                    mask_rxn[i] = 1

    if just_checking:
        return False
    else:
        return torch.Tensor(mask_rxn)

def get_mask_action(state, template_strs):
    if len(state) == 0:
        mask = [1, 0, 0, 0]

    elif len(state) == 1:
        mask = [1, 1, 0, 1]

    elif len(state) == 2:
        if get_mask_reaction_merge(state, template_strs, just_checking=True):
            mask = [0, 1, 1, 0]
        else:
            mask = [0, 1, 0, 0]

    else:
        raise ValueError(f"invalid state with length: {len(state)}")

    return torch.Tensor(mask)

def gen_synth_tree(smis, template_strs,
                   temp_to_rcts, rct_to_temps,
                   t_max=10):

    tree = SynthesisTree()
    most_recent_mol_smi = None

    try:
        for t in range(t_max):
            state = tree.eval_state()

            # sample a random, but valid, action
            # 0 --> "ADD", 1 --> "EXPAND", 2 --> "MERGE", 3 --> "END"
            probs_action = torch.rand(4)
            mask_action = get_mask_action(state, template_strs)
            probs_action_masked = probs_action * mask_action
            action = int(torch.argmax(probs_action_masked).item())

            if action == 3: # END
                break

            elif action == 0: # ADD
                # for data generation, just randomly sample
                # for f_rt1 prediction, do a nearest-neighbour search
                rct1_smi = random.choice(smis)

            else: # EXPAND or MERGE
                rct1_smi = most_recent_mol_smi

            # sample a valid reaction template (that rct1_smi can undergo)
            # if using f_rxn, we can pass mask_rxn into model.forward()
            # to apply the mask before softmax
            probs_rxn = torch.rand(len(template_strs))

            if action == 2:
                # if merge, the reaction must fit both sub-tree root molecules
                # thus, reaction mask has to be specially calculated
                mask_rxn = get_mask_reaction_merge(state, template_strs)
            else: # if not merge, reaction just has to fit rct1_smi, we can sample rct2_smi later
                mask_rxn, rct1_temps, rct2_temps = get_mask_rxn_and_order(rct1_smi,
                                                                        template_strs,
                                                                        rct_to_temps)
            probs_rxn_masked = probs_rxn * mask_rxn
            rxn_idx = int(torch.argmax(probs_rxn_masked).item())
            rxn_str = template_strs[rxn_idx]

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

                    probs_rct2 = torch.rand(len(smis))
                    mask_rct2 = get_mask_rct2(rxn_idx, rct2_is_second,
                                            template_strs, temp_to_rcts, len(smis))
                    if sum(mask_rct2) == 0:
                        # no building block can match as reactant #2 of template
                        break

                    probs_rct2_masked = probs_rct2 * mask_rct2
                    rct2_idx = int(torch.argmax(probs_rct2_masked).item())
                    rct2_smi = smis[rct2_idx]

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
        # print(e)
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

def gen_one_tree(i, t_max=10):
    tree = gen_synth_tree(smis, template_strs,
               temp_to_rcts, rct_to_temps,
               t_max)
    return tree

if __name__ == "__main__":
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks_cleaned.csv")
    parser.add_argument("--path_templates", type=Path, default="data/templates_cleaned.txt")
    parser.add_argument("--path_rct_to_temps", type=Path, default="data/rct_to_temps_cleaned.pickle")
    parser.add_argument("--path_temp_to_rcts", type=Path, default="data/temp_to_rcts_cleaned.pickle")
    parser.add_argument("--path_trees", type=Path, default="data/trees.pickle")
    parser.add_argument("--num_trees", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=25000)
    parser.add_argument("--ncpu", type=int, default=24)
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

    p = Pool(args.ncpu)

    seed_everything(args.seed)

    NUM_TREES = args.num_trees
    trees = []
    gen_one_tree_ = partial(gen_one_tree, t_max=args.max_steps)

    cnt_success, cnt_fail = 0, 0
    for tree in tqdm(
            p.imap(gen_one_tree_, range(NUM_TREES), chunksize=1),
            total=NUM_TREES
        ):
        if tree:
            cnt_success += 1
            trees.append(tree)

            if cnt_success > 0 and cnt_success % args.checkpoint_every == 0:
                # checkpoint trees
                with open(args.path_trees, 'wb') as f:
                    pickle.dump(trees, f)

        else:
            cnt_fail += 1

    print(f"num success: {cnt_success}")
    print(f"num fail: {cnt_fail}")
    # test with 10000 trees
    # num success: 5709
    # num fail: 4291

    # save trees
    with open(args.path_trees, 'wb') as f:
        pickle.dump(trees, f)
