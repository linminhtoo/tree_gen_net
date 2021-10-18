import argparse
import pickle
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
from tqdm import tqdm

# fix to add package directory to path
# sys.path.append("/raid/notebooks/minhtoolin/repos/tree_gen_net")
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)

from data_scripts.utils import is_valid_reactant

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def check_one_mol(mol_smi):
    # template_strs is global shared object
    # set_1 = idxs of templates for which mol is reactant1 (canonical order)
    # set_2 = idxs of templates for which mol is reactant2 (canonical order)
    match_templates = [set(), set()]

    mol = Chem.MolFromSmiles(mol_smi)

    for i, template_str in enumerate(template_strs):
        valid_rt1, valid_rt2 = is_valid_reactant(mol, template_str)

        if valid_rt1:
            match_templates[0].add(i)

        if valid_rt2:
            match_templates[1].add(i)

    return mol_smi, match_templates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_build_block", type=Path, default="data/enamine_us.csv")
    parser.add_argument("--path_templates", type=Path, default="data/templates.txt")
    parser.add_argument("--path_rct_to_temps", type=Path, default="data/rct_to_temps.pickle")
    parser.add_argument("--ncpu", type=int, default=24)
    args = parser.parse_args()

    # load building blocks
    df_build_block = pd.read_csv(args.path_build_block)
    build_block_smis = df_build_block.SMILES.tolist()

    # load templates
    with open(args.path_templates, 'r') as f:
        template_strs = [l.strip().split('|')[1] for l in f.readlines()]

    p = Pool(args.ncpu)

    # takes about 3 mins on 24 cores for 150k molecules with 91 templates
    rct_to_temps = {}
    for rst in tqdm(
            p.imap(check_one_mol, build_block_smis),
            total=len(build_block_smis)
        ):
        mol_smi, match_templates = rst

        # only add to dict if the molecule matched at least once as either rct1 or rct2
        if len(match_templates[0]) > 0 or len(match_templates[1]) > 0:
            rct_to_temps[mol_smi] = match_templates

    print(f'original number of mols: {len(df_build_block)}')
    print(f'final number of mols: {len(rct_to_temps)}')

    # save reactant_to_template dictionary
    with open(args.path_rct_to_temps, 'wb') as f:
        pickle.dump(rct_to_temps, f)
