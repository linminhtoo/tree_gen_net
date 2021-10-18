import argparse
import pickle
from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from tqdm import tqdm

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_templates", type=Path, default="data/templates.txt")
    parser.add_argument("--path_rct_to_temps", type=Path, default="data/rct_to_temps.pickle")
    parser.add_argument("--path_temp_to_rcts", type=Path, default="data/temp_to_rcts.pickle")
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv")
    args = parser.parse_args()

    # load matched building blocks
    with open(args.path_rct_to_temps, 'rb') as f:
        rct_to_temps = pickle.load(f)
    matched_mol_smis = list(rct_to_temps.keys())

    # load templates
    with open(args.path_templates, 'r') as f:
        template_strs = [l.strip().split('|')[1] for l in f.readlines()]

    # takes ~10 seconds
    temp_to_rcts = {}
    for template_idx, template_str in tqdm(
            enumerate(template_strs),
            total=len(template_strs)
        ):
        # set_1 = molecules which can be used as reactant1 (canonical order)
        # set_2 = molecules which can be used as reactant2 (canonical order)
        match_rcts = [set(), set()]

        for i, mol_smi in enumerate(matched_mol_smis):
            matched_templates = rct_to_temps[mol_smi]
            match_rct1_idxs, match_rct2_idxs = matched_templates

            if template_idx in match_rct1_idxs:
                match_rcts[0].add(i)
            if template_idx in match_rct2_idxs:
                match_rcts[1].add(i)

        temp_to_rcts[template_str] = match_rcts

    # save template_to_reactant dictionary
    with open(args.path_temp_to_rcts, 'wb') as f:
        pickle.dump(temp_to_rcts, f)

    # save CSV of SMILES strings of matched building block
    df_matched = pd.DataFrame({'SMILES': matched_mol_smis})
    df_matched.to_csv(args.path_csv_matched_rcts, index=False)
    print(f'number of matched molecules: {len(df_matched)}')
