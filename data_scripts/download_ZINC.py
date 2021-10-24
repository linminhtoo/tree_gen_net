import os
import pickle
import random
import sys
import requests
from pathlib import Path

from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import seed_everything

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ZINC_urls", type=Path, default="data/ZINC_urls.txt")
    parser.add_argument("--path_save_all", type=Path, default="data/ZINC_smi_all.pickle")
    parser.add_argument("--path_save_seeds", type=Path, default="data/ZINC_smi_seeds.txt")
    parser.add_argument("--save_all", action="store_true",
                        help="whether to save all ZINC SMILES, NOTE: needs 24 GB of disk space")
    parser.add_argument("--num_seeds", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=1337)
    args = parser.parse_args()

    seed_everything(args.random_seed)

    # get urls from ZINC website
    # http://zinc15.docking.org/tranches/home/
    with open(args.path_ZINC_urls, 'r') as f:
        urls = [l.strip() for l in f.readlines()]

    all_data = []
    for url in tqdm(urls):
        # download
        r = requests.get(url)

        # parse into list of str with each string containing "{smiles} {zinc_id}"
        data = [l.strip() for l in r.text.split('\n')][1:]
        all_data.extend(data)

    print(f"number of ZINC molecules: {len(all_data)}")

    # sample just the number we need for genetic algorithm
    idxs = random.sample(range(len(all_data)), args.num_seeds)
    seeds = []
    for idx in idxs:
        data = all_data[idx]
        # split "{smiles} {zinc_id}" --> smiles
        smi = data.split()[0]
        seeds.append(smi)

    # save seeds as text
    with open(args.path_save_seeds, 'w') as f:
        f.write('\n'.join(seeds))
    print('saved seed molecule SMILES as text')

    if args.save_all:
        print('saving all ZINC SMILES')
        with open(args.path_save_all, 'wb') as f:
            pickle.dump(all_data, f)
        print('saved all ZINC SMILES as pickle')

