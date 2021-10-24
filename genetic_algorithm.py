import argparse
import os
import pickle
import random
import sys
from collections import deque
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import nmslib
import numpy as np
import pandas as pd
import yaml
from rdkit import RDLogger
from scipy import sparse
from tdc import Oracle
from tqdm import tqdm

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)
from data_scripts.utils import seed_everything, smi_to_bit_fp
from decode_trees import decode_synth_tree, load_models

# silence annoying RDKit logging
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def crossover(parents, num_offsprings=512, mean=2048, std=410):
    num_parents, fp_size = parents.shape

    # for each offspring, randomly sample no. of bits to inherit from parent_a vs parent_b from N(2048, 410)
    num_bits_a = np.random.normal(mean, std, num_offsprings).astype(int)

    # for each offspring, randomly sample which bits come from parent_a vs parent_b
    bit_idxs = np.array([np.random.permutation(fp_size) for _ in range(num_offsprings)])

    # for each offspring, randomly sample which 2 parents to inherit from
    parent_idxs = np.random.choice(num_parents, size=num_offsprings * 2, replace=True)
    parent_idxs = parent_idxs.reshape(num_offsprings, 2)

    offsprings = []
    for i in range(num_offsprings):
        # initialise offspring vector of zeros
        offspring = np.zeros(fp_size, dtype='int32')

        # indices to inherit from parent_a vs parent_b
        bit_idxs_from_a = bit_idxs[i, :num_bits_a[i]]
        bit_idxs_from_b = bit_idxs[i, num_bits_a[i]:]

        # inherit accordingly
        offspring[bit_idxs_from_a] = parents[parent_idxs[i, 0], bit_idxs_from_a]
        offspring[bit_idxs_from_b] = parents[parent_idxs[i, 1], bit_idxs_from_b]

        offsprings.append(offspring)

    return np.array(offsprings)

def mutate(offsprings, num_bits=24, p=0.5):
    num_offsprings, fp_size = offsprings.shape

    offsprings = []
    for i in range(num_offsprings):
        offspring = offsprings[i]

        if random.random() < p:
            # do mutate
            # sample bit idxs to mutate
            mutate_idxs = np.random.choice(fp_size, size=num_bits, replace=False)
            # # flip the chosen bits from 0 to 1 and 1 to 0
            offspring[mutate_idxs] = 1 - offspring[mutate_idxs]

        else:
            # do not mutate
            offsprings.append(offspring)

    return np.array(offsprings)

def get_oracle(property):
    # we can also define custom oracles with complex combinations of different scores
    # they just need to have __call__(self, smi) method

    if property == "GSK3B":
        # Glycogen synthase kinase 3 beta, also known as GSK3β, is an enzyme that in humans is encoded by the GSK3β gene
        # Abnormal regulation and expression of GSK3β is associated with an increased susceptibility towards bipolar disorder
        # The oracle is a random forest classifer using ECFP6 fingerprints using ExCAPE-DB dataset.
        oracle = Oracle(name = "GSK3B")

    elif property == "JNK3":
        # DRD2 stands for dopamine type 2 receptor
        # The oracle is constructed by Olivercrona et al., using a support vector machine classifier
        # with a Gaussian kernel with ECFP6 fingerprint on ExCAPE-DB dataset.
        oracle = Oracle(name = "JNK3")

    elif property == "SA":
        # how hard or how easy it is to synthesize a given molecule,
        # based on a combination of the molecule’s fragments contributions.
        # The oracle is caluated via RDKit, using a set of chemical rules defined by Ertl et al.
        oracle = Oracle(name = "SA")

    else:
        raise ValueError(f"unrecognized property to optimize: {property}")

    return oracle

def optimize(args):
    seed_everything(args.random_seed)

    ########### LOAD ALL INPUTS ###########
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

    with open(args.path_model_config, "r") as stream:
        model_config = yaml.safe_load(stream)

    # load 4 trained models from checkpoints
    f_act, f_rt1, f_rt2, f_rxn = load_models(model_config)
    print(f"finished loading 4 models from checkpoints")


    ########### PREPARE THE SEEDS ###########
    # load starting seed SMILES
    with open(args.path_seeds, 'r') as f:
        seed_smis = [l.strip() for l in f.readlines()]
    assert len(seed_smis) == args.num_parents, \
        f"number of seed SMILES: {len(seed_smis)} not equal to --num_parents {args.num_parents}"

    # get oracle function
    oracle = get_oracle(args.property)

    p = Pool(args.ncpu)

    # score seed SMILES against desired property
    seed_scores = []
    for s in tqdm(
            p.imap(oracle, seed_smis),
            total=len(seed_smis), desc="scoring seed SMILES"
        ):
        seed_scores.append(s)

    # decode with seed SMILES as targets
    if args.path_seed_trees is None:
        print('decoding with seed SMILES as target_smi')
        seed_trees = []
        cnt_success, cnt_fail = 0, 0
        for seed_smi in tqdm(seed_smis):
            tree = decode_synth_tree(
                    f_act=f_act, f_rt1=f_rt1, f_rt2=f_rt2, f_rxn=f_rxn,
                    target_smi=seed_smi, target_z=None,
                    mol_fps=mol_fps, smis=smis, index_all_mols=index_all_mols,
                    template_strs=template_strs, temp_to_rcts=temp_to_rcts, rct_to_temps=rct_to_temps,
                    input_dim=model_config['input_fp_dim'], radius=model_config['radius'],
                    t_max=args.max_steps
                )
            if tree:
                cnt_success += 1
                seed_trees.append(tree)
            else:
                cnt_fail += 1
        print(f"num targets: {len(seed_smis)}")
        print(f"num success: {cnt_success} ({cnt_success / len(seed_smis) * 100:.2f}%)")
        print(f"num fail: {cnt_fail} ({cnt_fail / len(seed_smis) * 100:.2f}%)")

        with open(args.path_save_ckpt_dir / "seed_trees.pickle", 'wb') as f:
            pickle.dump(seed_trees, f)
    else:
        print('loading trees decoded from seed SMILES as target_smi')
        with open(args.path_seed_trees, 'rb') as f:
            seed_trees = pickle.load(f)

    # score decoded SMILES against desired property
    seed_decoded_smis = [tree.molecules[-1].smi for tree in seed_trees]
    seed_decoded_scores = []
    for score in tqdm(
            p.imap(oracle, seed_decoded_smis),
            total=len(seed_decoded_smis), desc="scoring SMILES decoded from seeds"
        ):
        seed_decoded_scores.append(score)

    print(f"average score of seed SMILES: {sum(seed_scores)/len(seed_scores):.4f}")
    print(f"average score of SMILES decoded from seeds: {sum(seed_decoded_scores)/len(seed_decoded_scores):.4f}")

    ########### RUN THE GENETIC ALGORITHM ###########
    # encode seeds into fingerprints
    seed_fps = [smi_to_bit_fp(smi, radius=args.radius, fp_size=args.fp_size) for smi in seed_smis]
    parents = seed_fps

    # crossover --> mutate --> score, until stopping criteria
    score_history = deque(maxlen=args.early_stop_patience)
    for gen_idx in tqdm(range(args.generations), desc="running genetic algorithm"):
        print('#'*50)
        print(f'generation {gen_idx}')

        offsprings = crossover(
            parents, num_offsprings=args.num_offsprings,
            mean=args.cross_mean, std=args.cross_std
        )

        offsprings = mutate(offsprings, num_bits=args.mutate_bits, p=args.mutate_prob)

        # run the decoding on single process
        decoded_trees = []
        cnt_success, cnt_fail = 0, 0
        for offspring in tqdm(offsprings, desc="decoding offsprings"):
            tree = decode_synth_tree(
                    f_act=f_act, f_rt1=f_rt1, f_rt2=f_rt2, f_rxn=f_rxn,
                    target_smi=None, target_z=offspring,
                    mol_fps=mol_fps, smis=smis, index_all_mols=index_all_mols,
                    template_strs=template_strs, temp_to_rcts=temp_to_rcts, rct_to_temps=rct_to_temps,
                    input_dim=model_config['input_fp_dim'], radius=model_config['radius'],
                    t_max=args.max_steps
                )
            if tree:
                cnt_success += 1
                decoded_trees.append(tree)
            else:
                cnt_fail += 1
        print(f"num targets: {len(offsprings)}")
        print(f"num success: {cnt_success} ({cnt_success / len(offsprings) * 100:.2f}%)")
        print(f"num fail: {cnt_fail} ({cnt_fail / len(offsprings) * 100:.2f}%)")

        # score decoded SMILES against desired property
        decoded_smis = [tree.molecules[-1].smi for tree in decoded_trees]
        decoded_scores = []
        for score in tqdm(
                p.imap(oracle, decoded_smis),
                total=len(decoded_smis), desc="scoring decoded SMILES"
            ):
            decoded_scores.append(score)
        mean_decoded_score = sum(decoded_scores)/len(decoded_scores)
        print(f"average score of all decoded SMILES: {mean_decoded_score:.4f}")

        # select the best molecules (SMILES) as parents for next generation
        decoded_scores = np.array(decoded_scores)
        idxs_desc = np.argsort(decoded_scores)[::-1]

        decoded_scores = decoded_scores[idxs_desc]
        decoded_smis = np.array(decoded_smis)[idxs_desc]
        decoded_trees = np.array(decoded_trees)[idxs_desc]

        parents = [smi_to_bit_fp(smi, radius=args.radius, fp_size=args.fp_size) \
                    for smi in decoded_smis[:args.num_parents]]
        print(f"average score of top-{args.num_parents} decoded SMILES: {decoded_scores[:args.num_parents].mean():.4f}")
        print(f"average score of top-1 decoded SMILES: {decoded_scores[0]:.4f}")
        print(f"average score of top-10 decoded SMILES: {decoded_scores[:10].mean():.4f}")

        if args.save_every_gen:
            # checkpoint
            with open(args.path_save_ckpt_dir / f"trees_{gen_idx}.pickle", 'wb') as f:
                pickle.dump(decoded_trees, f)
            with open(args.path_save_ckpt_dir / f"smis_{gen_idx}.txt", 'w') as f:
                f.write('\n'.join(decoded_smis))
            with open(args.path_save_ckpt_dir / f"scores_{gen_idx}.txt", 'w') as f:
                f.write('\n'.join(decoded_scores))
            sparse.save_npz(str(args.save_ckpt_dir / f"fps_{gen_idx}.npz"), offsprings)

        # decide whether to early stop
        score_history.appendleft(mean_decoded_score)
        if len(score_history) == args.early_stop_patience: # num elapsed generations >= early_stop_patience
            if score_history[-1] - score_history[0] < args.early_stop_delta:
                print(f"early stopping because rise in scores: {score_history[-1] - score_history[0]:.4f}")
                print(f"less than --early_stop_delta: {args.early_stop_delta:.4f}")
                break

    return offsprings, decoded_trees, decoded_smis, decoded_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # inputs
    parser.add_argument("--path_csv_matched_rcts", type=Path, default="data/matched_building_blocks.csv")
    parser.add_argument("--path_templates", type=Path, default="data/templates_cleaned.txt")
    parser.add_argument("--path_rct_to_temps", type=Path, default="data/rct_to_temps_cleaned.pickle")
    parser.add_argument("--path_temp_to_rcts", type=Path, default="data/temp_to_rcts_cleaned.pickle")
    parser.add_argument("--path_fps", type=Path, default="data/rct_fps.npz")
    parser.add_argument("--path_index", type=Path, default="data/knn_rct_fps.index")
    parser.add_argument("--path_model_config", type=Path, default="config/models.yaml")
    parser.add_argument("--path_seed_smis", type=Path, default="data/ZINC_smi_seeds.txt")
    parser.add_argument("--path_seed_trees", type=Path) # default="data/checkpoints/genetic_algorithm/seed_trees.pickle"
    # parser.add_argument("--path_seed_scores", type=Path) # default="data/checkpoints/genetic_algorithm/seed_scores.txt"
    # parser.add_argument("--path_seed_decoded_scores", type=Path) # default="data/checkpoints/genetic_algorithm/seed_decoded_scores.txt"
    # outputs
    parser.add_argument("--path_save_ckpt_dir", type=Path, default="data/checkpoints/genetic_algorithm/")
    parser.add_argument("--path_save_final_fps", type=Path, default="data/final_fps.npz")
    parser.add_argument("--path_save_final_trees", type=Path, default="data/final_trees.pickle")
    parser.add_argument("--path_save_final_smis", type=Path, default="data/final_smis.txt")
    parser.add_argument("--path_save_final_scores", type=Path, default="data/final_scores.txt")
    # genetic algorithm parameters
    parser.add_argument("--property", type=str, default="GSK3B",
                    help="oracle function to score generated molecules, ['GSK3B', 'JNK3', 'SA']")
    parser.add_argument("--num_offsprings", type=int, default=512)
    parser.add_argument("--num_parents", type=int, default=128)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--early_stop_delta", type=float, default=0.01)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--cross_mean", type=int, default=2048)
    parser.add_argument("--cross_std", type=int, default=410)
    parser.add_argument("--mutate_bits", type=int, default=24)
    parser.add_argument("--mutate_prob", type=float, default=0.5)
    parser.add_argument("--save_every_gen", aciton="store_true")
    # decoding parameters
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp_size", type=int, default=4096)
    parser.add_argument("--max_steps", type=int, default=10)
    # misc args
    parser.add_argument("--ncpu", type=int, default=24)
    args = parser.parse_args()

    (args.path_save_ckpt_dir).mkdir(parents=True, exist_ok=True)

    fps, trees, smis, scores = optimize(args)

    # save outputs from final generation for metric evaluation
    with open(args.path_save_final_trees, 'wb') as f:
        pickle.dump(trees, f)
    with open(args.path_save_final_smis, 'w') as f:
        f.write('\n'.join(smis))
    with open(args.path_save_final_scores, 'w') as f:
        f.write('\n'.join(scores))
    sparse.save_npz(str(args.save_final_fps), fps)
