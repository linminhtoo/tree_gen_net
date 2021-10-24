# setup (conda environment)
```
# init conda, such as using .bashrc
source ~/.bashrc

conda create -n tree python=3.6 pip
conda activate tree

# rdkit can only be installed via conda, not pip
conda install -c rdkit rdkit -y

/path/to/conda/env/tree/bin/pip install -r requirements.txt
```

# how to run
## data preparation
Please run all the scripts in `tree_gen_net/bash_scripts/` following the numerical order from 1 to 10. The necessary input data has been included. Rough timing benchmarks have also been included. Some routines run much faster on multiprocessing.

## training
Please run the 4 scripts in `tree_gen_net/train/`, each named after the model they train. Alternatively, you can just run `bash tree_gen_net/train/train_all.sh`.

## inference
For synthesis planning, please run `bash tree_gen_net/train/decode_trees.sh`, which decodes synthetic trees greedily (k=1) conditional on product SMILES in the test set trees. Afterwards, please run `bash tree_gen_net/train/calc_simil.sh` to calculate the pairwise Tanimoto similarity of the decoded SMILES versus those product SMILES in the test set trees.

For property optimization via genetic algorithm, please run `bash tree_gen_net/train/genetic_algorithm.sh`

Of course, the arguments in all of these scripts can be modified and experimented with. None of the code is hardcoded and should be quite flexible. Currently, I have used either the settings from the paper, or otherwise scaled down some hyperparameters just to get working prototype out as fast as possible, given I have only spent 2 weekends (aka roughly 3-4 man days) on this project.

# citation
all credits goes to the authors
```
Gao, Wenhao, Rocío Mercado, and Connor W. Coley. "Amortized Tree Generation for Bottom-up Synthesis Planning and Synthesizable Molecular Design." arXiv preprint arXiv:2110.06389 (2021).
```

# TODOs
- potential next steps
    - some research ideas
    - unit tests