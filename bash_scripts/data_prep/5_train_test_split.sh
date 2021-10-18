source ~/.bashrc
conda activate tree

# 20 seconds for 4016 trees without multiprocess, speed is okay
python3 data_scripts/train_test_split.py \
    --path_trees "data/trees_filtered.pickle" \
    --seed 1337
