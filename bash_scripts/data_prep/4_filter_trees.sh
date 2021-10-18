source ~/.bashrc
conda activate tree

# 20 seconds for 5709 trees without multiprocess, speed is okay
# kept 4016 on seed 1337
python3 data_scripts/filter_trees.py \
    --seed 1337 \
    --path_trees "data/trees.pickle" \
    --path_trees_filtered "data/trees_filtered.pickle"