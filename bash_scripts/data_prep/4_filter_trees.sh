source ~/.bashrc
conda activate tree

# num trees original: 349597
# num trees kept: 241470
python3 data_scripts/filter_trees.py \
    --seed 1337 \
    --path_trees "data/trees_maxstep15_num600k.pickle" \
    --path_trees_filtered "data/trees_maxstep15_num600k_filtered.pickle"

# num trees original: 56895
# num trees kept: 40227
# python3 data_scripts/filter_trees.py \
#     --seed 1337 \
#     --path_trees "data/trees_maxstep10_num100k.pickle" \
#     --path_trees_filtered "data/trees_maxstep10_num100k_filtered.pickle"

# 20 seconds for 5709 trees without multiprocess, speed is okay
# kept 4016 on seed 1337
# python3 data_scripts/filter_trees.py \
#     --seed 1337 \
#     --path_trees "data/trees.pickle" \
#     --path_trees_filtered "data/trees_filtered.pickle"