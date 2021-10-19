source ~/.bashrc
conda activate tree

# total: 40227
# #################### initial random shuffle ####################
# num train: 24136
# num val: 8045
# num test: 8046
# no. of unique molecules at any part of train + val trees --> 93454
# no. of test trees with final product already seen in a train or val tree --> 41
# #################### after de-duplication ####################
# num train: 24136
# num val: 8086
# num test: 8005
python3 data_scripts/train_test_split.py \
    --path_trees "data/trees_maxstep10_num100k_filtered.pickle" \
    --path_train "data/split/maxstep10_num100k_train.pickle" \
    --path_val "data/split/maxstep10_num100k_val.pickle" \
    --path_test "data/split/maxstep10_num100k_test.pickle" \
    --seed 1337
