source ~/.bashrc
conda activate tree

suffix="maxstep15_num600k"
# total: 241470
# #################### initial random shuffle ####################
# num train: 144882
# num val: 48294
# num test: 48294
# no. of unique molecules at any part of train + val trees --> 389545
# no. of test trees with final product already seen in a train or val tree --> 614
# #################### after de-duplication ####################
# num train: 144882
# num val: 48908
# num test: 47680
python3 data_scripts/train_test_split.py \
    --path_trees "data/trees_${suffix}_filtered.pickle" \
    --path_train "data/split/${suffix}_train.pickle" \
    --path_val "data/split/${suffix}_val.pickle" \
    --path_test "data/split/${suffix}_test.pickle" \
    --seed 1337

# suffix="maxstep10_num100k"
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