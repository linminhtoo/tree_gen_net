source ~/.bashrc
conda activate tree

expt_name="maxstep10_num100k"

python3 calc_similarity.py \
    --path_target_smis "data/split/${expt_name}_test.pickle" \
    --path_decoded_trees "data/${expt_name}_test_decoded_trees.pickle" \
    --ncpu 4