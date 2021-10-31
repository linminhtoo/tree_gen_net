source ~/.bashrc
conda activate tree

expt_name="maxstep10_num100k"

python3 calc_similarity.py \
    --path_target_smis "data/split/${expt_name}_test.pickle" \
    --path_decoded_trees "data/${expt_name}_test_decoded_trees_v2.pickle" \
    --ncpu 8
# average tanimoto similarity (including failed decoding): 0.5224 (+-0.3012)
# exact match (including failed decoding): 19.63
# exact match (excluding failed decoding): 20.68
# average tanimoto similarity (excluding failed decoding): 0.5505 (+-0.2831)