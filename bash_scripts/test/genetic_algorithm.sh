source ~/.bashrc
conda activate tree

# expt_name="maxstep10_num100k_64p_256o_100g_01d_10s"
expt_name="maxstep10_num100k_128p_512o_100g_01d_10s_multi"

# NOTE: --path_seed_trees can only be provided after running at least once
CUDA_VISIBLE_DEVICES=0 python3 genetic_algorithm_multiprocess.py \
    --path_save_ckpt_dir "checkpoints/genetic_algorithm/${expt_name}/" \
    --path_seed_trees "checkpoints/genetic_algorithm/${expt_name}/seed_trees.pickle" \
    --property "GSK3B" \
    --generations 100 \
    --num_parents 128 \
    --num_offsprings 512 \
    --early_stop_delta 0.01 \
    --early_stop_patience 10 \
    --max_steps 10 \
    --save_every_gen \
    --random_seed 1337 \
    --ncpu 8 \
        2>&1 | tee "logs/GA_${expt_name}_v1.log"

# 8 processes --> ~1.1 it/s, 5 min/generation with 64 parents, 256 offsprings