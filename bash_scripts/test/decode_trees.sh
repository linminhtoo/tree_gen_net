source ~/.bashrc
conda activate tree

CUDA_VISIBLE_DEVICES=3 python3 decode_trees.py \
    --path_target_smis "data/split/maxstep10_num100k_test.pickle" \
    --path_model_config "config/models.yaml" \
    --path_save_decoded_trees "data/maxstep10_num100k_test_decoded_trees.pickle" \
    --max_steps 10 \
    --checkpoint_every 5000
    # --ncpu 8
# multiprocessing is too slow, about 2.2 sec/tree even on 8 processes
# not able to init models in initializer, CUDA OOM
# █████████████| 8005/8005 [4:49:19<00:00,  2.17s/it]
# num targets: 8005
# num success: 7598 (94.92%)
# num fail: 407 (5.08%)