source ~/.bashrc
conda activate tree

# run this after build_rct_to_temps.sh, about 30 mins on 32 cores for 10,000 trees, not fast...
# TODO: how to choose --max_steps
python3 data_scripts/build_random_trees.py \
    --path_templates "data/templates_cleaned.txt" \
    --path_rct_to_temps "data/rct_to_temps_cleaned.pickle" \
    --path_temp_to_rcts "data/temp_to_rcts_cleaned.pickle" \
    --path_csv_matched_rcts "data/matched_building_blocks_cleaned.csv" \
    --path_trees "data/trees_maxstep10_num100k.pickle" \
    --seed 1337 \
    --num_trees 100000 \
    --max_steps 10 \
    --ncpu 32

# with 10,000 attempts at max_steps=10, got 5709 success (57%)
# for their paper, seems they tried 600,000 attempts and got 347,740 success (58%)
# this ratio is similar, so our random tree generation algorithm should be working