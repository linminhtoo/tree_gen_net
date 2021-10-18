source ~/.bashrc
conda activate tree

# 20 seconds for 4016 trees without multiprocess, speed is okay
python3 data_scripts/tree_to_steps.py \
    --path_csv_matched_rcts "data/matched_building_blocks.csv" \
    --path_templates "data/templates_cleaned.txt" \
    --path_trees "data/trees_filtered.pickle" \
    --path_steps "data/steps.npz" \
    --path_states "data/states.npz" \
