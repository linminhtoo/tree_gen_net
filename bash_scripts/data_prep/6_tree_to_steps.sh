source ~/.bashrc
conda activate tree

# change this
tree_prefix="maxstep10_num100k"

declare -a phases=("train" "val" "test")

for phase in "${phases[@]}"; do
    echo "processing ${phase}"

    python3 data_scripts/tree_to_steps.py \
        --path_csv_matched_rcts "data/matched_building_blocks.csv" \
        --path_templates "data/templates_cleaned.txt" \
        --path_trees "data/split/${tree_prefix}_${phase}.pickle" \
        --path_steps "data/${phase}/steps_${tree_prefix}.npz" \
        --path_states "data/${phase}/states_${tree_prefix}.npz"
done