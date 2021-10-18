source ~/.bashrc
conda activate tree

# run this first
python3 data_scripts/build_rct_to_temps.py \
    --path_templates "data/templates_cleaned.txt" \
    --path_rct_to_temps "data/rct_to_temps_cleaned.pickle"