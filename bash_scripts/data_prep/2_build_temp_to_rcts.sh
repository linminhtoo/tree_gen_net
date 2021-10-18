source ~/.bashrc
conda activate tree

# run this after build_rct_to_temps.sh
python3 data_scripts/build_temp_to_rcts.py \
    --path_templates "data/templates_cleaned.txt" \
    --path_rct_to_temps "data/rct_to_temps_cleaned.pickle" \
    --path_temp_to_rcts "data/temp_to_rcts_cleaned.pickle" \
    --path_csv_matched_rcts "data/matched_building_blocks_cleaned.csv"