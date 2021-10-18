source ~/.bashrc
conda activate tree

# TODO: write loop for train, val & test
python3 data_scripts/steps_to_train_data.py \
    --path_templates "data/templates_cleaned.txt" \
    --path_steps "data/steps.npz" \
    --path_states "data/states.npz" \

# saved X f_act: data/states_f_act.npz --> length: 9595
# saved y f_act: data/steps_f_act.npz --> length: 9595
# saved X f_rt1: data/states_f_rt1.npz --> length: 4179
# saved y f_rt1: data/steps_f_rt1.npz --> length: 4179
# saved X f_rxn: data/states_f_rxn.npz --> length: 5579
# saved y f_rxn: data/steps_f_rxn.npz --> length: 5579
# saved X f_rt2: data/states_f_rt2.npz --> length: 5416
# saved y f_rt2: data/steps_f_rt2.npz --> length: 5416