source ~/.bashrc
conda activate tree

# change this
# tree_prefix="maxstep10_num100k"
tree_prefix="maxstep15_num600k"

declare -a phases=("train" "val" "test")

for phase in "${phases[@]}"; do
    echo "processing ${phase}"

    python3 data_scripts/steps_to_train_data.py \
        --path_templates "data/templates_cleaned.txt" \
        --path_steps "data/${phase}/steps_${tree_prefix}.npz" \
        --path_states "data/${phase}/states_${tree_prefix}.npz"
done

# with tree_prefix="maxstep10_num100k"
# processing train
# saved X f_act: data/train/states_maxstep10_num100k_f_act.npz --> length: 56979
# saved y f_act: data/train/steps_maxstep10_num100k_f_act.npz --> length: 56979
# saved X f_rt1: data/train/states_maxstep10_num100k_f_rt1.npz --> length: 25007
# saved y f_rt1: data/train/steps_maxstep10_num100k_f_rt1.npz --> length: 25007
# saved X f_rxn: data/train/states_maxstep10_num100k_f_rxn.npz --> length: 32843
# saved y f_rxn: data/train/steps_maxstep10_num100k_f_rxn.npz --> length: 32843
# saved X f_rt2: data/train/states_maxstep10_num100k_f_rt2.npz --> length: 31972
# saved y f_rt2: data/train/steps_maxstep10_num100k_f_rt2.npz --> length: 31972
# processing val
# saved X f_act: data/val/states_maxstep10_num100k_f_act.npz --> length: 18916
# saved y f_act: data/val/steps_maxstep10_num100k_f_act.npz --> length: 18916
# saved X f_rt1: data/val/states_maxstep10_num100k_f_rt1.npz --> length: 8337
# saved y f_rt1: data/val/steps_maxstep10_num100k_f_rt1.npz --> length: 8337
# saved X f_rxn: data/val/states_maxstep10_num100k_f_rxn.npz --> length: 10830
# saved y f_rxn: data/val/steps_maxstep10_num100k_f_rxn.npz --> length: 10830
# saved X f_rt2: data/val/states_maxstep10_num100k_f_rt2.npz --> length: 10579
# saved y f_rt2: data/val/steps_maxstep10_num100k_f_rt2.npz --> length: 10579
# processing test
# saved X f_act: data/test/states_maxstep10_num100k_f_act.npz --> length: 19039
# saved y f_act: data/test/steps_maxstep10_num100k_f_act.npz --> length: 19039
# saved X f_rt1: data/test/states_maxstep10_num100k_f_rt1.npz --> length: 8302
# saved y f_rt1: data/test/steps_maxstep10_num100k_f_rt1.npz --> length: 8302
# saved X f_rxn: data/test/states_maxstep10_num100k_f_rxn.npz --> length: 11034
# saved y f_rxn: data/test/steps_maxstep10_num100k_f_rxn.npz --> length: 11034
# saved X f_rt2: data/test/states_maxstep10_num100k_f_rt2.npz --> length: 10737
# saved y f_rt2: data/test/steps_maxstep10_num100k_f_rt2.npz --> length: 10737

# original debugging
# python3 data_scripts/steps_to_train_data.py \
#     --path_templates "data/templates_cleaned.txt" \
#     --path_steps "data/steps.npz" \
#     --path_states "data/states.npz" \

# saved X f_act: data/states_f_act.npz --> length: 9595
# saved y f_act: data/steps_f_act.npz --> length: 9595
# saved X f_rt1: data/states_f_rt1.npz --> length: 4179
# saved y f_rt1: data/steps_f_rt1.npz --> length: 4179
# saved X f_rxn: data/states_f_rxn.npz --> length: 5579
# saved y f_rxn: data/steps_f_rxn.npz --> length: 5579
# saved X f_rt2: data/states_f_rt2.npz --> length: 5416
# saved y f_rt2: data/steps_f_rt2.npz --> length: 5416