source ~/.bashrc
conda activate tree

# change these
# choose from "f_rt1", "f_rt2", "f_rxn", "f_act"
model="f_act"
# tree_prefix="maxstep10_num100k"
tree_prefix="maxstep15_num600k"
version="v1"

# f_act --> tested, working, trains (loss converges)
# f_rxn --> tested, working, trains (loss converges)
# f_rt1 --> tested, working, trains (loss converges) --> can try ReLU or sigmoid as final activation
# f_rt2 --> tested, working, trains (loss converges)

CUDA_VISIBLE_DEVICES=3 python train.py \
    --model ${model} \
    --path_states_train "data/train/states_${tree_prefix}_${model}.npz" \
    --path_states_val "data/val/states_${tree_prefix}_${model}.npz" \
    --path_steps_train "data/train/steps_${tree_prefix}_${model}.npz" \
    --path_steps_val "data/val/steps_${tree_prefix}_${model}.npz" \
    --path_checkpoint "checkpoints/${tree_prefix}_${model}_${version}" \
    --checkpoint_existok \
    --dropout 0.25 \
    --epochs 200 \
    --early_stop \
    --early_stop_patience 10  \
        2>&1 | tee "logs/${tree_prefix}_${model}_${version}.log"

# EXCERPT FROM PAPER
# The action and reaction networks reach > 99% and 85.8% validation accuracy, respectively.
# The first reactant network, frt1, with a validation accuracy of only 30.8% after k-NN retrieval.
# To compare, frt2 reaches a validation accuracy of 70.0% without masking out invalid actions.