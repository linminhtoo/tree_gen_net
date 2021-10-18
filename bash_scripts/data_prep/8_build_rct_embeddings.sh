source ~/.bashrc
conda activate tree

# about 2.5 mins for 150k molecules, acceptable
python3 data_scripts/build_rct_embeddings.py
