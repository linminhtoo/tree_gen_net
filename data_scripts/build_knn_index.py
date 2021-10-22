import argparse
import os
import sys
import time
from pathlib import Path

import nmslib
from scipy import sparse

package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_dir)

def create_knn_index(
        data,
        metric="cosinesimil", method="hnsw"
    ):
    index = nmslib.init(method=method, space=metric)
    index.addDataPointBatch(data)
    index.createIndex(
        {
            "M": 30,
            "indexThreadQty": 8,
            "efConstruction": 100,
            "post": 0,
        }
    )
    return index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_fps", type=Path, default="data/rct_fps.npz")
    parser.add_argument("--path_index", type=Path, default="data/knn_rct_fps.index")
    args = parser.parse_args()

    # load fingerprint embeddings of all building blocks
    mol_fps = sparse.load_npz(args.path_fps)
    mol_fps_np = mol_fps.toarray()

    # build the approximate nearest neighbour search index using HNSW algorithm
    start = time.time()
    index = create_knn_index(mol_fps_np)
    end = time.time()
    print(f"Indexing time = {end-start:.3f}s")

    # save the index
    index.saveIndex(str(args.path_index), save_data=True)
