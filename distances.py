import numpy as np
import json
import os
import time
import argparse
import state

from scipy import sparse


def unweighted_distance(matrix, centroid):
    l2_dist = np.linalg.norm(centroid - matrix)

    return 0.5 * (l2_dist**2)


def weighted_distance(weights, matrix, centroid):

    diff = np.square(matrix - centroid)
    weights = np.outer(weights, weights)
    diff_weighted = diff * weights
    return 0.5 * (diff_weighted.sum())


start = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_partition", help="custom partition", action="store_true"
    )
    parser.add_argument("--json_dir", help="where the maps jsons will be loaded")
    parser.add_argument("--matrix_dir", help="where the matrices will be saved")
    parser.add_argument("--progress_dir", help="where the progress will be saved")
    parser.add_argument(
        "--burn", type=int, default=2000, help="number of burn maps in markov chain"
    )
    parser.add_argument(
        "--num_graphs", type=int, default=200000, help="number of total graphs"
    )
    parser.add_argument(
        "--state",
        type=lambda election_state: state.State[election_state],
        choices=list(state.State),
        default=state.State.PA,
    )

    args = parser.parse_args()

    progress_dir = args.progress_dir
    matrix_dir = args.matrix_dir
    json_dir = args.json_dir

    election_state = args.state
    num_districts = election_state.num_districts
    burn = args.burn
    num_graphs = args.num_graphs

    centroid = sparse.load_npz(f"{matrix_dir}/centroid.npz")

    weighted = False

    indices = {}

    districts = [str(x + 1) for x in range(num_districts)]

    num_nodes = 0

    weighted_distances = np.zeros(num_graphs - burn)
    unweighted_distances = np.zeros(num_graphs - burn)

    fdir = f"{json_dir}/plot1.json"
    with open(fdir) as a:
        dict1 = json.load(a)

    # numbering the districts
    for district in districts:
        for node in dict1[district]["id"]:
            if node not in indices:
                indices[node] = num_nodes
                num_nodes += 1

    def weighted_distance():
        pass

    def unweighted_distance(matrix, centroid):
        l2_dist = linalg.norm(centroid - matrix)

        return 0.5 * (l2_dist**2)

    for i in range(burn + 1, num_graphs + 1):
        compressed_save_path = f"{matrix_dir}/matrix_{i}.npz"

        current_graph = sparse.load_npz(compressed_save_path)

        if weighted:
            weighted_distance()
            assert (False, "Implement this method!")

        else:
            dist = unweighted_distance(current_graph, centroid)
            unweighted_distances[i - (burn + 1)] = dist

        if i % 1000 == 0:
            os.makedirs(
                os.path.dirname(f"{progress_dir}/distance{i}.txt"),
                exist_ok=True,
            )
            with open(f"{progress_dir}/distance{i}.txt", "w") as f:
                f.write("Time elapsed: " + str((time.time() - start) / 60) + " minutes")

    np.save(f"{matrix_dir}/unweighted.npy", unweighted_distances)
    print(
        "unweighted distances saved in " + str((time.time() - start) / 60) + " minutes"
    )
