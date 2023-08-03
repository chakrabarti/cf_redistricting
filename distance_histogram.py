import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

num_bins = 50


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    timestr = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    parser.add_argument(
        "--experiment_dir",
        default=f"./output_{timestr}",
        help="where experiment results will be dumped",
    )
    parser.add_argument(
        "--custom_partition", help="custom partition", action="store_true"
    )

    parser.add_argument(
        "--sample_freq",
        type=lambda x: int(float(x)),
        nargs="+",
        default=[1],
        help="frequency to sample chain",
    )

    args = parser.parse_args()

    for freq in args.sample_freq:
        unweighted_distances = np.load(
            f"{args.experiment_dir}/matrices/unweighted_distances_{freq}.npy"
        )
        weighted_distances = np.load(
            f"{args.experiment_dir}/matrices/weighted_distances_{freq}.npy"
        )
        unweighted_distances = unweighted_distances[1:]
        weighted_distances = weighted_distances[1:]

        print(
            f"Minimum distance to centroid for {args.experiment_dir}: {np.min(unweighted_distances):.6E}"
        )
        print(
            f"Minimum weighted distance to centroid for {args.experiment_dir}: {np.min(weighted_distances):.6E}"
        )
        normalized_unweighted_distances = unweighted_distances / np.max(
            unweighted_distances
        )
        normalized_weighted_distances = weighted_distances / np.max(weighted_distances)

        plt.hist(unweighted_distances, bins=num_bins, color="skyblue")

        plt.xlabel("Unweighted distance")
        plt.ylabel("Frequency")
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.savefig(
            f"{args.experiment_dir}/unweighted_histogram_{freq}.png", format="png"
        )
        plt.show()
        plt.close()

        plt.hist(weighted_distances, bins=num_bins, color="skyblue")
        plt.xlabel("Population-weighted distance")
        plt.ylabel("Frequency")
        plt.savefig(
            f"{args.experiment_dir}/weighted_histogram_{freq}.png", format="png"
        )
        plt.show()
        plt.close()
