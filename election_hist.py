import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import argparse

import state

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--custom_partition", help="custom partition", action="store_true"
    )
    parser.add_argument(
        "--state",
        type=lambda election_state: state.State[election_state],
        choices=list(state.State),
        default="PA",
    )
    parser.add_argument(
        "--experiment_dir",
        help="where experiment results will be pulled from",
    )
    parser.add_argument(
        "--num_bins", type=int, default=13, help="num of histogram bins"
    )
    args = parser.parse_args()

    election_dir = f"{args.experiment_dir}/election"

    election_state = args.state

    if election_state == state.State.PA:
        election_names = ["PRES16", "SEN16", "AG16", "GOV14"]
    elif election_state == state.State.NC:
        election_names = ["PRES16", "SEN16", "GOV16"]
    elif election_state == state.State.MD:
        election_names = ["PRES16", "SEN16", "GOV14", "AG18"]
    else:
        assert (False, "Not supported!")

    start = time.time()

    num_bins = np.array(list(range(args.num_bins))) - 0.5

    for election_name in election_names:
        loaded = np.load(f"{election_dir}/{election_name.lower()}_1.npy")
        plt.hist(loaded, bins=num_bins, color="skyblue", density=True, rwidth=0.75)
        plt.xticks(num_bins + 0.5)
        plt.xlabel("Number of Democratic seats")
        plt.ylabel("Frequency")
        plt.savefig(f"{election_dir}/{election_name.lower()}_hist.png")
        plt.close()
