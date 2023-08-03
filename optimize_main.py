from dis import dis
from turtle import distance
import matplotlib.pyplot as plt
from gerrychain.random import random
from datetime import datetime

import pickle
from collections import defaultdict
from functools import partial
import os
import json
import geopandas as gpd
import time
import argparse
import state
import numpy as np
import sys
from timer import Timer
from distances import unweighted_distance, weighted_distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    timestr = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    parser.add_argument(
        "--explore", type=lambda x: int(float(x)), help="random walk length"
    )
    parser.add_argument(
        "--improvement_tolerance",
        type=lambda x: int(float(x)),
        help="improvement tolerance",
    )
    parser.add_argument(
        "--experiment_dir",
        default=f"./output_{timestr}",
        help="where experiment results will be dumped",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="seed for MCMC algorithm and random initial map",
        default=0,
    )
    parser.add_argument(
        "--state",
        type=lambda election_state: state.State[election_state],
        choices=list(state.State),
        default="PA",
    )
    parser.add_argument(
        "--total_steps",
        type=lambda x: int(float(x)),
        help="number of steps to run chain",
    )

    parser.add_argument(
        "--matrix_seed", type=str, default=None, help="path to matrix seed"
    )

    parser.add_argument("--centroid", type=str, default=None, help="path to centroid")

    parser.add_argument("--weighted_distance", action="store_true")

    args = parser.parse_args()

    assert (
        args.original_assignment is not None
    ), "Original assignment dictionary needs to be provided"
    assert args.centroid is not None, "Centroid needs to be provided"
    assert args.original_matrix is not None, "Original matrix needs to be provided"

    # these directories will be mostly empty. we will be saving very infrequently just for debugging purposes
    json_dir = f"{args.experiment_dir}/JSONs/"
    map_dir = f"{args.experiment_dir}/maps/"
    matrix_dir = f"{args.experiment_dir}/matrices/"  # the centroids will also be stored
    election_dir = f"{args.experiment_dir}/election/"

    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    os.makedirs(os.path.dirname(map_dir), exist_ok=True)
    os.makedirs(os.path.dirname(json_dir), exist_ok=True)
    os.makedirs(os.path.dirname(matrix_dir), exist_ok=True)
    os.makedirs(os.path.dirname(election_dir), exist_ok=True)

    arguments_called = " ".join(sys.argv[1:])

    with open(f"{args.experiment_dir}/args_called.out", "w") as f:
        f.write(arguments_called)

    election_state = args.state

    random.seed(args.seed)

    from gerrychain.tree import recursive_tree_part
    from gerrychain import (
        GeographicPartition,
        Partition,
        Graph,
        MarkovChain,
        proposals,
        updaters,
        constraints,
        accept,
        Election,
    )
    from gerrychain.proposals import recom

    if election_state == state.State.PA:
        graph = Graph.from_json("./Data/PA_VTDALL.json")
        pop_col_name = "TOT_POP"
        geo_key = "GEOID10"
        df = gpd.read_file("./Data/VTD_FINAL.shp")
        print(df.columns)
        # df = gpd.read_file("./Data/RemedialPlanShapefile.shp")
    elif election_state == state.State.NC:
        graph = Graph.from_json("./Data/NC_VTD.json")
        pop_col_name = "TOTPOP"
        geo_key = "index"
        df = gpd.read_file("./Data/NC_VTD.shp")
        df = df.reset_index()
        elections = [
            Election("PRES16", {"D": "EL16G_PR_D", "R": "EL16G_PR_R"}),
            Election("SEN16", {"D": "EL16G_US_1", "R": "EL16G_USS_"}),
            Election("GOV16", {"D": "EL16G_GV_D", "R": "EL16G_GV_R"}),
        ]
        election_names = ["PRES16", "SEN16", "GOV16"]

    elif election_state == state.State.MD:
        graph = Graph.from_json("./Data/MD-precincts_no_islands.json")
        pop_col_name = "ADJ_POP"
        geo_key = "index"
        df = gpd.read_file("./Data/MD-precincts.shp")
        df = df.reset_index()
        elections = [
            Election("PRES16", {"D": "PRES16D", "R": "PRES16R"}),
            Election("SEN16", {"D": "SEN16D", "R": "SEN16R"}),
            Election("GOV14", {"D": "GOV14D", "R": "GOV14R"}),
            Election("AG18", {"D": "AG18D", "R": "AG18R"}),
        ]
        election_names = ["PRES16", "SEN16", "GOV14", "AG18"]
    else:
        assert (False, "Not supported!")

    Timer.TimerClassReset()

    population_calculation_timer = Timer("population_calculation")
    populations = np.array([graph.nodes[n][pop_col_name] for n in graph.nodes()])
    np.save(f"{matrix_dir}/pop_weights.npy", populations)
    tot_pop = sum([graph.nodes[n][pop_col_name] for n in graph.nodes()])
    num_districts = election_state.num_districts
    if election_state == state.State.NC:
        districts = [str(x) for x in range(1, num_districts + 1)]
    else:
        districts = list(range(1, num_districts + 1))

    num_nodes = len(graph.nodes())

    # numbering the nodes
    indices = dict((node_id, index) for index, node_id in enumerate(graph.nodes()))
    reverse_indices = dict(
        (index, node_id) for index, node_id in enumerate(graph.nodes())
    )

    target_pop = tot_pop / num_districts
    population_calculation_timer.Accumulate()

    distance_function = unweighted_distance
    if args.weighted_distance:
        distance_function = partial(weighted_distance, populations)

    # election updater

    parties = ["D", "R"]
    updaters = {"population": updaters.Tally(pop_col_name, alias="population")}
    election_updaters = {election.name: election for election in elections}
    updaters.update(election_updaters)

    proposal = partial(
        recom,
        pop_col=pop_col_name,
        pop_target=target_pop,
        epsilon=0.02,
        node_repeats=1,
    )

    experiment_timer = Timer("experiment")

    t = 1
    map_num = 1

    print(f"Using centroid located at {args.centroid}")
    centroid = np.load(args.centroid)
    print("Successfully loaded centroid.")

    print(f"Using matrix loaded at {args.matrix_seed}")
    our_matrix = np.load(args.matrix_seed)
    print("Successfully loaded matrix seed")

    with open(args.original_assignment, "rb") as f:
        initial_partition = json.loads(f)

    # In case we want to use different updaters here than when we originally generated the partition
    Partition(graph, initial_partition.assignment, updaters=updaters)
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
    )
    pop_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, 0.02
    )

    our_matrix = 
    current_dist = distance_function(centroid, our_matrix)
    print(f"Original dist is {current_dist:.7E}")
    t = 1

    def opt_accept(state):
        if opt_accept.improve_counter >= args.improvement_tolerance:
            opt_accept.explore_counter += 1
            if opt_accept.explore_counter >= args.explore:
                opt_accept.improve_counter = 0
            return True
        else:
            partition = state
            current_matrix = np.zeros((num_nodes, num_nodes))
            d = defaultdict(dict)

            for district in partition["population"]:
                d[district]["population"] = partition["population"][district]
                d[district]["id"] = []

            for key in partition.assignment:
                district = partition.assignment[key]
                d[district]["id"].append(key)

            current_partitions = d
            for district in partition["population"]:
                current_district = current_partitions[district][
                    "id"
                ]  # array of all the nodes in the same district

                row = np.zeros(num_nodes)

                current_district_numeric = [
                    v for k, v in indices.items() if k in current_district
                ]
                row[current_district_numeric] = 1

                for node_numeric in current_district_numeric:
                    current_matrix[node_numeric] = row

            current_dist = distance_function(centroid, current_matrix)
            opt_accept.current_dist = current_dist

            if opt_accept.explore_counter != 0:
                opt_accept.local_best_distance = current_dist
                opt_accept.explore_counter = 0
                df["current_seed"] = df[geo_key].map(dict(partition.assignment))
                df.plot(column="current_seed", cmap="tab20")
                plt.savefig(f"{map_dir}/current_seed_map.png")
                plt.close()
                np.save(
                    f"{map_dir}/current_seed.npy",
                    current_matrix,
                )
                with open(
                    f"{json_dir}/current_seed_assignment_dictionary.pkl", "wb"
                ) as fh:
                    pickle.dump(partition.assignment, fh)

            if current_dist < opt_accept.local_best_distance:
                opt_accept.improve_counter = 0
                opt_accept.local_best_distance = current_dist
                df["current_seed"] = df[geo_key].map(dict(partition.assignment))
                df.plot(column="current_seed", cmap="tab20")
                plt.savefig(f"{map_dir}/current_seed_map.png")
                plt.close()
                np.save(
                    f"{map_dir}/current_seed.npy",
                    current_matrix,
                )
                with open(
                    f"{json_dir}/current_seed_assignment_dictionary.pkl", "wb"
                ) as fh:
                    pickle.dump(partition.assignment, fh)

                if opt_accept.local_best_distance < opt_accept.best_distance:
                    opt_accept.best_distance = opt_accept.local_best_distance
                    df["unweighted_medoid"] = df[geo_key].map(
                        dict(partition.assignment)
                    )
                    df.plot(column="unweighted_medoid", cmap="tab20")
                    plt.savefig(f"{map_dir}/closest_map.png")
                    plt.close()
                    np.save(
                        f"{map_dir}/closest_map.npy",
                        current_matrix,
                    )
                    with open(
                        f"{json_dir}/closest_map_assignment_dictionary.pkl", "wb"
                    ) as fh:
                        pickle.dump(partition.assignment, fh)
                return True
            else:
                opt_accept.improve_counter += 1
                return False

    opt_accept.best_distance = distance_function(centroid, our_matrix)
    opt_accept.local_best_distance = distance_function(centroid, our_matrix)
    opt_accept.current_dist = distance_function(centroid, our_matrix)
    opt_accept.explore_counter = 0
    opt_accept.improve_counter = 0

    chain = MarkovChain(
        proposal=proposal,
        constraints=[compactness_bound, pop_constraint],
        accept=opt_accept,
        initial_state=initial_partition,
        total_steps=args.total_steps,
    )

    local_best_distance = np.zeros(args.total_steps)
    best_distance = np.zeros(args.total_steps)
    current_distance = np.zeros(args.total_steps)
    for i, partition in enumerate(chain):
        chain_optimize_timer = Timer("optimization_chain")
        local_best_distance[i] = opt_accept.local_best_distance
        best_distance[i] = opt_accept.best_distance
        current_distance[i] = opt_accept.current_dist
        if i % 10 == 0:
            print(
                f"Iteration {i}: Current distance: {opt_accept.current_dist:.6E} | best distance: {opt_accept.best_distance:.6E} | local best distance: {opt_accept.local_best_distance:.6E} | explore: {opt_accept.explore_counter} | improve_counter: {opt_accept.improve_counter}"
            )

        if i % 100 == 0:
            np.save(f"{args.experiment_dir}/best_distance.npy", best_distance)
            np.save(
                f"{args.experiment_dir}/local_best_distance.npy", local_best_distance
            )
            np.save(f"{args.experiment_dir}/current_distance.npy", current_distance)

        chain_optimize_timer.Accumulate()
    experiment_timer.Accumulate()
    print("Experiment finished!")
    final_timer_str = Timer.PrintAccumulated()
    with open(f"{args.experiment_dir}/final_time.out", "w") as f:
        f.write(final_timer_str)
