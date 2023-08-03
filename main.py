import matplotlib.pyplot as plt
from gerrychain.random import random
from datetime import datetime


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
        "--experiment_dir",
        default=f"./output_{timestr}",
        help="where experiment results will be dumped",
    )
    parser.add_argument(
        "--custom_partition", help="custom partition", action="store_true"
    )
    parser.add_argument(
        "--save_freq",
        type=lambda x: int(float(x)),
        default=25000,
        help="frequency to save jsons and maps and centroids with",
    )
    parser.add_argument(
        "--sample_freq",
        type=lambda x: int(float(x)),
        nargs="+",
        default=[1],
        help="frequency to sample chain",
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
        "--burn",
        type=lambda x: int(float(x)),
        default=2000,
        help="number of burn maps in markov chain",
    )
    parser.add_argument("--timer_freq", type=lambda x: int(float(x)), default=5000)
    parser.add_argument("--centroid", type=str, default=None, help="path to centroid")

    args = parser.parse_args()

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
        partition_dict = "2011_PLA_1"
        df = gpd.read_file("./Data/VTD_FINAL.shp")
        elections = [
            Election("PRES16", {"D": "T16PRESD", "R": "T16PRESR"}),
            Election("SEN16", {"D": "T16SEND", "R": "T16SENR"}),
            Election("AG16", {"D": "T16ATGD", "R": "T16ATGR"}),
            Election("GOV14", {"D": "F2014GOVD", "R": "F2014GOVR"}),
        ]

        election_names = ["PRES16", "SEN16", "AG16", "GOV14"]
    elif election_state == state.State.NC:
        graph = Graph.from_json("./Data/NC_VTD.json")
        pop_col_name = "TOTPOP"
        geo_key = "index"
        partition_dict = "CD"
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
        partition_dict = "CD"
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

    target_pop = tot_pop / num_districts
    population_calculation_timer.Accumulate()

    if args.custom_partition:
        recursive_tree_part_timer = Timer("recursive_tree_part")
        partition_dict = recursive_tree_part(
            graph, range(1, num_districts + 1), target_pop, pop_col_name, 0.01, 1
        )
        recursive_tree_part_timer.Accumulate()

    # election updater

    parties = ["D", "R"]

    updaters = {"population": updaters.Tally(pop_col_name, alias="population")}

    election_updaters = {election.name: election for election in elections}
    updaters.update(election_updaters)

    initial_partition = Partition(graph, partition_dict, updaters=updaters)

    proposal = partial(
        recom,
        pop_col=pop_col_name,
        pop_target=target_pop,
        epsilon=0.02,
        node_repeats=1,
    )

    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]), 2 * len(initial_partition["cut_edges"])
    )

    pop_constraint = constraints.within_percent_of_ideal_population(
        initial_partition, 0.02
    )

    experiment_timer = Timer("experiment")
    chain = MarkovChain(
        proposal=proposal,
        constraints=[compactness_bound, pop_constraint],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=args.total_steps,
    )

    t = 1
    map_num = 1

    # we need to create the centroids
    centroids = [np.zeros((num_nodes, num_nodes)) for _ in range(len(args.sample_freq))]
    tot_counts = np.zeros(len(args.sample_freq), dtype=int)
    num_graphs = args.total_steps - args.burn
    sen16 = [np.zeros(num_graphs // sample_freq) for sample_freq in args.sample_freq]
    pres16 = [np.zeros(num_graphs // sample_freq) for sample_freq in args.sample_freq]
    ag16 = [np.zeros(num_graphs // sample_freq) for sample_freq in args.sample_freq]
    gov14 = [np.zeros(num_graphs // sample_freq) for sample_freq in args.sample_freq]
    election_data = {
        election_name: [
            np.zeros(num_graphs // sample_freq) for sample_freq in args.sample_freq
        ]
        for election_name in election_names
    }

    chain_and_centroid_timer = Timer("chain_and_centroid")

    # INITIAL PARTITION

    initial_d = defaultdict(dict)

    for district in initial_partition["population"]:
        initial_d[district]["population"] = initial_partition["population"][district]
        initial_d[district]["id"] = []

    for key in initial_partition.assignment:
        district = initial_partition.assignment[key]
        initial_d[district]["id"].append(key)

    for electionName in election_names:
        initial_d[electionName] = {}
        if initial_partition[electionName].votes("D") > initial_partition[
            electionName
        ].votes("R"):
            initial_d[electionName]["winner"] = "D"
        elif initial_partition[electionName].votes("D") < initial_partition[
            electionName
        ].votes("R"):
            initial_d[electionName]["winner"] = "R"
        else:
            initial_d[electionName]["winner"] = "TIE"

        initial_d[electionName]["D"] = {}
        initial_d[electionName]["R"] = {}

        for party in parties:
            initial_d[electionName][party]["seats"] = initial_partition[
                electionName
            ].seats(party)
            initial_d[electionName][party]["percent_wins"] = initial_partition[
                electionName
            ].percent(party)
            initial_d[electionName][party]["votes"] = int(
                sum(initial_partition[electionName].counts(party))
            )

    with open(f"{json_dir}/plot_{0}.json", "w") as f:
        json.dump(initial_d, f)

    df[f"plot_{0}"] = df[geo_key].map(dict(initial_partition.assignment))

    df.plot(column=f"plot_{0}", cmap="tab20")
    plt.savefig(f"{map_dir}/plot_{0}.png")
    plt.close()

    if args.centroid is None:
        for partition in chain:
            partition_construction_timer = Timer("partition_construction")

            current_matrix = np.zeros((num_nodes, num_nodes))
            d = defaultdict(dict)

            for district in partition["population"]:
                d[district]["population"] = partition["population"][district]
                d[district]["id"] = []

            for key in partition.assignment:
                district = partition.assignment[key]
                d[district]["id"].append(key)

            for electionName in election_names:
                d[electionName] = {}
                if partition[electionName].votes("D") > partition[electionName].votes(
                    "R"
                ):
                    d[electionName]["winner"] = "D"
                elif partition[electionName].votes("D") < partition[electionName].votes(
                    "R"
                ):
                    d[electionName]["winner"] = "R"
                else:
                    d[electionName]["winner"] = "TIE"

                d[electionName]["D"] = {}
                d[electionName]["R"] = {}

                for party in parties:
                    d[electionName][party]["seats"] = partition[electionName].seats(
                        party
                    )
                    d[electionName][party]["percent_wins"] = partition[
                        electionName
                    ].percent(party)
                    d[electionName][party]["votes"] = int(
                        sum(partition[electionName].counts(party))
                    )

            if (t % args.save_freq) == 0:
                with open(f"{json_dir}/plot_{t}.json", "w") as f:
                    json.dump(d, f)

                df[f"plot_{t}"] = df[geo_key].map(dict(partition.assignment))

                df.plot(column=f"plot_{t}", cmap="tab20")
                plt.savefig(f"{map_dir}/plot_{t}.png")
                plt.close()

            partition_construction_timer.Accumulate()
            if t > args.burn:
                post_burn_matrix_timer = Timer("post_burn_matrix")
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
                post_burn_matrix_timer.Accumulate()

                centroid_calculation_timer = Timer("centroid_calculation")
                for sample_idx in range(len(args.sample_freq)):
                    if ((t - args.burn) % args.sample_freq[sample_idx]) == 0:
                        sample_centroid_election_timer = Timer(
                            f"sample_centroid_election_{args.sample_freq[sample_idx]}"
                        )
                        tot_count = tot_counts[sample_idx]
                        centroids[sample_idx] *= (tot_count) / (tot_count + 1)
                        centroids[sample_idx] += current_matrix / (tot_count + 1)
                        for election_name in election_names:
                            election_data[election_name][sample_idx][
                                tot_count - 1
                            ] = int(current_partitions[election_name]["D"]["seats"])
                        tot_counts[sample_idx] += 1
                        sample_centroid_election_timer.Accumulate()

                centroid_calculation_timer.Accumulate()

                # intermediate centroids
                if ((t - args.burn) % args.save_freq) == 0:
                    centroid_saving_timer = Timer("centroid_saving")
                    for sample_idx, sample_freq in enumerate(args.sample_freq):
                        np.save(
                            f"{matrix_dir}/centroid_intermediate_{t-args.burn}_{sample_freq}.npy",
                            centroids[sample_idx],
                        )
                    centroid_saving_timer.Accumulate()

            if (t % args.timer_freq) == 0:
                print(f"At {t} graphs: ")
                chain_and_centroid_timer.PrintTimeElapsed()
                print(f"Total time elapsed: {experiment_timer.TimeElapsed()}")

            t += 1
        partition_save_timer = Timer("partition_save")
        partition.graph.to_json(f"{json_dir}/plot_{t-1}.json")
        partition_save_timer.Accumulate()

        final_centroid_save_timer = Timer("final_centroid_save")
        for sample_idx, sample_freq in enumerate(args.sample_freq):
            np.save(
                f"{matrix_dir}/centroid_final_{sample_freq}.npy", centroids[sample_idx]
            )
            for election_name in election_names:
                np.save(
                    f"{election_dir}/{election_name.lower()}_{sample_freq}.npy",
                    election_data[election_name][sample_idx],
                )
        final_centroid_save_timer.Accumulate()

    else:
        print(f"Using centroid located at {args.centroid}")
        centroids[0] = np.load(args.centroid)
        print("Successfully loaded centroid.")
    chain_and_centroid_timer.Accumulate()

    t = 1

    tot_counts = np.zeros(len(args.sample_freq), dtype=int)
    unweighted_distances = [
        np.zeros(num_graphs // sample_freq + 1) for sample_freq in args.sample_freq
    ]

    weighted_distances = [
        np.zeros(num_graphs // sample_freq + 1) for sample_freq in args.sample_freq
    ]

    chain_and_distance_timer = Timer("chain_and_distance")
    unweighted_medoid_map = None
    unweighted_medoid_JSON = None

    unweighted_medoid_idx = None
    weighted_medoid_idx = None

    weighted_medoid_map = None
    weighted_medoid_JSON = None

    unweighted_medoid_distance = None
    weighted_medoid_distance = None

    for partition in chain:
        second_partition_construction_timer = Timer("second_partition_construction")
        current_matrix = np.zeros((num_nodes, num_nodes))
        d = defaultdict(dict)

        for district in partition["population"]:
            d[district]["population"] = partition["population"][district]
            d[district]["id"] = []

        for key in partition.assignment:
            district = partition.assignment[key]
            d[district]["id"].append(key)

        second_partition_construction_timer.Accumulate()

        if t > args.burn or t == 1:
            second_post_matrix_burn_timer = Timer("second_post_matrix_burn")
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
            second_post_matrix_burn_timer.Accumulate()

            distance_computation_timer = Timer("distance_computation")

            if t == args.burn + 1:
                dist = unweighted_distance(current_matrix, centroids[sample_idx])
                weighted_dist = weighted_distance(
                    populations, current_matrix, centroids[sample_idx]
                )
                unweighted_medoid_distance = dist
                weighted_medoid_distance = weighted_dist
                df["unweighted_medoid"] = df[geo_key].map(dict(partition.assignment))
                df.plot(column="unweighted_medoid", cmap="tab20")
                plt.savefig(f"{map_dir}/unweighted_medoid.png")
                plt.close()
                with open(f"{json}/unweighted_medoid.json", "rb") as f:
                    json.dump(partition, f)
                np.save(f"{map_dir}/unweighted_medoid_matrix.npy", current_matrix)

                df["weighted_medoid"] = df[geo_key].map(dict(partition.assignment))
                df.plot(column="weighted_medoid", cmap="tab20")
                plt.savefig(f"{map_dir}/weighted_medoid.png")
                plt.close()
                with open(f"{json}/weighted_medoid.json", "rb") as f:
                    json.dump(partition, f)
                np.save(f"{map_dir}/weighted_medoid_matrix.npy", current_matrix)

            if t == 1:
                for sample_idx in range(len(args.sample_freq)):
                    sample_distance_timer = Timer(
                        f"sample_distance_{args.sample_freq[sample_idx]}"
                    )
                    dist = unweighted_distance(current_matrix, centroids[sample_idx])
                    weighted_dist = weighted_distance(
                        populations, current_matrix, centroids[sample_idx]
                    )
                    tot_count = tot_counts[sample_idx]
                    unweighted_distances[sample_idx][tot_count] = dist
                    weighted_distances[sample_idx][tot_count] = weighted_dist
                    tot_counts[sample_idx] += 1

                    sample_distance_timer.Accumulate()
            else:
                for sample_idx in range(len(args.sample_freq)):
                    if ((t - args.burn) % args.sample_freq[sample_idx]) == 0:
                        sample_distance_timer = Timer(f"sample_distance_{sample_idx}")
                        dist = unweighted_distance(
                            current_matrix, centroids[sample_idx]
                        )
                        weighted_dist = weighted_distance(
                            populations, current_matrix, centroids[sample_idx]
                        )
                        tot_count = tot_counts[sample_idx]
                        unweighted_distances[sample_idx][tot_count] = dist
                        weighted_distances[sample_idx][tot_count] = weighted_dist
                        tot_counts[sample_idx] += 1
                        sample_distance_timer.Accumulate()

                        if unweighted_medoid_distance > dist:
                            unweighted_medoid_idx = t
                            unweighted_medoid_distance = dist
                            df["unweighted_medoid"] = df[geo_key].map(
                                dict(partition.assignment)
                            )
                            df.plot(column="unweighted_medoid", cmap="tab20")
                            plt.savefig(f"{map_dir}/unweighted_medoid.png")
                            plt.close()
                            np.save(
                                f"{map_dir}/unweighted_medoid_matrix.npy",
                                current_matrix,
                            )
                            with open(f"{json}/unweighted_medoid.json", "rb") as f:
                                json.dump(partition, f)

                        if weighted_medoid_distance > dist:
                            weighted_medoid_idx = t
                            weighted_medoid_distance = dist
                            df["weighted_medoid"] = df[geo_key].map(
                                dict(partition.assignment)
                            )
                            df.plot(column="weighted_medoid", cmap="tab20")
                            plt.savefig(f"{map_dir}/weighted_medoid.png")
                            plt.close()
                            np.save(
                                f"{map_dir}/weighted_medoid_matrix.npy",
                                current_matrix,
                            )
                            with open(f"{json}/weighted_medoid.json", "rb") as f:
                                json.dump(partition, f)

            distance_computation_timer.Accumulate()

        if (t % args.timer_freq) == 0:
            print(f"At {t} graphs: ")
            chain_and_distance_timer.PrintTimeElapsed()
            print(f"Total time elapsed: {experiment_timer.TimeElapsed()}")

        t += 1

    chain_and_distance_timer.Accumulate()

    unweighted_distances_save_timer = Timer("unweighted_distances_save")
    for sample_idx, sample_freq in enumerate(args.sample_freq):
        np.save(
            f"{matrix_dir}/unweighted_distances_{sample_freq}.npy",
            unweighted_distances[sample_idx],
        )
    unweighted_distances_save_timer.Accumulate()

    weighted_distances_save_timer = Timer("weighted_distances_save")
    for sample_idx, sample_freq in enumerate(args.sample_freq):
        np.save(
            f"{matrix_dir}/weighted_distances_{sample_freq}.npy",
            weighted_distances[sample_idx],
        )
    weighted_distances_save_timer.Accumulate()

    experiment_timer.Accumulate()

    print("Experiment finished!")
    final_timer_str = Timer.PrintAccumulated()
    with open(f"{args.experiment_dir}/final_time.out", "w") as f:
        f.write(
            f"Unweighted medoid idx: {unweighted_medoid_idx} and weighted medoid idx: {weighted_medoid_idx}\n\n"
        )
        f.write(final_timer_str)
