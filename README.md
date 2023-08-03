* Dependencies are in `requirements.txt`

* Use `main.py` to run chain

* Use `distance_histogram.py` to plot distance histogram (edit file for hardcoded points for specific map distances) in the original experiment directory

* Use `election_hist.py` for election histograms which will appear in the original experiment directory

* Use `optimize_main.py` to run optimization heuristic

* Example: `python3 main.py --state NC --total_steps 100 --burn 10 --timer_freq 10 --experiment_dir NC_100/`

* Example: `python3 distance_histogram --experiment_dir NC_100/`

* Example `python3 election_hist.py --experiment_dir NC_100/` 

* Example `python3 optimize_main.py --experiment_dir NC_100_medoid_optimization --state NC --centroid NC_100_medoid_optimization/matrices/centroid_final_1.npy --matrix_seed NC_100_medoid_optimization/maps/unweighted_medoid_matrix.npy`

