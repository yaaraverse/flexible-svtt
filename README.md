# Flexible SVTT

Flexible SVTT implements extensions of spatial variation in temporal trends (SVTT) using polynomial and spline models. This repository contains code and saved simulation datasets for evaluating cluster detection under different population settings and temporal patterns.

The implemented methods are **Linear, Quadratic, Poly3, Poly4, Spline3, and Spline4**.

## Repository contents

| Location | Purpose |
|---|---|
| `Code/svtt_pooled_ro.py` | Core scanner implementation and batch-processing functions |
| `Code/run_svtt_simulations.py` | Runs the saved simulation datasets and saves analysis summaries |
| `Data/` | Simulation datasets, county coordinates, and supporting input files |
| `JupyterNotebook/` | Notebooks used during development and analysis |
| `Results/` | Analysis outputs; created automatically when the runner executes |

## Requirements

Use a Python environment containing the dependencies imported by `Code/svtt_pooled_ro.py`.

The simulation runner additionally requires `pandas`. `IPython` is optional for formatted display of results; without it, the runner prints summaries to the terminal.

## Input data

Simulation files must contain these columns:

| Column | Description |
|---|---|
| `county` | County identifier |
| `year` | Observation year |
| `cases` | Observed case count |
| `pop` | County population for that year |

Each simulated dataset contains 33 counties observed over 1980–2000. Files are named `sim_001.csv` through `sim_100.csv`.

Save the simulation folders either directly inside `Data/` or inside `Data/Sim_Data/`. Use one layout consistently.

Single-cluster datasets follow this structure:

```text
Single_Cluster/<configuration>/<population_setting>/<residual_scenario>/sim_001.csv
```

Two-cluster datasets follow this structure:

```text
Two_Cluster/<scenario>/<population_setting>/sim_001.csv
```

Place `nm_county_coordinates.txt` inside `Data/`. A different coordinate-file location can be supplied using `--coord-path`.

## Simulation configurations

Two population settings are evaluated: `Constant` and `Original_Population`.

The single-cluster analysis includes three residual scenarios (`Res1`, `Res2`, and `Res3`) for each configuration:

| Configuration | Counties in the true cluster |
|---|---|
| `Single_County` | 15 |
| `Double_County` | 15, 31 |
| `Triple_County` | 3, 15, 31 |

These configurations each contain **one spatial cluster**, containing one, two, or three counties.

The two-cluster analysis evaluates `Scenario_1` and `Scenario_2`, with true clusters `{3, 15, 31}` and `{12, 21, 32}`.

There are 18 single-cluster batches and four two-cluster batches, with 100 datasets per batch.

## Running the analyses

Download or clone the repository and open a terminal in its root folder.

Run all single-cluster batches first, followed by all two-cluster batches:

```bash
python Code/run_svtt_simulations.py
```

Alternatively, run the two parts separately:

```bash
python Code/run_svtt_simulations.py --part single
python Code/run_svtt_simulations.py --part two
```

From a notebook located inside `JupyterNotebook/`, use:

```python
%run ../Code/run_svtt_simulations.py --part single
%run ../Code/run_svtt_simulations.py --part two
```

To specify another coordinate file, provide its path relative to the repository root:

```bash
python Code/run_svtt_simulations.py --coord-path Data/my_coordinates.txt
```

Paths are resolved from the script’s location, so the repository can be moved to another computer without changing personal directory paths.

## Default analysis settings

| Setting | Value |
|---|---|
| Datasets per batch | 100 |
| Monte Carlo replications | 100 |
| Significance level | 0.05 |
| Maximum window population share | 25% |
| Minimum window size | 1 county |
| Window step | 1 |
| Random seed | 218720 |
| Maximum selected clusters | 1 for single-cluster analyses; 2 for two-cluster analyses |
| Parallel jobs | 1 |
| Resume mode | Enabled |

The runner expects scanner implementation version `2026-10-01-fixed-R0-population-v1`.

## Saved outputs

Outputs are saved under:

```text
Results/Single_Cluster/SVTT_pooled_R0_q25_B100_<hash>/
Results/Two_Cluster/SVTT_pooled_R0_q25_B100_<hash>/
```

The hash is generated from the analysis settings to separate analyses using different settings.

Each configuration has its own output folder containing `batch_summary.csv` and the additional files saved by the scanner.

The single-cluster analysis saves:

```text
all_single_cluster_summaries.csv
```

The two-cluster analysis saves:

```text
all_two_cluster_summaries.csv
```

Both analysis folders also contain `analysis_settings.json`. Combined summaries are updated after each batch.

If a batch reports failed datasets, inspect its `failed_files.csv`, resolve the errors, and rerun the analysis with resume mode enabled.

## License

See `LICENSE` for the terms governing use and redistribution of this code.
