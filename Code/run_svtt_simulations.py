"""Run saved SVTT simulations with portable repository paths.

Save as:
    Code/run_svtt_simulations.py

Run both parts, in order:
    python Code/run_svtt_simulations.py

Run separately:
    python Code/run_svtt_simulations.py --part single
    python Code/run_svtt_simulations.py --part two

From a notebook inside JupyterNotebook/:
    %run ../Code/run_svtt_simulations.py --part single
    %run ../Code/run_svtt_simulations.py --part two
"""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import pandas as pd

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value.to_string(index=False))


# ------------------------------------------------------------
# Analysis settings
# ------------------------------------------------------------

N_FILES = 100
BMC = 100
ALPHA = 0.05
MIN_SIZE = 1
MAX_SIZE_PERCENT = 25.0
STEP = 1
SEED = 218720
RESUME = True
N_JOBS = 1

ACTIVE_METHOD_NAMES = (
    "linear", "quadratic", "poly3",
    "poly4", "spline3", "spline4",
)

FILE_PATTERN = "sim_[0-9][0-9][0-9].csv"

EXPECTED_IMPLEMENTATION_VERSION = (
    "2026-10-01-fixed-R0-population-v1"
)

SINGLE_CLUSTERS = {
    "Single_County": [15],
    "Double_County": [15, 31],
    "Triple_County": [3, 15, 31],
}

TRUE_TWO_CLUSTERS = [
    [3, 15, 31],
    [12, 21, 32],
]

POPULATION_SETTINGS = (
    "Constant",
    "Original_Population",
)


# ------------------------------------------------------------
# Portable paths and scanner import
# ------------------------------------------------------------

def get_paths(part, coordinate_override=None):
    # This file lives in Code/, one level below the repository root.
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "Data"

    required_folders = {
        "single": ("Single_Cluster",),
        "two": ("Two_Cluster",),
        "all": ("Single_Cluster", "Two_Cluster"),
    }[part]

    candidates = (
        data_dir,
        data_dir / "Sim_Data",
    )

    matches = [
        folder for folder in candidates
        if all(
            (folder / name).is_dir()
            for name in required_folders
        )
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one simulation root under Data/ or "
            "Data/Sim_Data/ containing: "
            + ", ".join(required_folders)
            + ". Found: "
            + (", ".join(str(p) for p in matches) or "none")
        )

    sim_root = matches[0]

    if coordinate_override is not None:
        coord_path = Path(coordinate_override).expanduser()
        if not coord_path.is_absolute():
            coord_path = project_root / coord_path
    else:
        coord_path = data_dir / "nm_county_coordinates.txt"

        if not coord_path.is_file():
            matches = sorted({
                path.resolve()
                for directory in ("Data", "Code")
                for path in (project_root / directory).rglob(
                    "nm_county_coordinates.txt"
                )
                if path.is_file()
            })

            if len(matches) != 1:
                raise FileNotFoundError(
                    "Place nm_county_coordinates.txt inside Data/, "
                    "or supply --coord-path with its location."
                )

            coord_path = matches[0]

    coord_path = coord_path.resolve()

    if not coord_path.is_file():
        raise FileNotFoundError(
            f"Coordinate file not found: {coord_path}"
        )

    return project_root, sim_root, coord_path


def load_scanner(project_root):
    scanner_path = (
        project_root / "Code" / "svtt_pooled_ro.py"
    )

    if not scanner_path.is_file():
        raise FileNotFoundError(
            f"Scanner file not found: {scanner_path}"
        )

    module_name = "_flexible_svtt_scanner"
    spec = importlib.util.spec_from_file_location(
        module_name, scanner_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import scanner: {scanner_path}"
        )

    scanner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = scanner

    # Support imports of other Python files inside Code/.
    sys.path.insert(0, str(scanner_path.parent))
    try:
        spec.loader.exec_module(scanner)
    finally:
        sys.path.pop(0)

    required_names = (
        "SVTT_Scan",
        "run_100_saved_files",
        "select_methods",
        "_natural_path_key",
        "IMPLEMENTATION_VERSION",
        "EXPECTED_COUNT_DEFINITION",
    )

    missing = [
        name for name in required_names
        if not hasattr(scanner, name)
    ]

    if missing:
        raise RuntimeError(
            "svtt_pooled_ro.py is missing: "
            + ", ".join(missing)
        )

    if (
        scanner.IMPLEMENTATION_VERSION
        != EXPECTED_IMPLEMENTATION_VERSION
    ):
        raise RuntimeError(
            "Unexpected scanner version: "
            f"{scanner.IMPLEMENTATION_VERSION}. "
            f"Expected {EXPECTED_IMPLEMENTATION_VERSION}."
        )

    return scanner


# ------------------------------------------------------------
# Batch configurations
# ------------------------------------------------------------

def get_configurations(family):
    if family == "Single_Cluster":
        return [
            {
                "parts": (
                    configuration,
                    population,
                    f"Res{residual}",
                ),
                "true_cluster": members,
                "labels": {
                    "configuration": configuration,
                    "population_setting": population,
                    "residual_scenario": residual,
                },
            }
            for configuration, members in SINGLE_CLUSTERS.items()
            for population in POPULATION_SETTINGS
            for residual in (1, 2, 3)
        ]

    return [
        {
            "parts": (scenario, population),
            "true_cluster": TRUE_TWO_CLUSTERS,
            "labels": {
                "scenario": scenario,
                "population_setting": population,
            },
        }
        for scenario in ("Scenario_1", "Scenario_2")
        for population in POPULATION_SETTINGS
    ]


def check_inputs(configs, family, sim_root, scanner):
    problems = []

    for config in configs:
        input_dir = (
            sim_root / family / Path(*config["parts"])
        )

        if not input_dir.is_dir():
            problems.append(
                f"Input folder not found: {input_dir}"
            )
            continue

        files = sorted(
            input_dir.glob(FILE_PATTERN),
            key=scanner._natural_path_key,
        )

        if len(files) < N_FILES:
            problems.append(
                f"{input_dir}: found {len(files)} datasets; "
                f"need {N_FILES}."
            )
        else:
            print(
                f"{input_dir.relative_to(sim_root)}: "
                f"{len(files)} datasets available."
            )

    if problems:
        raise ValueError(
            "Input checks failed:\n"
            + "\n".join(problems)
        )


# ------------------------------------------------------------
# Run and save one family of batches
# ------------------------------------------------------------

def run_family(
    family,
    scanner,
    project_root,
    sim_root,
    coord_path,
):
    configs = get_configurations(family)
    check_inputs(configs, family, sim_root, scanner)

    # Double_County and Triple_County each form one spatial cluster.
    n_max = 1 if family == "Single_Cluster" else 2

    settings = {
        "implementation_version": scanner.IMPLEMENTATION_VERSION,
        "expected_count_definition": scanner.EXPECTED_COUNT_DEFINITION,
        "Bmc": BMC,
        "alpha": ALPHA,
        "min_size": MIN_SIZE,
        "max_size_percent": MAX_SIZE_PERCENT,
        "step": STEP,
        "seed": SEED,
        "n_files": N_FILES,
        "n_max": n_max,
        "n_jobs": N_JOBS,
        "methods": ACTIVE_METHOD_NAMES,
        "true_clusters": (
            SINGLE_CLUSTERS
            if family == "Single_Cluster"
            else TRUE_TWO_CLUSTERS
        ),
    }

    settings_hash = hashlib.sha256(
        json.dumps(
            settings, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:10]

    result_root = (
        project_root
        / "Results"
        / family
        / (
            f"SVTT_pooled_R0_q{MAX_SIZE_PERCENT:g}"
            f"_B{BMC}_{settings_hash}"
        )
    )

    result_root.mkdir(parents=True, exist_ok=True)

    with (
        result_root / "analysis_settings.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(
            settings,
            handle,
            indent=2,
            sort_keys=True,
        )

    methods = scanner.select_methods(
        ACTIVE_METHOD_NAMES
    )

    summary_filename = (
        "all_single_cluster_summaries.csv"
        if family == "Single_Cluster"
        else "all_two_cluster_summaries.csv"
    )

    summaries = []
    all_complete = True

    for config in configs:
        relative_path = Path(*config["parts"])

        input_dir = sim_root / family / relative_path
        output_dir = result_root / relative_path

        print(f"\nRUNNING {family}: {relative_path}")
        print("Output:", output_dir)

        batch = scanner.run_100_saved_files(
            data_dir=input_dir,
            coord_path=coord_path,
            output_dir=output_dir,
            true_cluster=config["true_cluster"],
            n_files=N_FILES,
            n_max=n_max,
            methods=methods,
            Bmc=BMC,
            alpha=ALPHA,
            min_size=MIN_SIZE,
            max_size_percent=MAX_SIZE_PERCENT,
            step=STEP,
            seed=SEED,
            n_jobs=N_JOBS,
            file_pattern=FILE_PATTERN,
            resume=RESUME,
            stop_on_error=False,
        )

        summary = batch["summary"].copy()

        for label, value in reversed(
            list(config["labels"].items())
        ):
            summary.insert(0, label, value)

        summary["batch_complete"] = bool(
            batch["batch_complete"]
        )

        output_dir.mkdir(
            parents=True, exist_ok=True
        )

        summary.to_csv(
            output_dir / "batch_summary.csv",
            index=False,
        )

        summaries.append(summary)

        # Save progress after each batch.
        combined_summary = pd.concat(
            summaries, ignore_index=True
        )

        combined_summary.to_csv(
            result_root / summary_filename,
            index=False,
        )

        display(summary)

        if not batch["batch_complete"]:
            all_complete = False
            print(
                "Some files failed. See failed_files.csv in:",
                output_dir,
            )

    print(
        f"\nFinished {len(configs)} batch calls."
    )
    print(
        "Combined summary:",
        result_root / summary_filename,
    )

    display(combined_summary)

    return all_complete


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run saved SVTT simulation batches."
    )

    parser.add_argument(
        "--part",
        choices=("all", "single", "two"),
        default="all",
    )

    parser.add_argument(
        "--coord-path",
        type=Path,
        default=None,
        help=(
            "Coordinate file path, absolute or relative "
            "to the repository root."
        ),
    )

    args = parser.parse_args(argv)

    project_root, sim_root, coord_path = get_paths(
        args.part,
        args.coord_path,
    )

    scanner = load_scanner(project_root)

    print("Repository:", project_root)
    print("Simulation data:", sim_root)
    print("Coordinates:", coord_path)
    print(
        "Scanner version:",
        scanner.IMPLEMENTATION_VERSION,
    )

    families = {
        "single": ("Single_Cluster",),
        "two": ("Two_Cluster",),
        "all": ("Single_Cluster", "Two_Cluster"),
    }[args.part]

    all_complete = True

    for family in families:
        complete = run_family(
            family,
            scanner,
            project_root,
            sim_root,
            coord_path,
        )
        all_complete = complete and all_complete

    if not all_complete:
        print(
            "\nSome batches are incomplete. Resolve the "
            "failures and rerun with RESUME=True."
        )

    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())