"""SVTT with a pooled reference rate and a Poisson expected-count offset.

For each dataset, R0 = sum(observed counts) / sum(county-year population),
E_it = population_it * R0, and SIR_it (or SMR_it) = observed_it / E_it.
The fitted response remains the original count, with log(E_it) as offset.
Results default to C:/Users/ansarims/Desktop/Sem3_Fall2027/Code/Oct_1.

Portable folder layout
----------------------
Keep this file in the downloaded ``SVTT`` folder alongside
``nm_county_coordinates.txt`` and the data folders.  No computer-specific
absolute path is required.  If the project is stored elsewhere, set the
``SVTT_ROOT`` environment variable or pass ``coord_path`` explicitly when
constructing :class:`SVTT_Scan`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Any, Mapping, Sequence
import ast
import gc
import hashlib
import json
import os
import re
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, rand_score


def find_svtt_root(start: str | Path | None = None) -> Path:
    """Locate the portable SVTT project folder.

    Search order:
      1. ``SVTT_ROOT`` environment variable;
      2. an explicitly supplied ``start`` location;
      3. the directory containing this Python file;
      4. the current working directory;
      5. those locations' parents and nearby ``SVTT`` subfolders.

    The coordinate file is used as the project marker.  If it has not yet
    been copied into the project, the function returns the script directory
    (or the current directory in a notebook); ``SVTT_Scan`` then raises a
    clear error explaining what is missing.
    """

    marker_name = "nm_county_coordinates.txt"
    candidates: list[Path] = []

    root_override = os.environ.get("SVTT_ROOT")
    if root_override:
        candidates.append(Path(root_override).expanduser())

    if start is not None:
        start_path = Path(start).expanduser()
        candidates.append(
            start_path.parent if start_path.is_file() else start_path
        )

    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent)

    candidates.append(Path.cwd())

    expanded_candidates: list[Path] = []
    seen: set[Path] = set()

    for initial in candidates:
        initial = initial.resolve()
        for candidate in (initial, *initial.parents):
            # Prefer an explicitly named SVTT child. This avoids selecting
            # an older parent folder that happens to contain another copy of
            # nm_county_coordinates.txt.
            for possible_root in (candidate / "SVTT", candidate):
                possible_root = possible_root.resolve()
                if possible_root not in seen:
                    seen.add(possible_root)
                    expanded_candidates.append(possible_root)

    for candidate in expanded_candidates:
        if (candidate / marker_name).is_file():
            return candidate

    if candidates:
        return candidates[0].resolve()

    return Path.cwd().resolve()


SVTT_ROOT = find_svtt_root()
DEFAULT_COORD_PATH = SVTT_ROOT / "nm_county_coordinates.txt"
IMPLEMENTATION_VERSION = "2026-10-07-pooled-R0-count-offset-v1"
EXPECTED_COUNT_DEFINITION = "population_it * pooled_R0"
DEFAULT_SIM_ROOT = Path(r"C:\Users\ansarims\Desktop\Sem3_Fall2027\Sim_Data")
DEFAULT_OUTPUT_ROOT = Path(
    r"C:\Users\ansarims\Desktop\Sem3_Fall2027\Code\Oct_1"
)

@dataclass(frozen=True)
class MethodSpec:
    """
    Temporal basis specification.

    Polynomial:
        p is the polynomial degree and number of basis columns.

    Restricted cubic spline:
        p is the number of basis columns.
        The number of internal knots is p - 1.
    """

    name: str
    basis: Literal["poly", "spline"]
    p: int

    def __post_init__(self):

        if self.basis not in {"poly", "spline"}:
            raise ValueError(
                "basis must be either 'poly' or 'spline'."
            )

        if not isinstance(self.p, (int, np.integer)):
            raise TypeError(
                "p must be an integer."
            )

        if self.p < 1:
            raise ValueError(
                "p must be at least 1."
            )



DEFAULT_METHODS: tuple[MethodSpec, ...] = (

    MethodSpec(
        name="linear",
        basis="poly",
        p=1
    ),

    MethodSpec(
        name="quadratic",
        basis="poly",
        p=2
    ),

    MethodSpec(
        name="poly3",
        basis="poly",
        p=3
    ),

    MethodSpec(
        name="poly4",
        basis="poly",
        p=4
    ),

    MethodSpec(
        name="poly5",
        basis="poly",
        p=5
    ),

    # spline1 is identical to the linear model,
    # so it is omitted.

    MethodSpec(
        name="spline2",
        basis="spline",
        p=2
    ),

    MethodSpec(
        name="spline3",
        basis="spline",
        p=3
    ),

    MethodSpec(
        name="spline4",
        basis="spline",
        p=4
    ),

    MethodSpec(
        name="spline5",
        basis="spline",
        p=5
    ),
)


def select_methods(names, available=DEFAULT_METHODS):
    """Select named temporal methods in the caller's requested order."""
    requested = tuple(names)
    lookup = {method.name: method for method in available}
    unknown = sorted(set(requested) - set(lookup))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(lookup)}")
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Supply at least one method, without duplicate names.")
    return tuple(lookup[name] for name in requested)


@dataclass
class SVTT_Scan:

    coord_path: str | Path = field(
        default_factory=lambda: DEFAULT_COORD_PATH
    )

    T_start: int = 1980
    T_end: int = 2000

    Bmc: int = 199

    # A flat list denotes one true cluster. A nested list may be used
    # when metrics should be evaluated against several true clusters.
    true_cluster: list = field(
        default_factory=lambda: [15, 29, 39]
    )

    # Backward-compatible reference values. When population_data is None,
    # Pop is used as a constant population. If population_data varies, the
    # default baseline rate remains E / Pop.
    Pop: float = 10000.0
    E: float = 100.0

    # Flexible population input used only by generate():
    #   scalar; county mapping; county vector; county-by-year matrix;
    #   DataFrame(county, year, population); or callable.
    population_data: Any = None

    # Optional scalar/year-specific baseline rate. When None, E / Pop is used.
    baseline_rate: Any = None

    # Label for the O/E diagnostic: SIR for incidence, SMR for mortality.
    # The label does not change the Poisson count response or its offset.
    ratio_name: str = "SIR"

    n_sim: int = 1
    alpha: float = 0.05

    # Candidate clusters must contain at least min_size counties. Their
    # upper limit is based on population, not on the county count.
    min_size: int = 2
    max_size_percent: float = 25.0
    step: int = 1

    seed: int = 218720

    # Maximum number of non-overlapping clusters to select.
    # Pass either nclust_max=K or n_max=K.
    nclust_max: int = 1
    n_max: int | None = None

    # Accepted scenario numbers
    par: tuple = (1, 2, 3)
    inc: tuple = (1, 2, 3)
    dec: tuple = (1, 2, 3)
    cyc: tuple = (1, 2, 3, 4)
    linear: tuple = (1, 2, 3, 4)
    non_linear: tuple = (1, 2, 3)

    def __post_init__(self):

        self.ratio_name = str(self.ratio_name).strip().upper()
        if self.ratio_name not in {"SIR", "SMR"}:
            raise ValueError("ratio_name must be 'SIR' or 'SMR'.")

        # =====================================================
        # Validate settings
        # =====================================================
        if self.T_start >= self.T_end:
            raise ValueError(
                "T_start must be smaller than T_end so that "
                "at least two time points are available."
            )

        if self.n_sim < 1:
            raise ValueError(
                "n_sim must be at least 1."
            )

        if self.Bmc < 1:
            raise ValueError(
                "Bmc must be at least 1."
            )

        if self.E <= 0:
            raise ValueError(
                "E must be positive."
            )

        if self.Pop <= 0:
            raise ValueError(
                "Pop must be positive."
            )

        if not 0 < self.alpha < 1:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        if self.min_size < 1:
            raise ValueError(
                "min_size must be at least 1."
            )

        if not 0 < self.max_size_percent <= 100:
            raise ValueError(
                "max_size_percent must be in (0, 100]."
            )

        if self.step < 1:
            raise ValueError(
                "step must be at least 1."
            )

        if self.n_max is not None:
            if not isinstance(self.n_max, (int, np.integer)):
                raise TypeError("n_max must be an integer.")
            self.nclust_max = int(self.n_max)

        if not isinstance(self.nclust_max, (int, np.integer)):
            raise TypeError("nclust_max must be an integer.")

        self.nclust_max = int(self.nclust_max)
        self.n_max = self.nclust_max

        if self.nclust_max < 1:
            raise ValueError(
                "nclust_max/n_max must be at least 1."
            )

        # =====================================================
        # Read coordinates
        # =====================================================
        coord_path = Path(self.coord_path).expanduser()

        if not coord_path.is_absolute():
            coord_path = SVTT_ROOT / coord_path

        coord_path = coord_path.resolve()

        if not coord_path.is_file():
            raise FileNotFoundError(
                "The county-coordinate file was not found. Expected: "
                f"{coord_path}. Keep nm_county_coordinates.txt in the "
                "downloaded SVTT folder, pass coord_path explicitly, or "
                "set the SVTT_ROOT environment variable."
            )

        self.coord_path = coord_path

        self.coords = pd.read_csv(
            self.coord_path,
            sep=r"\s+",
            header=None,
            names=["county", "x", "y"]
        )

        self.coords["county"] = (
            self.coords["county"].astype(int)
        )

        if self.coords["county"].duplicated().any():
            duplicates = (
                self.coords.loc[
                    self.coords["county"].duplicated(),
                    "county"
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicate counties found in coordinate file: "
                f"{duplicates}"
            )

        if self.coords[
            ["county", "x", "y"]
        ].isna().any().any():
            raise ValueError(
                "Coordinate data contain missing values."
            )

        # Keep coordinates in county order
        self.coords = (
            self.coords
            .sort_values("county")
            .reset_index(drop=True)
        )

        self.counties = (
            self.coords["county"]
            .to_numpy(dtype=int)
        )

        # =====================================================
        # Time arrays
        # =====================================================
        self.years = np.arange(
            self.T_start,
            self.T_end + 1,
            dtype=int
        )

        self.t = np.arange(
            1,
            len(self.years) + 1,
            dtype=float
        )

        # =====================================================
        # True cluster(s)
        # =====================================================
        raw_true = self.true_cluster

        if raw_true is None:
            self.true_clusters = []
        elif (
            isinstance(raw_true, (list, tuple, set, np.ndarray, pd.Series))
            and len(raw_true) > 0
            and isinstance(
                list(raw_true)[0],
                (list, tuple, set, np.ndarray, pd.Series),
            )
        ):
            self.true_clusters = [
                np.atleast_1d(cluster).astype(int).tolist()
                for cluster in raw_true
            ]
        else:
            self.true_clusters = [
                np.atleast_1d(raw_true).astype(int).tolist()
            ]

        # Flatten for backward compatibility and for binary recovery metrics.
        self.true_cluster = []
        seen_true = set()
        for cluster in self.true_clusters:
            for county in cluster:
                county = int(county)
                if county not in seen_true:
                    seen_true.add(county)
                    self.true_cluster.append(county)

        missing = set(self.true_cluster) - set(self.counties)
        if missing:
            raise ValueError(
                "True-cluster counties were not found in "
                f"the coordinate file: {sorted(missing)}"
            )

        self._true_cluster_set = set(self.true_cluster)
        self._true_cluster_mask = np.isin(
            self.counties,
            self.true_cluster,
        )

        # =====================================================
        # Cached objects
        # =====================================================
        self._basis_cache = {}
        self._design_cache = {}
        self._null_stats_cache = {}

        # Created later by create_candidate_clusters()
        self.candidate_clusters = []

        self.candidate_masks = np.empty(
            (0, len(self.counties)),
            dtype=float
        )

        # Population diagnostics populated by create_candidate_clusters().
        self.reference_population_by_county = np.empty(0, dtype=float)
        self.total_reference_population = np.nan
        self.max_candidate_population = np.nan
        self.candidate_populations = np.empty(0, dtype=float)
        self.candidate_population_percents = np.empty(0, dtype=float)

    def _resolve_population_matrix(self):
        """Return a positive county-by-year population matrix for generation."""

        n_counties = len(self.counties)
        n_times = len(self.years)
        source = self.Pop if self.population_data is None else self.population_data

        if callable(source):
            values = source(self.counties.copy(), self.years.copy())
            population_matrix = np.asarray(values, dtype=float)

        elif isinstance(source, pd.DataFrame):
            required = {"county", "year", "population"}
            missing = required - set(source.columns)
            if missing:
                raise ValueError(
                    "population_data DataFrame is missing columns: "
                    f"{sorted(missing)}"
                )

            population_df = source.loc[:, ["county", "year", "population"]].copy()
            population_df["county"] = pd.to_numeric(
                population_df["county"], errors="raise"
            ).astype(int)
            population_df["year"] = pd.to_numeric(
                population_df["year"], errors="raise"
            ).astype(int)
            population_df["population"] = pd.to_numeric(
                population_df["population"], errors="raise"
            ).astype(float)

            if population_df.duplicated(["county", "year"]).any():
                raise ValueError(
                    "population_data contains duplicated county-year rows."
                )

            pivot = population_df.pivot(
                index="county",
                columns="year",
                values="population",
            ).reindex(index=self.counties, columns=self.years)

            if pivot.isna().any().any():
                raise ValueError(
                    "population_data does not contain every required "
                    "county-year combination."
                )

            population_matrix = pivot.to_numpy(dtype=float)

        elif isinstance(source, Mapping):
            rows = []
            for county in self.counties:
                if int(county) not in source:
                    raise ValueError(
                        f"No population was supplied for county {int(county)}."
                    )
                county_values = np.asarray(source[int(county)], dtype=float)
                if county_values.ndim == 0:
                    county_values = np.full(n_times, float(county_values))
                else:
                    county_values = county_values.reshape(-1)
                    if county_values.size != n_times:
                        raise ValueError(
                            f"Population for county {int(county)} has "
                            f"{county_values.size} values; expected {n_times}."
                        )
                rows.append(county_values)
            population_matrix = np.vstack(rows)

        elif np.isscalar(source):
            population_matrix = np.full(
                (n_counties, n_times),
                float(source),
                dtype=float,
            )

        else:
            values = np.asarray(source, dtype=float)
            if values.shape == (n_counties,):
                population_matrix = np.repeat(values[:, None], n_times, axis=1)
            elif values.shape == (n_counties, n_times):
                population_matrix = values.copy()
            elif values.size == n_counties * n_times:
                population_matrix = values.reshape(n_counties, n_times)
            else:
                raise ValueError(
                    "population_data must be a scalar, mapping, callable, "
                    "DataFrame, county vector, or county-by-year matrix."
                )

        if population_matrix.shape != (n_counties, n_times):
            raise ValueError(
                f"Population matrix has shape {population_matrix.shape}; "
                f"expected {(n_counties, n_times)}."
            )

        if np.any(~np.isfinite(population_matrix)):
            raise ValueError("Population contains non-finite values.")
        if np.any(population_matrix <= 0):
            raise ValueError("Population values must be positive.")

        return population_matrix

    def _resolve_baseline_rate(self):
        """Return the baseline disease rate for every study year."""

        if self.baseline_rate is None:
            if not np.isscalar(self.Pop) or float(self.Pop) <= 0:
                raise ValueError(
                    "Pop must be a positive scalar when baseline_rate is None."
                )
            rates = np.full(
                len(self.years),
                float(self.E) / float(self.Pop),
                dtype=float,
            )
        else:
            source = self.baseline_rate
            if callable(source):
                source = source(self.years.copy())

            if isinstance(source, pd.Series):
                if set(self.years).issubset(set(source.index)):
                    rates = source.reindex(self.years).to_numpy(dtype=float)
                else:
                    rates = source.to_numpy(dtype=float)
            elif np.isscalar(source):
                rates = np.full(len(self.years), float(source), dtype=float)
            else:
                rates = np.asarray(source, dtype=float).reshape(-1)

            if rates.size != len(self.years):
                raise ValueError(
                    f"baseline_rate has {rates.size} values; "
                    f"expected {len(self.years)}."
                )

        if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("Baseline rates must be finite and positive.")

        return rates

    @staticmethod
    def _flatten_clusters(clusters):
        """Recursively flatten one cluster or a collection of clusters."""

        if clusters is None:
            return set()

        if isinstance(clusters, float) and np.isnan(clusters):
            return set()

        if isinstance(clusters, (list, tuple, set, np.ndarray, pd.Series)):
            result = set()
            for item in clusters:
                if isinstance(item, (list, tuple, set, np.ndarray, pd.Series)):
                    result.update(SVTT_Scan._flatten_clusters(item))
                elif item is not None and not (
                    isinstance(item, float) and np.isnan(item)
                ):
                    result.add(int(item))
            return result

        return {int(clusters)}

    def get_scenario_functions(
        self,
        scenario_name,
        scenario_number
    ):
        """
        Return the inside- and outside-cluster relative-risk
        functions for a requested scenario.
        """

        scenario_name = str(
            scenario_name
        ).strip().lower()

        scenario_map = {
            "linear": (
                self.linear,
                self._linear_scenario
            ),
            "lin": (
                self.linear,
                self._linear_scenario
            ),
            "parabolic": (
                self.par,
                self._parabolic_scenario
            ),
            "par": (
                self.par,
                self._parabolic_scenario
            ),
            "increasing": (
                self.inc,
                self._increasing_scenario
            ),
            "inc": (
                self.inc,
                self._increasing_scenario
            ),
            "decreasing": (
                self.dec,
                self._decreasing_scenario
            ),
            "dec": (
                self.dec,
                self._decreasing_scenario
            ),
            "cyclic": (
                self.cyc,
                self._cyclic_scenario
            ),
            "cyc": (
                self.cyc,
                self._cyclic_scenario
            ),
            "residual": (
                self.non_linear,
                self._nonlinear_residual_scenario
            ),
            "res": (
                self.non_linear,
                self._nonlinear_residual_scenario
            ),
            "nonlinear": (
                self.non_linear,
                self._nonlinear_residual_scenario
            ),
            "non-linear": (
                self.non_linear,
                self._nonlinear_residual_scenario
            ),
            "nonlinear_residual": (
                self.non_linear,
                self._nonlinear_residual_scenario
            )
        }

        if scenario_name not in scenario_map:
            raise ValueError(
                "scenario_name must be one of: "
                "linear, parabolic, increasing, decreasing, "
                "cyclic, or residual."
            )

        valid_numbers, scenario_function = (
            scenario_map[scenario_name]
        )

        if scenario_number not in valid_numbers:
            raise ValueError(
                f"{scenario_name} scenario must be one of "
                f"{valid_numbers}."
            )

        return scenario_function(
            scenario_number
        )

    def generate(
        self,
        scenario_name,
        scenario_number
    ):
        """
        Generate spatial-temporal Poisson datasets.

        Population may be constant or vary by county and year. Expected
        baseline counts are population multiplied by the baseline rate.
        """

        inside_fun, outside_fun = self.get_scenario_functions(
            scenario_name,
            scenario_number,
        )

        theta_inside = np.asarray(inside_fun(self.t), dtype=float)
        theta_outside = np.asarray(outside_fun(self.t), dtype=float)

        if theta_inside.shape != self.t.shape:
            raise ValueError(
                f"Inside theta has shape {theta_inside.shape}; "
                f"expected {self.t.shape}."
            )
        if theta_outside.shape != self.t.shape:
            raise ValueError(
                f"Outside theta has shape {theta_outside.shape}; "
                f"expected {self.t.shape}."
            )
        if (
            np.any(~np.isfinite(theta_inside))
            or np.any(~np.isfinite(theta_outside))
            or np.any(theta_inside <= 0)
            or np.any(theta_outside <= 0)
        ):
            raise ValueError(
                "Relative-risk curves must be finite and positive."
            )

        n_counties = len(self.counties)
        n_times = len(self.years)

        population_matrix = self._resolve_population_matrix()
        baseline_rates = self._resolve_baseline_rate()
        E_matrix = population_matrix * baseline_rates[None, :]

        theta_matrix = np.where(
            self._true_cluster_mask[:, None],
            theta_inside[None, :],
            theta_outside[None, :],
        )
        lambda_matrix = E_matrix * theta_matrix

        county_column = np.repeat(self.counties, n_times)
        year_column = np.tile(self.years, n_counties)
        cluster_column = np.repeat(self._true_cluster_mask, n_times)
        x_column = np.repeat(
            self.coords["x"].to_numpy(dtype=float), n_times
        )
        y_column = np.repeat(
            self.coords["y"].to_numpy(dtype=float), n_times
        )
        population_column = population_matrix.reshape(-1)
        E_column = E_matrix.reshape(-1)

        simulation_frames = []
        for sim in range(1, self.n_sim + 1):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(self.seed), int(sim)])
            )
            observed_cases = rng.poisson(lam=lambda_matrix).reshape(-1)

            simulation_frames.append(pd.DataFrame({
                "sim": sim,
                "county": county_column,
                "x": x_column,
                "y": y_column,
                "cluster": cluster_column,
                "population": population_column,
                "E": E_column,
                "observed_cases": observed_cases,
                "year": year_column,
            }))

        return pd.concat(simulation_frames, ignore_index=True)

    def _baseline_curve(self, t):
        """Baseline relative risk outside the true cluster."""
        return np.ones_like(
            np.asarray(t, dtype=float)
        )

    def _scale_time(self, t):
        """Scale time to [0, 1] using the complete study period."""
        t = np.asarray(t, dtype=float)

        denominator = (
            self.t.max() - self.t.min()
        )

        if denominator <= 0:
            raise ValueError(
                "At least two time points are required."
            )

        return (
            (t - self.t.min())
            / denominator
        )

    def _nonlinear_residual_scenario(
        self,
        scenario_number
    ):
        """
        Generate a nonlinear signal after removing its fitted
        intercept, linear, and quadratic components.

        Scenarios
        ---------
        1 : S-shaped signal
        2 : Two-bend signal
        3 : Narrow-bump signal
        """

        if scenario_number not in self.non_linear:
            raise ValueError(
                "Nonlinear residual scenario must be one of "
                f"{self.non_linear}."
            )

        t_reference = np.asarray(
            self.t,
            dtype=float
        )

        t_min = t_reference.min()
        t_max = t_reference.max()
        t_range = t_max - t_min
        centre = (t_min + t_max) / 2

        if t_range <= 0:
            raise ValueError(
                "At least two time points are required."
            )

        if scenario_number == 1:

            steepness = 0.55

            logistic_min = 1.0 / (
                1.0
                + np.exp(
                    -steepness
                    * (t_min - centre)
                )
            )

            logistic_max = 1.0 / (
                1.0
                + np.exp(
                    -steepness
                    * (t_max - centre)
                )
            )

            logistic_range = (
                logistic_max - logistic_min
            )

            def original_curve(t):
                t = np.asarray(t, dtype=float)

                y = 1.0 / (
                    1.0
                    + np.exp(
                        -steepness
                        * (t - centre)
                    )
                )

                return (
                    (y - logistic_min)
                    / logistic_range
                )

        elif scenario_number == 2:

            def original_curve(t):
                t = np.asarray(t, dtype=float)

                scaled = (
                    2.0
                    * np.pi
                    * (t - t_min)
                    / t_range
                )

                return np.sin(scaled)

        else:

            width = 0.06 * t_range

            def original_curve(t):
                t = np.asarray(t, dtype=float)

                return np.exp(
                    -0.5
                    * ((t - centre) / width) ** 2
                )

        # Fit the quadratic model only once
        y_original = original_curve(
            t_reference
        )

        X_reference = np.column_stack([
            np.ones_like(t_reference),
            t_reference,
            t_reference**2
        ])

        beta_hat, *_ = np.linalg.lstsq(
            X_reference,
            y_original,
            rcond=None
        )

        residual_reference = (
            y_original
            - X_reference @ beta_hat
        )

        residual_scale = np.max(
            np.abs(residual_reference)
        )

        if np.isclose(residual_scale, 0.0):
            raise ValueError(
                "The nonlinear residual is identically zero."
            )

        effect_size = 0.5

        def inside_fun(t):
            t = np.asarray(t, dtype=float)

            X_new = np.column_stack([
                np.ones_like(t),
                t,
                t**2
            ])

            residual = (
                original_curve(t)
                - X_new @ beta_hat
            )

            standardized_residual = (
                residual / residual_scale
            )

            theta = (
                1.0
                + effect_size
                * standardized_residual
            )

            if np.any(~np.isfinite(theta)):
                raise ValueError(
                    "The nonlinear relative-risk curve contains "
                    "non-finite values."
                )

            if np.any(theta <= 0):
                raise ValueError(
                    "The nonlinear relative-risk curve contains "
                    "non-positive values."
                )

            return theta

        return inside_fun, self._baseline_curve

    def _linear_scenario(self, scenario):

        if scenario not in self.linear:
            raise ValueError(
                f"Linear scenario must be one of {self.linear}."
            )

        endpoints = (
            (3.0, 0.1),
            (2.0, 1.0),
            (0.1, 3.0),
            (1.0, 2.0)
        )

        start_rr, end_rr = endpoints[
            scenario - 1
        ]

        def inside_fun(t):
            u = self._scale_time(t)

            return (
                start_rr
                + (end_rr - start_rr) * u
            )

        return inside_fun, self._baseline_curve

    def _parabolic_scenario(self, scenario):

        if scenario not in self.par:
            raise ValueError(
                f"Parabolic scenario must be one of {self.par}."
            )

        beta2_values = (
            0.015,
            -0.015,
            -0.03
        )

        R_values = (
            1.5,
            2.0,
            3.0
        )

        beta2 = beta2_values[
            scenario - 1
        ]

        R = R_values[
            scenario - 1
        ]

        reference_curve = (
            beta2
            * (self.t - 4.0)
            * (self.t - 19.0)
        )

        maximum = np.max(
            reference_curve
        )

        if np.isclose(maximum, 0.0):
            raise ValueError(
                "Cannot scale the parabolic curve because its "
                "maximum is zero."
            )

        scale_factor = (
            np.log(R) / maximum
        )

        def inside_fun(t):
            t = np.asarray(t, dtype=float)

            curve = (
                beta2
                * (t - 4.0)
                * (t - 19.0)
            )

            return np.exp(
                scale_factor * curve
            )

        return inside_fun, self._baseline_curve

    def _method_cache_key(
        self,
        method: MethodSpec
        ):
        return (
            method.basis,
            int(method.p),
        )

    def _increasing_scenario(self, scenario):

        if scenario not in self.inc:
            raise ValueError(
                f"Increasing scenario must be one of {self.inc}."
            )

        lambda_values = (
            5.5,
            2.0,
            -2.0
        )

        gamma_values = (
            0.85,
            1.00,
            1.15
        )

        lam = lambda_values[
            scenario - 1
        ]

        gamma = gamma_values[
            scenario - 1
        ]

        log_RR_start = np.log(0.3)
        log_RR_end = np.log(3.0)

        denominator = (
            1.0 - np.exp(-lam)
        )

        def inside_fun(t):
            u = self._scale_time(t)

            if np.isclose(lam, 0.0):
                H = u
            else:
                H = (
                    1.0 - np.exp(-lam * u)
                ) / denominator

            G = H**gamma

            return np.exp(
                log_RR_start
                + (
                    log_RR_end
                    - log_RR_start
                ) * G
            )

        return inside_fun, self._baseline_curve

    def _decreasing_scenario(self, scenario):

        if scenario not in self.dec:
            raise ValueError(
                f"Decreasing scenario must be one of {self.dec}."
            )

        lambda_values = (
            2.0,
            -2.0,
            -6.5
        )

        gamma_values = (
            1.00,
            1.15,
            1.38
        )

        lam = lambda_values[
            scenario - 1
        ]

        gamma = gamma_values[
            scenario - 1
        ]

        log_RR_start = np.log(3.0)
        log_RR_end = np.log(0.3)

        denominator = (
            1.0 - np.exp(-lam)
        )

        def inside_fun(t):
            u = self._scale_time(t)

            if np.isclose(lam, 0.0):
                H = u
            else:
                H = (
                    1.0 - np.exp(-lam * u)
                ) / denominator

            G = H**gamma

            return np.exp(
                log_RR_start
                + (
                    log_RR_end
                    - log_RR_start
                ) * G
            )

        return inside_fun, self._baseline_curve

    def _cyclic_scenario(self, scenario):

        if scenario not in self.cyc:
            raise ValueError(
                f"Cyclic scenario must be one of {self.cyc}."
            )

        amplitude_values = (
            1.2,
            1.5,
            2.0,
            3.0
        )

        frequency_values = (
            1,
            2,
            3,
            4
        )

        amplitude = amplitude_values[
            scenario - 1
        ]

        frequency = frequency_values[
            scenario - 1
        ]

        RR_min = 0.01

        # Use the complete time range to obtain fixed scaling
        reference_u = self._scale_time(
            self.t
        )

        reference_raw = np.sin(
            2.0
            * np.pi
            * frequency
            * reference_u
        )

        raw_min = reference_raw.min()
        raw_range = (
            reference_raw.max() - raw_min
        )

        if np.isclose(raw_range, 0.0):
            raise ValueError(
                "The cyclic curve has zero variation."
            )

        def inside_fun(t):
            u = self._scale_time(t)

            raw = np.sin(
                2.0
                * np.pi
                * frequency
                * u
            )

            standardized = (
                (raw - raw_min)
                / raw_range
            )

            return (
                RR_min
                + (amplitude - RR_min)
                * standardized
            )

        return inside_fun, self._baseline_curve

    def create_candidate_clusters(self):
        """
        Create circular nearest-neighbour candidate clusters and
        their membership masks.

        ``max_size_percent`` is the maximum percentage of the total
        study population allowed inside a candidate cluster. For
        time-varying populations, population is summed over all study
        years. This gives the same shares as using each county's mean
        population over time.
        """

        number_of_counties = len(
            self.counties
        )

        population_matrix = (
            self._resolve_population_matrix()
        )

        # Sum over time for every county. Using the temporal mean would
        # produce exactly the same population percentages.
        reference_population = population_matrix.sum(
            axis=1
        )

        total_reference_population = float(
            reference_population.sum()
        )

        maximum_candidate_population = (
            total_reference_population
            * self.max_size_percent
            / 100.0
        )

        # Protect comparisons at the population boundary from harmless
        # floating-point rounding.
        population_tolerance = max(
            1e-12,
            abs(maximum_candidate_population) * 1e-12,
        )

        coordinates = self.coords[
            ["x", "y"]
        ].to_numpy(dtype=float)

        # Compute distances and sorting only once
        distance_matrix = cdist(
            coordinates,
            coordinates
        )

        nearest_orders = np.argsort(
            distance_matrix,
            axis=1
        )

        candidate_clusters = []
        candidate_indices = []
        candidate_populations = []
        seen = set()

        for centre_index in range(
            number_of_counties
        ):
            nearest = nearest_orders[
                centre_index
            ]

            for size in range(
                self.min_size,
                number_of_counties,
                self.step
            ):
                indices = nearest[:size]

                candidate_population = float(
                    reference_population[indices].sum()
                )

                # Populations are positive, so every larger prefix will
                # also exceed the threshold once this prefix exceeds it.
                if (
                    candidate_population
                    > maximum_candidate_population
                    + population_tolerance
                ):
                    break

                # Sorted index tuples are efficient unique keys
                unique_key = tuple(
                    sorted(indices.tolist())
                )

                if unique_key in seen:
                    continue

                seen.add(unique_key)

                candidate_indices.append(
                    np.asarray(indices, dtype=int)
                )

                candidate_populations.append(
                    candidate_population
                )

                candidate_clusters.append(
                    self.counties[indices].tolist()
                )

        if not candidate_clusters:
            raise ValueError(
                "No candidate cluster satisfies both constraints: "
                f"min_size={self.min_size} and "
                f"max_size_percent={self.max_size_percent}% of the "
                "total study population. Reduce min_size or increase "
                "max_size_percent."
            )

        number_of_candidates = len(
            candidate_clusters
        )

        candidate_masks = np.zeros(
            (
                number_of_candidates,
                number_of_counties
            ),
            dtype=float
        )

        for candidate_id, indices in enumerate(
            candidate_indices
        ):
            candidate_masks[
                candidate_id,
                indices
            ] = 1.0

        self.candidate_clusters = (
            candidate_clusters
        )

        self.candidate_masks = (
            candidate_masks
        )

        self.reference_population_by_county = (
            reference_population
        )

        self.total_reference_population = (
            total_reference_population
        )

        self.max_candidate_population = (
            maximum_candidate_population
        )

        self.candidate_populations = np.asarray(
            candidate_populations,
            dtype=float,
        )

        self.candidate_population_percents = (
            100.0
            * self.candidate_populations
            / self.total_reference_population
        )

        # Candidate configuration changed, so cached null results
        # are no longer valid.
        self._null_stats_cache.clear()

        return candidate_clusters

    @staticmethod
    def _positive_cube(x):

        x = np.asarray(
            x,
            dtype=float
        )

        return np.maximum(
            x,
            0.0
        ) ** 3

    @staticmethod
    def _center_and_orthogonalize(
        basis,
        p,
        tol=1e-10
    ):

        basis = np.asarray(
            basis,
            dtype=float
        )

        if basis.ndim != 2:
            raise ValueError(
                "The raw basis must be two-dimensional."
            )

        if basis.shape[1] != p:
            raise ValueError(
                f"The raw basis has {basis.shape[1]} columns; "
                f"expected p={p}."
            )

        centred_basis = (
            basis
            - basis.mean(
                axis=0,
                keepdims=True
            )
        )

        rank = np.linalg.matrix_rank(
            centred_basis,
            tol=tol
        )

        if rank < p:
            raise ValueError(
                f"Basis rank {rank} is smaller than p={p}."
            )

        Q, _ = np.linalg.qr(
            centred_basis,
            mode="reduced"
        )

        return np.asarray(
            Q[:, :p],
            dtype=float
        )

    def _polynomial_basis(
        self,
        x,
        p
    ):

        x = np.asarray(
            x,
            dtype=float
        )

        # Transform [0, 1] to [-1, 1]
        z = 2.0 * x - 1.0

        raw_basis = np.column_stack([
            z**power
            for power in range(
                1,
                p + 1
            )
        ])

        return self._center_and_orthogonalize(
            raw_basis,
            p
        )

    def _spline_basis(
        self,
        x,
        p
    ):
        """
        Restricted cubic spline with p basis columns
        and p-1 internal knots.
        """

        x = np.asarray(
            x,
            dtype=float
        )

        # spline1 is equivalent to linear
        if p == 1:
            return self._polynomial_basis(
                x,
                p=1
            )

        # Produces p-1 internal-knot probabilities
        probabilities = np.linspace(
            0.0,
            1.0,
            p + 1
        )[1:-1]

        internal_knots = np.asarray(
            np.quantile(
                x,
                probabilities
            ),
            dtype=float
        )

        internal_knots = np.atleast_1d(
            internal_knots
        )

        if np.unique(internal_knots).size != p - 1:
            raise ValueError(
                f"Could not construct {p - 1} unique "
                f"internal knots for spline{p}."
            )

        knots = np.concatenate([
            [x.min()],
            internal_knots,
            [x.max()]
        ])

        right = knots[-1]
        next_right = knots[-2]
        denominator = right - next_right

        if denominator <= 0:
            raise ValueError(
                "Invalid restricted cubic spline knots."
            )

        columns = [x]

        for knot in knots[:-2]:

            column = (
                self._positive_cube(
                    x - knot
                )
                - (
                    (right - knot)
                    / denominator
                )
                * self._positive_cube(
                    x - next_right
                )
                + (
                    (next_right - knot)
                    / denominator
                )
                * self._positive_cube(
                    x - right
                )
            )

            columns.append(
                column
            )

        raw_basis = np.column_stack(
            columns
        )

        return self._center_and_orthogonalize(
            raw_basis,
            p
        )

    def _make_basis(
        self,
        method: MethodSpec
    ):

        key = self._method_cache_key(
            method
        )

        cached = self._basis_cache.get(
            key
        )

        if cached is not None:
            return cached

        scaled_time = np.asarray(
            self._scale_time(self.t),
            dtype=float
        )

        number_of_times = scaled_time.size
        p = int(method.p)

        if number_of_times < 2:
            raise ValueError(
                "At least two time points are required."
            )

        if not np.all(np.isfinite(scaled_time)):
            raise ValueError(
                "Scaled time contains non-finite values."
            )

        if np.unique(scaled_time).size != number_of_times:
            raise ValueError(
                "Time points must be distinct."
            )

        # p basis coefficients plus one intercept
        number_of_parameters = p + 1

        if number_of_parameters >= number_of_times:
            raise ValueError(
                f"{method.name!r} has too many parameters for "
                f"{number_of_times} time points. "
                f"Basis columns: {p}; "
                f"parameters including intercept: "
                f"{number_of_parameters}."
            )

        if method.basis == "poly":

            basis = self._polynomial_basis(
                scaled_time,
                p
            )

        elif method.basis == "spline":

            basis = self._spline_basis(
                scaled_time,
                p
            )

        else:
            raise ValueError(
                "method.basis must be either 'poly' or 'spline'."
            )

        expected_shape = (
            number_of_times,
            p
        )

        if basis.shape != expected_shape:
            raise ValueError(
                f"The basis for {method.name!r} has shape "
                f"{basis.shape}; expected {expected_shape}."
            )

        if not np.all(np.isfinite(basis)):
            raise ValueError(
                f"Non-finite basis values were generated "
                f"for {method.name!r}."
            )

        design_with_intercept = np.column_stack([
            np.ones(number_of_times),
            basis
        ])

        design_rank = np.linalg.matrix_rank(
            design_with_intercept
        )

        if design_rank < design_with_intercept.shape[1]:
            raise ValueError(
                f"The design matrix for {method.name!r} "
                "is rank deficient."
            )

        basis.setflags(
            write=False
        )

        self._basis_cache[key] = basis

        return basis

    def _get_design_matrices(
        self,
        method: MethodSpec
    ):
        """Create and cache alternative and null design matrices."""

        key = self._method_cache_key(method)

        cached = self._design_cache.get(key)

        if cached is not None:
            return cached

        basis = self._make_basis(
            method
        )

        number_of_times = len(
            self.t
        )

        # Design matrix used for separate inside/outside fits
        X_alt = np.column_stack([
            np.ones(number_of_times),
            basis
        ])

        # Null model:
        # separate inside/outside intercepts,
        # but a common temporal curve
        X_null = np.column_stack([
            np.concatenate([
                np.ones(number_of_times),
                np.zeros(number_of_times)
            ]),
            np.concatenate([
                np.zeros(number_of_times),
                np.ones(number_of_times)
            ]),
            np.vstack([
                basis,
                basis
            ])
        ])

        if (
            np.linalg.matrix_rank(X_alt)
            < X_alt.shape[1]
        ):
            raise ValueError(
                f"Alternative design matrix for "
                f"{method.name!r} is rank deficient."
            )

        if (
            np.linalg.matrix_rank(X_null)
            < X_null.shape[1]
        ):
            raise ValueError(
                f"Null design matrix for "
                f"{method.name!r} is rank deficient."
            )

        X_alt.setflags(write=False)
        X_null.setflags(write=False)

        self._design_cache[key] = (
            X_alt,
            X_null
        )

        return X_alt, X_null


    def _prepare_scan_arrays(
        self,
        df_sim
        ):
        
        """
        Convert one simulation to county-by-time matrices and
        calculate expected counts from ONE rate pooled over the study:

            R0 = sum_it(Y_it) / sum_it(P_it)
            E_it = P_it * R0

        Pool all counties and years within this dataset, including counties
        outside a candidate cluster. Each simulation gets its own R0.
        Pre-existing SIR/SMR/expected-count columns are not used for fitting.
        """

        required_columns = {
            "county",
            "year",
            "observed_cases",
            "population",
        }

        missing_columns = (
            required_columns
            - set(df_sim.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing scan columns: "
                f"{sorted(missing_columns)}"
            )

        number_of_counties = len(
            self.counties
        )

        number_of_times = len(
            self.years
        )

        expected_rows = (
            number_of_counties
            * number_of_times
        )

        if len(df_sim) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows for one simulation, "
                f"but received {len(df_sim)}."
            )

        ordered = (
            df_sim
            .sort_values(
                ["county", "year"]
            )
            .reset_index(drop=True)
        )

        actual_counties = (
            ordered["county"]
            .to_numpy()
            .reshape(
                number_of_counties,
                number_of_times,
            )
        )

        actual_years = (
            ordered["year"]
            .to_numpy()
            .reshape(
                number_of_counties,
                number_of_times,
            )
        )

        if not np.array_equal(
            actual_counties,
            np.repeat(self.counties[:, None], number_of_times, axis=1),
        ):
            raise ValueError(
                "The simulation contains unexpected or "
                "missing counties."
            )

        if not np.array_equal(
            actual_years[0],
            self.years,
        ):
            raise ValueError(
                "The simulation contains unexpected or "
                "missing years."
            )

        if not np.all(
            actual_years
            == self.years[None, :]
        ):
            raise ValueError(
                "County-year combinations are incomplete "
                "or duplicated."
            )

        # --------------------------------------------------------
        # Observed cases: county × year
        # --------------------------------------------------------
        y_matrix = (
            ordered["observed_cases"]
            .to_numpy(dtype=float)
            .reshape(
                number_of_counties,
                number_of_times,
            )
        )

        # --------------------------------------------------------
        # Population: county × year
        # --------------------------------------------------------
        population_matrix = (
            ordered["population"]
            .to_numpy(dtype=float)
            .reshape(
                number_of_counties,
                number_of_times,
            )
        )

        if np.any(~np.isfinite(y_matrix)):
            raise ValueError(
                "Observed cases contain non-finite values."
            )

        if np.any(y_matrix < 0):
            raise ValueError(
                "Observed cases cannot be negative."
            )

        if not np.allclose(y_matrix, np.rint(y_matrix), rtol=0.0, atol=1e-8):
            raise ValueError("Observed Poisson counts must be integer-valued.")

        if np.any(~np.isfinite(population_matrix)):
            raise ValueError(
                "Population contains non-finite values."
            )

        if np.any(population_matrix <= 0):
            raise ValueError(
                "Population values must be positive."
            )

        # --------------------------------------------------------
        # Pool the reference rate over ALL county-year rows in this dataset.
        # --------------------------------------------------------
        total_observed_cases = float(y_matrix.sum())
        total_person_time = float(population_matrix.sum())
        if not np.isfinite(total_person_time) or total_person_time <= 0:
            raise ValueError("Total county-year population must be finite and positive.")
        if not np.isfinite(total_observed_cases) or total_observed_cases <= 0:
            raise ValueError(
                "The dataset must contain at least one observed case to "
                "estimate a positive pooled R0 and log-offset."
            )
        pooled_R0 = total_observed_cases / total_person_time
        E_matrix = population_matrix * pooled_R0

        if np.any(~np.isfinite(E_matrix)):
            raise ValueError(
                "Calculated expected counts contain "
                "non-finite values."
            )

        if np.any(E_matrix <= 0):
            raise ValueError(
                "Calculated expected counts must be positive."
            )

        # Pooled expected and observed totals agree over the whole study.
        # Annual expected totals need not equal annual observed totals.
        if not np.allclose(
            E_matrix.sum(),
            total_observed_cases,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Pooled expected and observed study totals do not match."
            )

        self.pooled_R0 = pooled_R0
        self.total_person_time = total_person_time
        self.total_observed_cases = total_observed_cases

        return y_matrix, E_matrix

    def standardize_saved_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the exact pooled inputs used by fitting, plus an O/E audit.

        Population is treated as one year of exposure per county-year row.
        SIR is the incidence label; choose ratio_name='SMR' for death counts.
        The ratio is diagnostic, never a Poisson response.
        """
        ordered = _normalise_saved_dataset_columns(data)
        y_matrix, E_matrix = self._prepare_scan_arrays(ordered)
        ordered["pooled_R0"] = self.pooled_R0
        ordered["expected_cases"] = E_matrix.ravel()
        ordered["log_expected"] = np.log(E_matrix).ravel()
        ordered["observed_expected_ratio"] = (y_matrix / E_matrix).ravel()
        # Remove an opposite, possibly stale diagnostic label from old inputs.
        opposite_label = "SMR" if self.ratio_name == "SIR" else "SIR"
        ordered = ordered.drop(columns=[opposite_label], errors="ignore")
        ordered[self.ratio_name] = ordered["observed_expected_ratio"]
        ordered["ratio_name"] = self.ratio_name
        ordered["implementation_version"] = IMPLEMENTATION_VERSION
        ordered["expected_count_definition"] = EXPECTED_COUNT_DEFINITION
        return ordered

    def _aggregate_all_candidates(
        self,
        y_matrix,
        E_matrix
    ):
        """
        Aggregate cases and expected counts for all candidate
        clusters using matrix multiplication.
        """

        y_matrix = np.asarray(
            y_matrix,
            dtype=float
        )

        E_matrix = np.asarray(
            E_matrix,
            dtype=float
        )

        expected_shape = (
            len(self.counties),
            len(self.years)
        )

        if y_matrix.shape != expected_shape:
            raise ValueError(
                f"y_matrix has shape {y_matrix.shape}; "
                f"expected {expected_shape}."
            )

        if E_matrix.shape != expected_shape:
            raise ValueError(
                f"E_matrix has shape {E_matrix.shape}; "
                f"expected {expected_shape}."
            )

        if len(self.candidate_clusters) == 0:
            self.create_candidate_clusters()

        expected_mask_shape = (
            len(self.candidate_clusters),
            len(self.counties)
        )

        if (
            self.candidate_masks.shape
            != expected_mask_shape
        ):
            raise ValueError(
                "Candidate masks are missing or inconsistent. "
                "Run create_candidate_clusters() again."
            )

        y_inside = (
            self.candidate_masks @ y_matrix
        )

        E_inside = (
            self.candidate_masks @ E_matrix
        )

        y_total = y_matrix.sum(
            axis=0,
            keepdims=True
        )

        E_total = E_matrix.sum(
            axis=0,
            keepdims=True
        )

        y_outside = y_total - y_inside
        E_outside = E_total - E_inside

        return (
            y_inside,
            E_inside,
            y_outside,
            E_outside
        )

    def _fit_poisson_mle(
        self,
        y,
        E,
        X,
        max_iter=40,
        tol=1e-7
    ):
        """
        Fit counts y ~ Poisson(E * exp(X beta)) with offset log(E), using IRLS.

        E is derived from pooled R0 for the dataset. Never pass SIR/SMR as y.

        The returned log-likelihood omits log(y!), which cancels
        exactly when calculating the likelihood-ratio statistic.
        """

        y = np.asarray(y, dtype=float)
        E = np.asarray(E, dtype=float)
        X = np.asarray(X, dtype=float)

        if y.ndim != 1 or E.ndim != 1:
            raise ValueError(
                "y and E must be one-dimensional."
            )

        if y.shape != E.shape:
            raise ValueError(
                "y and E must have the same shape."
            )

        if X.ndim != 2 or X.shape[0] != len(y) or X.shape[1] < 1:
            raise ValueError(
                "The number of rows in X must match y."
            )

        if np.any(~np.isfinite(y)) or np.any(y < 0):
            raise ValueError(
                "Poisson observations must be finite and non-negative."
            )

        if not np.allclose(y, np.rint(y), rtol=0.0, atol=1e-8):
            raise ValueError("Fit integer observed counts, not SIR/SMR ratios.")
        if np.any(~np.isfinite(X)):
            raise ValueError("Model design contains non-finite values.")

        if np.any(~np.isfinite(E)) or np.any(E <= 0):
            raise ValueError(
                "Expected counts must be finite and positive."
            )

        beta = np.zeros(
            X.shape[1],
            dtype=float
        )

        offset = np.log(E)
        converged = False

        for _ in range(max_iter):

            linear_predictor = (
                offset + X @ beta
            )

            clipped_predictor = np.clip(
                linear_predictor,
                -30.0,
                30.0
            )

            mu = np.exp(
                clipped_predictor
            )

            working_response = (
                clipped_predictor
                - offset
                + (y - mu) / np.maximum(mu, 1e-12)
            )

            XtW = X.T * mu
            hessian = XtW @ X
            rhs = XtW @ working_response

            try:
                beta_new = np.linalg.solve(
                    hessian,
                    rhs
                )
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(
                    hessian,
                    rhs,
                    rcond=None
                )[0]

            if np.any(~np.isfinite(beta_new)):
                break

            difference = np.max(
                np.abs(beta_new - beta)
            )

            beta = beta_new

            if difference < tol:
                converged = True
                break

        final_predictor = np.clip(
            offset + X @ beta,
            -30.0,
            30.0
        )

        mu = np.exp(
            final_predictor
        )

        loglik = float(
            np.sum(
                y * final_predictor - mu
            )
        )

        return {
            "loglik": loglik,
            "converged": (
                converged
                and np.isfinite(loglik)
            )
        }

    def _llr_from_aggregates(
        self,
        y_in,
        E_in,
        y_out,
        E_out,
        method
    ):
        """Calculate the LLR for one candidate cluster."""

        X_alt, X_null = (
            self._get_design_matrices(method)
        )

        fit_inside = self._fit_poisson_mle(
            y=y_in,
            E=E_in,
            X=X_alt
        )

        fit_outside = self._fit_poisson_mle(
            y=y_out,
            E=E_out,
            X=X_alt
        )

        y_null = np.concatenate([
            y_in,
            y_out
        ])

        E_null = np.concatenate([
            E_in,
            E_out
        ])

        fit_null = self._fit_poisson_mle(
            y=y_null,
            E=E_null,
            X=X_null
        )

        converged = (
            fit_inside["converged"]
            and fit_outside["converged"]
            and fit_null["converged"]
        )

        if not converged:
            return {
                "LLR": np.nan,
                "ll_alt": np.nan,
                "ll_null": np.nan,
                "converged": False
            }

        ll_alt = (
            fit_inside["loglik"]
            + fit_outside["loglik"]
        )

        ll_null = fit_null["loglik"]

        llr = max(
            0.0,
            float(ll_alt - ll_null)
        )

        return {
            "LLR": llr,
            "ll_alt": float(ll_alt),
            "ll_null": float(ll_null),
            "converged": True
        }

    def _prepare_candidate_aggregates(
        self,
        y_matrix,
        E_matrix
    ):
        """
        Aggregate all candidates once so the same arrays can be
        reused during the scan.
        """

        return self._aggregate_all_candidates(
            y_matrix=y_matrix,
            E_matrix=E_matrix
        )

    def _find_max_llr_from_arrays(
        self,
        y_matrix,
        E_matrix,
        method
    ):
        """
        Return only the maximum LLR. Used for Monte Carlo null
        samples to avoid creating and sorting DataFrames.
        """

        (
            y_inside,
            E_inside,
            y_outside,
            E_outside
        ) = self._prepare_candidate_aggregates(
            y_matrix,
            E_matrix
        )

        best_llr = -np.inf

        for candidate_id in range(
            len(self.candidate_clusters)
        ):
            result = self._llr_from_aggregates(
                y_in=y_inside[candidate_id],
                E_in=E_inside[candidate_id],
                y_out=y_outside[candidate_id],
                E_out=E_outside[candidate_id],
                method=method
            )

            llr = result["LLR"]

            if (
                result["converged"]
                and np.isfinite(llr)
                and llr > best_llr
            ):
                best_llr = llr

        if not np.isfinite(best_llr):
            return np.nan

        return float(best_llr)

    def _find_ranked_llrs_from_arrays(
        self,
        y_matrix,
        E_matrix,
        method
    ):
        """
        Return the top nclust_max LLRs subject to spatial non-overlap.

        Rank 1 is the largest LLR. Rank 2 is the largest LLR among
        candidates not overlapping rank 1, and so on.
        """

        y_inside, E_inside, y_outside, E_outside = (
            self._prepare_candidate_aggregates(y_matrix, E_matrix)
        )

        candidate_values = []
        for candidate_id in range(len(self.candidate_clusters)):
            result = self._llr_from_aggregates(
                y_in=y_inside[candidate_id],
                E_in=E_inside[candidate_id],
                y_out=y_outside[candidate_id],
                E_out=E_outside[candidate_id],
                method=method,
            )
            llr = result["LLR"]
            if result["converged"] and np.isfinite(llr):
                candidate_values.append((float(llr), candidate_id))

        candidate_values.sort(key=lambda item: item[0], reverse=True)

        selected_counties = set()
        selected_llrs = []
        for llr, candidate_id in candidate_values:
            cluster_set = set(self.candidate_clusters[candidate_id])
            if cluster_set & selected_counties:
                continue
            selected_llrs.append(llr)
            selected_counties.update(cluster_set)
            if len(selected_llrs) >= self.nclust_max:
                break

        output = np.full(self.nclust_max, np.nan, dtype=float)
        output[:len(selected_llrs)] = selected_llrs
        return output

    def _find_top_llr_from_arrays(
        self,
        y_matrix,
        E_matrix,
        method
    ):
        """
        Find the best observed candidate cluster(s).

        When nclust_max=1, this avoids constructing and sorting a
        DataFrame containing every candidate.
        """

        (
            y_inside,
            E_inside,
            y_outside,
            E_outside
        ) = self._prepare_candidate_aggregates(
            y_matrix,
            E_matrix
        )

        # Fast single-cluster path
        if self.nclust_max == 1:

            best_candidate_id = None
            best_result = None
            best_llr = -np.inf

            for candidate_id in range(
                len(self.candidate_clusters)
            ):
                result = self._llr_from_aggregates(
                    y_in=y_inside[candidate_id],
                    E_in=E_inside[candidate_id],
                    y_out=y_outside[candidate_id],
                    E_out=E_outside[candidate_id],
                    method=method
                )

                llr = result["LLR"]

                if (
                    result["converged"]
                    and np.isfinite(llr)
                    and llr > best_llr
                ):
                    best_candidate_id = candidate_id
                    best_result = result
                    best_llr = llr

            if best_result is None:
                return pd.DataFrame([{
                    "rank": 1,
                    "candidate_id": None,
                    "method": method.name,
                    "LLR": 0.0,
                    "ll_alt": np.nan,
                    "ll_null": np.nan,
                    "converged": False,
                    "cluster": None,
                    "cluster_size": 0
                }])

            best_cluster = list(
                self.candidate_clusters[
                    best_candidate_id
                ]
            )

            return pd.DataFrame([{
                "rank": 1,
                "candidate_id": best_candidate_id,
                "method": method.name,
                "LLR": float(best_llr),
                "ll_alt": best_result["ll_alt"],
                "ll_null": best_result["ll_null"],
                "converged": True,
                "cluster": best_cluster,
                "cluster_size": len(best_cluster)
            }])

        # Multiple-cluster path
        candidate_results = []

        for candidate_id, cluster in enumerate(
            self.candidate_clusters
        ):
            result = self._llr_from_aggregates(
                y_in=y_inside[candidate_id],
                E_in=E_inside[candidate_id],
                y_out=y_outside[candidate_id],
                E_out=E_outside[candidate_id],
                method=method
            )

            if (
                result["converged"]
                and np.isfinite(result["LLR"])
            ):
                candidate_results.append({
                    "candidate_id": candidate_id,
                    "method": method.name,
                    "LLR": result["LLR"],
                    "ll_alt": result["ll_alt"],
                    "ll_null": result["ll_null"],
                    "converged": True,
                    "cluster": list(cluster),
                    "cluster_size": len(cluster)
                })

        if not candidate_results:
            return pd.DataFrame([{
                "rank": 1,
                "candidate_id": None,
                "method": method.name,
                "LLR": 0.0,
                "ll_alt": np.nan,
                "ll_null": np.nan,
                "converged": False,
                "cluster": None,
                "cluster_size": 0
            }])

        candidate_results.sort(
            key=lambda item: item["LLR"],
            reverse=True
        )

        selected = []
        selected_counties = set()

        for result in candidate_results:

            candidate_set = set(
                result["cluster"]
            )

            if candidate_set & selected_counties:
                continue

            selected.append(result)
            selected_counties.update(
                candidate_set
            )

            if len(selected) >= self.nclust_max:
                break

        result_df = pd.DataFrame(selected)
        result_df.insert(
            0,
            "rank",
            np.arange(1, len(result_df) + 1)
        )

        return result_df

    def _find_top_llr(
        self,
        df_sim,
        method
    ):
        """DataFrame-compatible wrapper for observed scans."""

        y_matrix, E_matrix = (
            self._prepare_scan_arrays(df_sim)
        )

        return self._find_top_llr_from_arrays(
            y_matrix=y_matrix,
            E_matrix=E_matrix,
            method=method
        )

    def _fit_global_null_probabilities(
        self,
        y_matrix,
        E_matrix,
        max_iter=10000,
        tol=1e-10,
    ):
        """
        Fit the conditional global null with county-specific intercepts.

        The global no-interaction model is

            log(mu_it) = log(E_it) + gamma_i + f_0(t).

        Conditional on the observed total N_t at time t, f_0(t) cancels and

            p_it = E_it * exp(gamma_i)
                   / sum_j E_jt * exp(gamma_j).

        The county multipliers exp(gamma_i) are estimated by iterative
        proportional fitting.  This is the conditional maximum-likelihood fit
        of the county main effects with the time margins held fixed.  The
        resulting fitted county totals match the observed county totals, up to
        the numerical tolerance.
        """

        y_matrix = np.asarray(y_matrix, dtype=float)
        E_matrix = np.asarray(E_matrix, dtype=float)

        expected_shape = (len(self.counties), len(self.years))
        if y_matrix.shape != expected_shape:
            raise ValueError(
                f"y_matrix has shape {y_matrix.shape}; "
                f"expected {expected_shape}."
            )
        if E_matrix.shape != expected_shape:
            raise ValueError(
                f"E_matrix has shape {E_matrix.shape}; "
                f"expected {expected_shape}."
            )
        if np.any(~np.isfinite(y_matrix)) or np.any(y_matrix < 0):
            raise ValueError("Observed counts must be finite and non-negative.")
        if np.any(~np.isfinite(E_matrix)) or np.any(E_matrix <= 0):
            raise ValueError("Expected counts must be finite and positive.")
        if not np.allclose(
            y_matrix,
            np.rint(y_matrix),
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError("Observed Poisson counts must be integer-valued.")

        total_cases_by_time = y_matrix.sum(axis=0)
        observed_county_totals = y_matrix.sum(axis=1)

        # A zero-count year is valid with pooled positive expected counts;
        # its conditional Monte Carlo total is simply zero.

        # Counties with a zero total have their conditional MLE on the
        # boundary, exp(gamma_i)=0.  Positive-total counties are fitted by
        # multiplicative iterative proportional fitting.
        active = observed_county_totals > 0
        if not np.any(active):
            raise ValueError("At least one county must contain observed cases.")

        risk_multipliers = np.ones(len(self.counties), dtype=float)
        risk_multipliers[~active] = 0.0
        converged = False

        for iteration in range(1, int(max_iter) + 1):
            weighted_expected = (
                E_matrix * risk_multipliers[:, None]
            )
            denominators = weighted_expected.sum(axis=0)

            if (
                np.any(~np.isfinite(denominators))
                or np.any(denominators <= 0)
            ):
                raise RuntimeError(
                    "The fitted global-null denominators are invalid."
                )

            probabilities = weighted_expected / denominators[None, :]
            fitted_mean = probabilities * total_cases_by_time[None, :]
            fitted_county_totals = fitted_mean.sum(axis=1)

            relative_error = np.max(
                np.abs(
                    fitted_county_totals[active]
                    - observed_county_totals[active]
                )
                / np.maximum(1.0, observed_county_totals[active])
            )

            if relative_error <= tol:
                converged = True
                break

            if np.any(fitted_county_totals[active] <= 0):
                raise RuntimeError(
                    "A positive-total county received a zero fitted total "
                    "under the global null."
                )

            risk_multipliers[active] *= (
                observed_county_totals[active]
                / fitted_county_totals[active]
            )

            # The intercepts are identifiable only up to a common additive
            # constant.  Centering their logarithms prevents numerical drift
            # without changing any conditional probability.
            log_active = np.log(risk_multipliers[active])
            risk_multipliers[active] = np.exp(
                log_active - log_active.mean()
            )

        if not converged:
            raise RuntimeError(
                "The fitted global-null county intercepts did not converge "
                f"within {max_iter} iterations."
            )

        # Recalculate once from the final multipliers retained at convergence.
        weighted_expected = E_matrix * risk_multipliers[:, None]
        probabilities = (
            weighted_expected
            / weighted_expected.sum(axis=0, keepdims=True)
        )
        fitted_mean = probabilities * total_cases_by_time[None, :]
        fitted_county_totals = fitted_mean.sum(axis=1)

        if not np.allclose(
            probabilities.sum(axis=0),
            1.0,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Fitted global-null probabilities do not sum to one."
            )

        max_margin_error = float(
            np.max(
                np.abs(
                    fitted_county_totals
                    - observed_county_totals
                )
            )
        )

        gamma = np.full(len(self.counties), -np.inf, dtype=float)
        gamma[active] = np.log(risk_multipliers[active])

        probabilities.setflags(write=False)
        fitted_mean.setflags(write=False)
        gamma.setflags(write=False)
        risk_multipliers.setflags(write=False)

        return {
            "probabilities": probabilities,
            "fitted_mean": fitted_mean,
            "gamma": gamma,
            "risk_multipliers": risk_multipliers,
            "observed_county_totals": observed_county_totals,
            "fitted_county_totals": fitted_county_totals,
            "total_cases_by_time": total_cases_by_time,
            "iterations": int(iteration),
            "max_county_margin_error": max_margin_error,
            "converged": True,
        }

    def _one_null_stat(
        self,
        y_matrix,
        E_matrix,
        null_probabilities,
        method,
        seed_b
    ):
        """
        Generate one conditional global-null dataset and return ranked LLRs.

        For every time point, the observed total number of cases is fixed and
        redistributed using

            p_it proportional to E_it * exp(gamma_i),

        where gamma_i is the fitted county-specific baseline intercept.  The
        common temporal component cancels after conditioning on the time total.
        """

        y_matrix = np.asarray(y_matrix, dtype=float)
        E_matrix = np.asarray(E_matrix, dtype=float)
        null_probabilities = np.asarray(
            null_probabilities,
            dtype=float,
        )

        expected_shape = (len(self.counties), len(self.years))

        if y_matrix.shape != expected_shape:
            raise ValueError(
                f"y_matrix has shape {y_matrix.shape}; "
                f"expected {expected_shape}."
            )
        if E_matrix.shape != expected_shape:
            raise ValueError(
                f"E_matrix has shape {E_matrix.shape}; "
                f"expected {expected_shape}."
            )
        if null_probabilities.shape != expected_shape:
            raise ValueError(
                "null_probabilities has shape "
                f"{null_probabilities.shape}; expected {expected_shape}."
            )
        if np.any(~np.isfinite(y_matrix)) or np.any(y_matrix < 0):
            raise ValueError("Observed counts must be finite and non-negative.")
        if np.any(~np.isfinite(E_matrix)) or np.any(E_matrix <= 0):
            raise ValueError("Expected counts must be finite and positive.")
        if (
            np.any(~np.isfinite(null_probabilities))
            or np.any(null_probabilities < 0)
        ):
            raise ValueError(
                "Global-null probabilities must be finite and non-negative."
            )
        if not np.allclose(
            null_probabilities.sum(axis=0),
            1.0,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Global-null probabilities must sum to one at every time."
            )

        total_cases_by_time = y_matrix.sum(axis=0)
        rounded_totals = np.rint(total_cases_by_time)

        if not np.allclose(
            total_cases_by_time,
            rounded_totals,
            rtol=0.0,
            atol=1e-8,
        ):
            raise ValueError(
                "Observed case totals must be integer-valued for multinomial "
                "null generation."
            )

        rng = np.random.default_rng(seed_b)
        n_counties, n_times = E_matrix.shape
        y_null_matrix = np.zeros((n_counties, n_times), dtype=float)

        for time_index in range(n_times):
            total_cases_t = int(rounded_totals[time_index])
            probabilities_t = null_probabilities[:, time_index]
            y_null_matrix[:, time_index] = rng.multinomial(
                n=total_cases_t,
                pvals=probabilities_t,
            )

        return self._find_ranked_llrs_from_arrays(
            y_matrix=y_null_matrix,
            E_matrix=E_matrix,
            method=method,
        )

    def _get_null_stats(
        self,
        y_matrix,
        E_matrix,
        null_probabilities,
        method,
        seed,
        n_jobs=1
    ):
        """
        Return a Bmc-by-nclust_max matrix of rank-specific null LLRs.

        n_jobs=1 is the safest option in Jupyter. Positive values greater
        than one use a thread backend rather than Windows process spawning.
        """

        y_matrix = np.asarray(y_matrix, dtype=float)
        E_matrix = np.asarray(E_matrix, dtype=float)
        null_probabilities = np.asarray(
            null_probabilities,
            dtype=float,
        )
        method_key = self._method_cache_key(method)

        cache_key = (
            method_key,
            int(seed),
            int(self.Bmc),
            int(self.nclust_max),
            y_matrix.shape,
            y_matrix.sum(axis=0).tobytes(),
            E_matrix.shape,
            E_matrix.tobytes(),
            null_probabilities.tobytes(),
            tuple(tuple(cluster) for cluster in self.candidate_clusters),
        )

        cached = self._null_stats_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if n_jobs in (None, 0, -1):
            safe_jobs = 1
        else:
            safe_jobs = max(1, int(n_jobs))

        seeds = [int(seed) + replication + 1 for replication in range(self.Bmc)]

        if safe_jobs == 1:
            null_stats = [
                self._one_null_stat(
                    y_matrix=y_matrix,
                    E_matrix=E_matrix,
                    null_probabilities=null_probabilities,
                    method=method,
                    seed_b=seed_b,
                )
                for seed_b in seeds
            ]
        else:
            null_stats = Parallel(
                n_jobs=safe_jobs,
                prefer="threads",
                verbose=0,
            )(
                delayed(self._one_null_stat)(
                    y_matrix=y_matrix,
                    E_matrix=E_matrix,
                    null_probabilities=null_probabilities,
                    method=method,
                    seed_b=seed_b,
                )
                for seed_b in seeds
            )

        null_stats = np.asarray(null_stats, dtype=float)
        expected_shape = (self.Bmc, self.nclust_max)
        if null_stats.shape != expected_shape:
            raise RuntimeError(
                f"Null-statistic matrix has shape {null_stats.shape}; "
                f"expected {expected_shape}."
            )

        self._null_stats_cache[cache_key] = null_stats.copy()
        return null_stats

    def _calibrate_mc_cutoff(
        self,
        y_matrix,
        E_matrix,
        null_probabilities,
        method: MethodSpec,
        seed,
        n_jobs=1
    ):
        """Calibrate a separate Monte Carlo cutoff for every selected rank."""

        null_stats = self._get_null_stats(
            y_matrix=y_matrix,
            E_matrix=E_matrix,
            null_probabilities=null_probabilities,
            method=method,
            seed=int(seed),
            n_jobs=n_jobs,
        )

        finite_mask = np.isfinite(null_stats)
        n_mc_valid = finite_mask.sum(axis=0).astype(int)
        n_mc_failed = (~finite_mask).sum(axis=0).astype(int)

        if np.any(n_mc_failed > 0):
            failed = {
                rank + 1: int(value)
                for rank, value in enumerate(n_mc_failed)
                if value > 0
            }
            raise RuntimeError(
                "Some null replications could not produce the requested "
                f"number of non-overlapping clusters: {failed}. "
                "Reduce n_max/nclust_max or candidate min_size."
            )

        cutoffs = np.full(self.nclust_max, np.inf, dtype=float)

        for rank_index in range(self.nclust_max):
            rank_null = null_stats[:, rank_index]
            n_valid = int(n_mc_valid[rank_index])

            max_exceedances = int(
                np.floor(self.alpha * (n_valid + 1) - 1.0 + 1e-12)
            )
            if max_exceedances >= 0:
                ordered = np.sort(rank_null)
                cutoffs[rank_index] = float(
                    ordered[-(max_exceedances + 1)]
                )

        return {
            "null_stats": null_stats,
            "cutoffs": cutoffs,
            # Backward-compatible first-rank entries.
            "cutoff": float(cutoffs[0]),
            "n_mc_valid_by_rank": n_mc_valid,
            "n_mc_failed_by_rank": n_mc_failed,
            "n_mc_valid": int(n_mc_valid[0]),
            "n_mc_failed": int(n_mc_failed[0]),
        }

    def monte_carlo_pvalue_one_sim(
        self,
        df_sim,
        method: MethodSpec,
        calibration
    ):
        """Scan one dataset and calculate rank-specific Monte Carlo p-values."""

        y_matrix, E_matrix = self._prepare_scan_arrays(df_sim)
        observed_top = self._find_top_llr_from_arrays(
            y_matrix=y_matrix,
            E_matrix=E_matrix,
            method=method,
        )

        if observed_top.empty or not observed_top["converged"].any():
            return pd.DataFrame([{
                "method": method.name,
                "rank": 1,
                "T_obs": 0.0,
                "ll_alt": np.nan,
                "ll_null": np.nan,
                "converged": False,
                "p_value": 1.0,
                "mc_cutoff": float(calibration["cutoffs"][0]),
                "significant": False,
                "best_cluster": None,
                "cluster_size": 0,
                "n_mc_valid": int(calibration["n_mc_valid_by_rank"][0]),
                "n_mc_failed": int(calibration["n_mc_failed_by_rank"][0]),
            }])

        null_stats = np.asarray(calibration["null_stats"], dtype=float)
        cutoffs = np.asarray(calibration["cutoffs"], dtype=float)
        result_rows = []

        for observation in observed_top.itertuples(index=False):
            rank = int(observation.rank)
            rank_index = rank - 1
            if rank_index >= self.nclust_max:
                raise RuntimeError(
                    f"Observed rank {rank} exceeds nclust_max={self.nclust_max}."
                )

            rank_null = null_stats[:, rank_index]
            T_obs = float(observation.LLR)
            exceedances = int(np.count_nonzero(rank_null >= T_obs))
            p_value = (1 + exceedances) / (len(rank_null) + 1)
            cutoff = float(cutoffs[rank_index])
            significant = bool(T_obs > cutoff)

            if significant != bool(p_value <= self.alpha):
                raise RuntimeError(
                    "The fixed-cutoff and Monte Carlo p-value rules "
                    f"disagree for {method.name}, rank {rank}."
                )

            result_rows.append({
                "method": method.name,
                "rank": rank,
                "T_obs": T_obs,
                "ll_alt": observation.ll_alt,
                "ll_null": observation.ll_null,
                "converged": True,
                "p_value": float(p_value),
                "mc_cutoff": cutoff,
                "significant": significant,
                "best_cluster": observation.cluster,
                "cluster_size": int(observation.cluster_size),
                "n_mc_valid": int(
                    calibration["n_mc_valid_by_rank"][rank_index]
                ),
                "n_mc_failed": int(
                    calibration["n_mc_failed_by_rank"][rank_index]
                ),
            })

        return pd.DataFrame(result_rows)

    def monte_carlo_all_simulations(
        self,
        generated_data,
        methods=DEFAULT_METHODS,
        seed=None,
        n_jobs=1
    ):
        """Run every simulation using rank-specific multiple-cluster nulls."""

        if seed is None:
            seed = self.seed

        grouped_simulations = list(generated_data.groupby("sim", sort=True))
        if not grouped_simulations:
            return pd.DataFrame()

        result_frames = []
        for sim_id, df_sim in grouped_simulations:
            y_matrix, E_matrix = self._prepare_scan_arrays(df_sim)
            global_null_fit = self._fit_global_null_probabilities(
                y_matrix=y_matrix,
                E_matrix=E_matrix,
            )
            null_probabilities = global_null_fit["probabilities"]

            for method_index, method in enumerate(methods):
                method_seed = (
                    int(seed)
                    + int(sim_id) * 1_000_000
                    + method_index * 10_000
                )
                calibration = self._calibrate_mc_cutoff(
                    y_matrix=y_matrix,
                    E_matrix=E_matrix,
                    null_probabilities=null_probabilities,
                    method=method,
                    seed=method_seed,
                    n_jobs=n_jobs,
                )

                result_df = self.monte_carlo_pvalue_one_sim(
                    df_sim=df_sim,
                    method=method,
                    calibration=calibration,
                )
                result_df.insert(0, "sim", int(sim_id))
                result_frames.append(result_df)

                # Do not retain a large calibration for all methods.
                del calibration

        if not result_frames:
            return pd.DataFrame()

        columns = [
            "sim", "method", "rank", "T_obs", "ll_alt", "ll_null",
            "converged", "p_value", "mc_cutoff", "significant",
            "best_cluster", "cluster_size", "n_mc_valid", "n_mc_failed",
        ]
        return pd.concat(result_frames, ignore_index=True)[columns]

    def compute_cluster_metrics(
        self,
        detected_cluster,
        true_cluster=None,
        all_counties=None
    ):
        """Calculate binary recovery metrics for one or many clusters."""

        if true_cluster is None:
            true_cluster = self.true_clusters
        if all_counties is None:
            all_counties = self.counties

        all_counties = np.asarray(all_counties, dtype=int)
        true_set = self._flatten_clusters(true_cluster)
        detected_set = self._flatten_clusters(detected_cluster)

        y_true = np.isin(all_counties, list(true_set))
        y_pred = np.isin(all_counties, list(detected_set))

        TP = int(np.count_nonzero(y_true & y_pred))
        TN = int(np.count_nonzero(~y_true & ~y_pred))
        FP = int(np.count_nonzero(~y_true & y_pred))
        FN = int(np.count_nonzero(y_true & ~y_pred))

        sensitivity = TP / (TP + FN) if TP + FN > 0 else 0.0
        specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        jaccard = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0.0
        accuracy = (TP + TN) / len(all_counties)

        return {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "jaccard": jaccard,
            "rand_index": float(rand_score(y_true, y_pred)),
            "adjusted_rand_index": float(
                adjusted_rand_score(y_true, y_pred)
            ),
            "purity": accuracy,
            "accuracy": accuracy,
            "exact_recovery": bool(true_set == detected_set),
        }

    def summarize_power_and_metrics(
        self,
        mc_df
    ):
        """
        Summarize one row per simulation-method, combining all significant
        non-overlapping clusters selected up to nclust_max.
        """

        if mc_df.empty:
            return pd.DataFrame()

        metric_rows = []
        for (sim_id, method_name), group in mc_df.groupby(
            ["sim", "method"], sort=False
        ):
            group = group.sort_values("rank").copy()
            valid = group["converged"].astype(bool)
            significant = valid & group["significant"].astype(bool)
            detected_clusters = group.loc[
                significant, "best_cluster"
            ].tolist()

            metrics = self.compute_cluster_metrics(
                detected_cluster=detected_clusters
            )
            rejected = bool(significant.any())

            metric_rows.append({
                "sim": int(sim_id),
                "method": method_name,
                "significant": rejected,
                "min_p_value": float(group["p_value"].min()),
                "T_obs_max": float(group["T_obs"].max()),
                "n_selected": int(valid.sum()),
                "n_significant": int(significant.sum()),
                "detected_clusters": detected_clusters,
                "jaccard_rej": metrics["jaccard"] if rejected else np.nan,
                **metrics,
            })

        metrics_df = pd.DataFrame(metric_rows)
        return (
            metrics_df
            .groupby("method", as_index=False)
            .agg(
                n_sim=("sim", "nunique"),
                n_rejected=("significant", "sum"),
                power=("significant", "mean"),
                mean_min_p_value=("min_p_value", "mean"),
                mean_T_obs_max=("T_obs_max", "mean"),
                mean_n_selected=("n_selected", "mean"),
                mean_n_significant=("n_significant", "mean"),
                sensitivity=("sensitivity", "mean"),
                specificity=("specificity", "mean"),
                precision=("precision", "mean"),
                jaccard=("jaccard", "mean"),
                jaccard_rej=("jaccard_rej", "mean"),
                rand_index=("rand_index", "mean"),
                adjusted_rand_index=("adjusted_rand_index", "mean"),
                purity=("purity", "mean"),
                accuracy=("accuracy", "mean"),
                exact_recovery=("exact_recovery", "mean"),
            )
        )

    def summarize_type1_error(
    self,
    mc_null_df
    ):
        
        if mc_null_df.empty:
            return pd.DataFrame()

        data = mc_null_df[
            [
                "sim",
                "method",
                "p_value",
                "significant",
                "converged"
            ]
        ].copy()

        # Use only valid converged simulations
        data = data[
            data["converged"]
        ].copy()

        data["false_rejection"] = (
            data["significant"].astype(bool)
        )

        summary = (
            data
            .groupby(
                "method",
                as_index=False
            )
            .agg(
                n_null_sim=("sim", "nunique"),

                n_false_rejections=(
                    "false_rejection",
                    "sum"
                ),

                type1_error=(
                    "false_rejection",
                    "mean"
                ),

                null_mean_p_value=(
                    "p_value",
                    "mean"
                )
            )
        )

        summary["type1_se"] = np.sqrt(
            summary["type1_error"]
            * (1.0 - summary["type1_error"])
            / summary["n_null_sim"]
        )

        return summary




# ======================================================================
# SAVED-DATA BATCH UTILITIES
# ======================================================================


def _natural_path_key(path: Path):
    """Sort file names naturally: sim_2.csv before sim_10.csv."""

    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", path.name)
    ]


def _normalise_saved_dataset_columns(
    data: pd.DataFrame,
    source_path: Path | None = None,
) -> pd.DataFrame:
    """
    Normalize one saved simulation file.

    Required analysis columns after normalization:
        county, year, population, observed_cases

    Population is read directly from every county-year row. It is never
    replaced by a fixed value such as 10,000.
    """

    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]

    # Normalize both capitalization differences and common aliases. For
    # example, Population, POPULATION, pop, and Population_2021 all become
    # the canonical analysis column ``population``.
    canonical_names = {
        "county": "county",
        "year": "year",
        "population": "population",
        "population_2021": "population",
        "pop": "population",
        "observed_cases": "observed_cases",
        "cases": "observed_cases",
        "case": "observed_cases",
        "observed": "observed_cases",
        "observed_case": "observed_cases",
        "deaths": "observed_cases",
        "observed_deaths": "observed_cases",
    }

    rename_columns = {}
    canonical_sources = {}

    for column in data.columns:
        normalized_name = str(column).strip().lower()
        canonical_name = canonical_names.get(normalized_name)

        if canonical_name is None:
            continue

        if canonical_name in canonical_sources:
            previous_column = canonical_sources[canonical_name]
            raise ValueError(
                "Multiple columns map to the same required field "
                f"'{canonical_name}' in {source_path}: "
                f"'{previous_column}' and '{column}'."
            )

        canonical_sources[canonical_name] = column
        rename_columns[column] = canonical_name

    data = data.rename(columns=rename_columns)

    required = {
        "county",
        "year",
        "population",
        "observed_cases",
    }
    missing = required - set(data.columns)
    if missing:
        location = f" in {source_path}" if source_path is not None else ""
        raise ValueError(
            f"Missing required columns{location}: {sorted(missing)}"
        )

    for column in ("county", "year", "population", "observed_cases"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    for column in ("county", "year"):
        values = data[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or not np.all(values == np.rint(values)):
            raise ValueError(f"{column} must contain finite integers in {source_path}.")
        data[column] = data[column].astype(int)
    data["population"] = data["population"].astype(float)
    data["observed_cases"] = data["observed_cases"].astype(float)

    if not np.all(np.isfinite(data["population"])):
        raise ValueError(f"Population contains non-finite values in {source_path}.")
    if np.any(data["population"] <= 0):
        raise ValueError(f"Population must be positive in {source_path}.")
    if not np.all(np.isfinite(data["observed_cases"])):
        raise ValueError(
            f"Observed cases contain non-finite values in {source_path}."
        )
    if np.any(data["observed_cases"] < 0):
        raise ValueError(f"Observed cases cannot be negative in {source_path}.")

    duplicate_mask = data.duplicated(["county", "year"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            data.loc[duplicate_mask, ["county", "year"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"Duplicate county-year rows in {source_path}: {duplicate_rows}"
        )

    return (
        data
        .sort_values(["county", "year"])
        .reset_index(drop=True)
    )


def _as_cluster_mask(values: pd.Series) -> np.ndarray:
    """Convert common TRUE/FALSE, 1/0, and string encodings to a mask."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_known = numeric.notna()

    result = np.zeros(len(values), dtype=bool)
    if numeric_known.any():
        result[numeric_known.to_numpy()] = (
            numeric.loc[numeric_known].to_numpy(dtype=float) > 0
        )

    unknown = ~numeric_known
    if unknown.any():
        normalized = (
            values.loc[unknown]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        true_tokens = {
            "true", "t", "yes", "y", "inside", "cluster", "1"
        }
        result[unknown.to_numpy()] = normalized.isin(true_tokens).to_numpy()

    return result


def infer_true_clusters_from_saved_data(
    data: pd.DataFrame,
) -> list[list[int]] | None:
    """
    Infer the true cluster definition when the saved file includes truth.

    Preferred format:
        cluster_id = 0 outside, 1..K for true clusters.

    Fallback format:
        cluster = TRUE/FALSE or 1/0 for one true cluster.
    """

    if "cluster_id" in data.columns:
        cluster_ids = pd.to_numeric(data["cluster_id"], errors="coerce")
        positive_ids = sorted(
            int(value)
            for value in cluster_ids.dropna().unique()
            if float(value) > 0
        )

        clusters = []
        for cluster_id in positive_ids:
            counties = sorted(
                data.loc[
                    cluster_ids == cluster_id,
                    "county",
                ]
                .astype(int)
                .unique()
                .tolist()
            )
            if counties:
                clusters.append(counties)

        if clusters:
            return clusters

    if "cluster" in data.columns:
        mask = _as_cluster_mask(data["cluster"])
        counties = sorted(
            data.loc[mask, "county"]
            .astype(int)
            .unique()
            .tolist()
        )
        if counties:
            return [counties]

    return None


def _canonical_clusters(
    clusters: Sequence[Sequence[int]] | Sequence[int] | None,
) -> tuple[tuple[int, ...], ...]:
    """Canonical representation used to compare truth across files."""

    if clusters is None:
        return tuple()

    values = list(clusters)
    if not values:
        return tuple()

    if not isinstance(
        values[0],
        (list, tuple, set, np.ndarray, pd.Series),
    ):
        values = [values]

    normalized = [
        tuple(sorted(int(county) for county in cluster))
        for cluster in values
    ]
    return tuple(sorted(normalized))


def _parse_cluster_cell(value):
    """Restore list-valued best_cluster cells after reading a checkpoint CSV."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [int(item) for item in value]

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None

    if parsed is None:
        return None
    return [int(item) for item in np.atleast_1d(parsed).tolist()]


def discover_saved_simulation_files(
    data_dir,
    n_files=100,
    pattern="*.csv",
    require_exact_count=True,
) -> list[Path]:
    """Return the first n_files in natural filename order."""

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")

    files = sorted(data_dir.glob(pattern), key=_natural_path_key)

    if require_exact_count and len(files) < n_files:
        raise FileNotFoundError(
            f"Requested {n_files} files, but only {len(files)} matching "
            f"{pattern!r} were found in {data_dir}."
        )

    selected = files[:n_files]
    if not selected:
        raise FileNotFoundError(
            f"No files matching {pattern!r} were found in {data_dir}."
        )

    return selected



def _atomic_write_csv(dataframe, path):
    """Write a CSV checkpoint atomically so interruption cannot corrupt it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    dataframe.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _read_result_checkpoint(path):
    """Read one result checkpoint and restore list-valued cluster columns."""

    dataframe = pd.read_csv(path)
    if "best_cluster" in dataframe.columns:
        dataframe["best_cluster"] = dataframe["best_cluster"].apply(
            _parse_cluster_cell
        )
    return dataframe


def _result_has_all_methods(result_df, methods, analysis_signature=None):
    """Accept only results from the pooled implementation and this analysis."""

    if result_df is None or result_df.empty or "method" not in result_df.columns:
        return False

    required_metadata = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "expected_count_definition": EXPECTED_COUNT_DEFINITION,
    }
    if analysis_signature is not None:
        required_metadata["analysis_signature"] = analysis_signature
    for column, expected in required_metadata.items():
        if column not in result_df or not result_df[column].eq(expected).all():
            return False

    expected_methods = {method.name for method in methods}
    completed_methods = set(result_df["method"].astype(str).unique())
    return expected_methods.issubset(completed_methods)


def _build_rank_summary(all_results):
    """Create the rank-specific summary used by the batch runner."""

    if all_results.empty:
        return pd.DataFrame()

    return (
        all_results
        .groupby(["method", "rank"], as_index=False)
        .agg(
            n_files=("sim", "nunique"),
            n_converged=("converged", "sum"),
            n_significant=("significant", "sum"),
            rejection_rate=("significant", "mean"),
            mean_p_value=("p_value", "mean"),
            mean_T_obs=("T_obs", "mean"),
            mean_cluster_size=("cluster_size", "mean"),
        )
    )


def _load_completed_file_results(per_file_dir, methods, analysis_signature=None):
    """Load only file-level checkpoints containing every requested method."""

    completed_frames = []

    for result_path in sorted(
        Path(per_file_dir).glob("*_svtt_results.csv"),
        key=_natural_path_key,
    ):
        try:
            result_df = _read_result_checkpoint(result_path)
        except Exception:
            # A malformed checkpoint is ignored and will be regenerated when
            # its original input file is encountered again.
            continue

        if _result_has_all_methods(result_df, methods, analysis_signature):
            completed_frames.append(result_df)

    if not completed_frames:
        return pd.DataFrame()

    return pd.concat(completed_frames, ignore_index=True)


def _save_live_batch_outputs(
    scanner,
    per_file_dir,
    output_dir,
    methods,
):
    """
    Rebuild and save cumulative batch outputs from completed file checkpoints.

    This is called after every completed input file. Therefore a kernel stop
    loses at most the currently running method, not all previous files.
    """

    all_results = _load_completed_file_results(
        per_file_dir=per_file_dir,
        methods=methods,
        analysis_signature=getattr(scanner, "analysis_signature", None),
    )

    if all_results.empty:
        return {
            "results": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "rank_summary": pd.DataFrame(),
        }

    output_dir = Path(output_dir)

    # Save a continuously updated result file while the batch is running.
    _atomic_write_csv(
        all_results,
        output_dir / "all_completed_files_mc_results.csv",
    )

    summary_input = all_results.copy()
    if "best_cluster" in summary_input.columns:
        summary_input["best_cluster"] = summary_input["best_cluster"].apply(
            _parse_cluster_cell
        )

    summary = scanner.summarize_power_and_metrics(summary_input)
    summary["implementation_version"] = IMPLEMENTATION_VERSION
    summary["expected_count_definition"] = EXPECTED_COUNT_DEFINITION
    summary["ratio_name"] = scanner.ratio_name
    _atomic_write_csv(
        summary,
        output_dir / "power_and_cluster_metrics_summary_live.csv",
    )

    rank_summary = _build_rank_summary(all_results)
    _atomic_write_csv(
        rank_summary,
        output_dir / "rank_specific_summary_live.csv",
    )

    return {
        "results": all_results,
        "summary": summary,
        "rank_summary": rank_summary,
    }


def run_100_saved_files(
    data_dir,
    coord_path,
    output_dir=None,
    true_cluster=None,
    n_files=100,
    n_max=1,
    methods=DEFAULT_METHODS,
    Bmc=199,
    alpha=0.05,
    min_size=1,
    max_size_percent=25.0,
    step=1,
    seed=218720,
    n_jobs=1,
    file_pattern="*.csv",
    resume=True,
    stop_on_error=False,
    ratio_name="SIR",
):
    """
    Run SVTT on saved simulation files with method-level checkpointing.

    Resume behavior
    ---------------
    1. Every method is saved separately for every input CSV.
    2. The complete result for an input CSV is rebuilt after each method.
    3. Cumulative batch results and summaries are updated after each file.
    4. On restart, completed methods, files, and summaries are reused.

    Population is always read from the population column of each county-year
    row. No fixed population value is imposed on saved data. Candidate
    clusters are restricted by ``max_size_percent`` of the total population,
    rather than by a percentage of the number of counties.

    Each dataset uses R0 = sum(observed_cases) / sum(population) across all
    county-year rows, then expected_cases = population * R0. The Poisson
    response is observed_cases and the offset is log(expected_cases).
    The SIR/SMR ratio is saved in pooled_ratio_diagnostics for inspection.

    If output_dir is omitted, results are created under
    C:/Users/ansarims/Desktop/Sem3_Fall2027/Code/Oct_1,
    with input-folder and settings-specific subfolders. Existing outputs
    from the earlier yearly-rate implementation cannot be resumed.
    """

    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one temporal method must be supplied.")

    if int(n_jobs) != 1:
        print(
            "For kernel safety, n_jobs has been changed to 1. "
            "The batch will run sequentially."
        )
    n_jobs = 1

    files = discover_saved_simulation_files(
        data_dir=data_dir,
        n_files=int(n_files),
        pattern=file_pattern,
        require_exact_count=True,
    )

    first_data = _normalise_saved_dataset_columns(
        pd.read_csv(files[0]),
        source_path=files[0],
    )

    study_years = np.sort(first_data["year"].unique().astype(int))
    if len(study_years) < 2:
        raise ValueError("At least two study years are required.")
    if not np.array_equal(
        study_years,
        np.arange(study_years.min(), study_years.max() + 1),
    ):
        raise ValueError(
            "Study years must be consecutive because the current scanner "
            "uses T_start:T_end."
        )

    inferred_truth = infer_true_clusters_from_saved_data(first_data)
    if true_cluster is None:
        if inferred_truth is None:
            raise ValueError(
                "true_cluster was not supplied and could not be inferred. "
                "Add cluster/cluster_id to the CSV files or set TRUE_CLUSTER."
            )
        true_cluster = inferred_truth

    expected_truth = _canonical_clusters(true_cluster)
    expected_counties = np.sort(first_data["county"].unique().astype(int))

    # The same reference population is used to construct one common set of
    # candidate clusters for the entire simulation batch.
    reference_population_data = (
        first_data.loc[
            :, ["county", "year", "population"]
        ]
        .sort_values(["county", "year"])
        .reset_index(drop=True)
    )

    scanner = SVTT_Scan(
        coord_path=str(coord_path),
        T_start=int(study_years.min()),
        T_end=int(study_years.max()),
        Bmc=int(Bmc),
        true_cluster=true_cluster,
        population_data=reference_population_data,
        alpha=float(alpha),
        min_size=int(min_size),
        max_size_percent=float(max_size_percent),
        step=int(step),
        seed=int(seed),
        n_max=int(n_max),
        ratio_name=ratio_name,
    )
    scanner.create_candidate_clusters()

    if not np.array_equal(expected_counties, scanner.counties):
        missing_from_data = sorted(
            set(scanner.counties.tolist()) - set(expected_counties.tolist())
        )
        extra_in_data = sorted(
            set(expected_counties.tolist()) - set(scanner.counties.tolist())
        )
        raise ValueError(
            "The first dataset and coordinate file contain different counties. "
            f"Missing from data: {missing_from_data}; "
            f"extra in data: {extra_in_data}."
        )

    # A different implementation, input, or parameter set needs a fresh
    # analysis folder. In particular, never resume old yearly-rate fits.
    input_hashes = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    analysis_settings = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "expected_count_definition": EXPECTED_COUNT_DEFINITION,
        "ratio_name": scanner.ratio_name,
        "data_dir": str(Path(data_dir).resolve()),
        "input_sha256": input_hashes,
        "coordinate_sha256": hashlib.sha256(Path(coord_path).read_bytes()).hexdigest(),
        "years": [int(year) for year in study_years],
        "true_clusters": [list(cluster) for cluster in expected_truth],
        "Bmc": int(Bmc), "alpha": float(alpha),
        "n_max": int(n_max), "min_size": int(min_size),
        "max_size_percent": float(max_size_percent), "step": int(step),
        "seed": int(seed), "n_files": len(files),
        "methods": [
            {"name": method.name, "basis": method.basis, "p": int(method.p)}
            for method in methods
        ],
    }
    settings_text = json.dumps(analysis_settings, sort_keys=True, indent=2)
    analysis_signature = hashlib.sha256(settings_text.encode("utf-8")).hexdigest()
    scanner.analysis_signature = analysis_signature

    if output_dir is None:
        try:
            input_label = Path(data_dir).resolve().relative_to(DEFAULT_SIM_ROOT.resolve())
        except ValueError:
            input_label = Path(Path(data_dir).name)
        output_dir = DEFAULT_OUTPUT_ROOT / input_label / f"run_{analysis_signature[:10]}"
    output_dir = Path(output_dir)
    settings_path = output_dir / "analysis_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != analysis_settings:
            raise ValueError(
                "Output folder contains different inputs or analysis settings. "
                "Omit output_dir to create a fresh settings-specific folder."
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Output folder already contains results without pooled settings metadata. "
            "Omit output_dir or choose a new empty output folder."
        )

    per_file_dir = output_dir / "per_file_results"
    diagnostic_dir = output_dir / "pooled_ratio_diagnostics"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    settings_tmp = settings_path.with_name(settings_path.name + ".tmp")
    settings_tmp.write_text(settings_text + "\n", encoding="utf-8")
    os.replace(settings_tmp, settings_path)

    manifest_path = output_dir / "file_manifest.csv"
    failed_path = output_dir / "failed_files.csv"
    progress_path = output_dir / "method_progress.csv"

    if resume and manifest_path.exists():
        manifest_rows = pd.read_csv(manifest_path).to_dict("records")
    else:
        manifest_rows = []

    if resume and failed_path.exists():
        failed_rows = pd.read_csv(failed_path).to_dict("records")
    else:
        failed_rows = []

    print("=" * 90)
    print("SAVED-DATA SVTT BATCH WITH METHOD CHECKPOINTS")
    print("=" * 90)
    print(f"Input directory : {Path(data_dir)}")
    print(f"Files selected  : {len(files)}")
    print(f"Study years     : {study_years.min()}--{study_years.max()}")
    print(f"True cluster(s) : {true_cluster}")
    print(f"n_max           : {scanner.nclust_max}")
    print(f"Candidates      : {len(scanner.candidate_clusters)}")
    print(
        "Candidate limit : "
        f"{scanner.max_size_percent:.2f}% of total population"
    )
    print(
        "Largest candidate: "
        f"{scanner.candidate_population_percents.max():.4f}% "
        "of total population"
    )
    print(f"Bmc             : {scanner.Bmc}")
    print(f"Methods         : {[method.name for method in methods]}")
    print("Population      : read from each CSV county-year row")
    print("Expected counts : population_it * pooled_R0 (all counties and years)")
    print(f"Ratio diagnostic: {scanner.ratio_name}; model response = observed counts")
    print(f"Output directory: {output_dir}")
    print("Checkpoint      : after every method and every input file")
    print("=" * 90)

    for simulation_id, file_path in enumerate(files, start=1):
        file_checkpoint_dir = (
            per_file_dir / f"{simulation_id:03d}_{file_path.stem}"
        )
        method_checkpoint_dir = file_checkpoint_dir / "methods"
        file_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        method_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        result_path = (
            per_file_dir
            / f"{simulation_id:03d}_{file_path.stem}_svtt_results.csv"
        )
        status_path = file_checkpoint_dir / "status.csv"

        print("\n" + "-" * 90)
        print(
            f"FILE {simulation_id:03d}/{len(files):03d}: "
            f"{file_path.name}"
        )
        print("-" * 90)

        try:
            data = _normalise_saved_dataset_columns(
                pd.read_csv(file_path),
                source_path=file_path,
            )

            current_years = np.sort(data["year"].unique().astype(int))
            current_counties = np.sort(data["county"].unique().astype(int))

            if not np.array_equal(current_years, study_years):
                raise ValueError(
                    f"Years differ from the first file: {current_years.tolist()}"
                )
            if not np.array_equal(current_counties, expected_counties):
                raise ValueError("County set differs from the first file.")

            current_population_data = (
                data.loc[
                    :, ["county", "year", "population"]
                ]
                .sort_values(["county", "year"])
                .reset_index(drop=True)
            )

            if not np.allclose(
                current_population_data["population"].to_numpy(dtype=float),
                reference_population_data["population"].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-8,
            ):
                raise ValueError(
                    "Population values differ from the first file. "
                    "Population-threshold candidate clusters must use a "
                    "common population table across the simulation batch."
                )

            current_truth = infer_true_clusters_from_saved_data(data)
            if (
                current_truth is not None
                and _canonical_clusters(current_truth) != expected_truth
            ):
                raise ValueError(
                    "The true-cluster definition differs from the first file. "
                    f"Found {current_truth}; expected {true_cluster}."
                )

            data["sim"] = int(simulation_id)
            data = scanner.standardize_saved_data(data)
            print(f"  Pooled R0   : {scanner.pooled_R0:.12g}")
            diagnostic_path = diagnostic_dir / (
                f"{simulation_id:03d}_{file_path.stem}_pooled_{scanner.ratio_name}.csv"
            )
            _atomic_write_csv(data, diagnostic_path)

            # Fast file-level resume: skip a file only when every requested
            # method is present in its aggregate checkpoint.
            if resume and result_path.exists():
                existing_result = _read_result_checkpoint(result_path)
            else:
                existing_result = pd.DataFrame()

            if _result_has_all_methods(existing_result, methods, analysis_signature):
                print("Complete file checkpoint found; all methods skipped.")
                file_result_df = existing_result
            else:
                # Expected counts and the fitted global-null probabilities are
                # prepared once and reused for every temporal method.
                y_matrix, E_matrix = scanner._prepare_scan_arrays(data)
                global_null_fit = (
                    scanner._fit_global_null_probabilities(
                        y_matrix=y_matrix,
                        E_matrix=E_matrix,
                    )
                )
                null_probabilities = global_null_fit["probabilities"]

                print(
                    "  Global null : fitted county intercepts; "
                    f"iterations={global_null_fit['iterations']}, "
                    "max county-margin error="
                    f"{global_null_fit['max_county_margin_error']:.3e}"
                )

                method_frames = []
                progress_rows = []

                for method_index, method in enumerate(methods):
                    method_result_path = (
                        method_checkpoint_dir / f"{method.name}_results.csv"
                    )

                    method_result = (
                        _read_result_checkpoint(method_result_path)
                        if resume and method_result_path.exists()
                        else pd.DataFrame()
                    )
                    if _result_has_all_methods(method_result, (method,), analysis_signature):
                        print(
                            f"  {method.name:<10}: checkpoint found; skipped"
                        )
                    else:
                        print(f"  {method.name:<10}: running")

                        method_seed = (
                            int(seed)
                            + int(simulation_id) * 1_000_000
                            + method_index * 10_000
                        )

                        calibration = scanner._calibrate_mc_cutoff(
                            y_matrix=y_matrix,
                            E_matrix=E_matrix,
                            null_probabilities=null_probabilities,
                            method=method,
                            seed=method_seed,
                            n_jobs=n_jobs,
                        )

                        method_result = scanner.monte_carlo_pvalue_one_sim(
                            df_sim=data,
                            method=method,
                            calibration=calibration,
                        )

                        method_result.insert(0, "sim", int(simulation_id))
                        method_result.insert(1, "file_name", file_path.name)
                        method_result.insert(2, "file_path", str(file_path))
                        method_result["implementation_version"] = IMPLEMENTATION_VERSION
                        method_result["expected_count_definition"] = EXPECTED_COUNT_DEFINITION
                        method_result["analysis_signature"] = analysis_signature
                        method_result["input_sha256"] = input_hashes[str(file_path.resolve())]
                        method_result["ratio_name"] = scanner.ratio_name
                        method_result["pooled_R0"] = scanner.pooled_R0

                        _atomic_write_csv(
                            method_result,
                            method_result_path,
                        )
                        print(
                            f"  {method.name:<10}: checkpoint saved"
                        )

                        del calibration

                    # Normalize identifying columns for old/resumed checkpoints.
                    if "sim" not in method_result.columns:
                        method_result.insert(0, "sim", int(simulation_id))
                    else:
                        method_result["sim"] = int(simulation_id)
                    if "file_name" not in method_result.columns:
                        method_result.insert(1, "file_name", file_path.name)
                    if "file_path" not in method_result.columns:
                        method_result.insert(2, "file_path", str(file_path))

                    method_frames.append(method_result)
                    progress_rows.append({
                        "sim": int(simulation_id),
                        "file_name": file_path.name,
                        "method": method.name,
                        "status": "completed",
                        "checkpoint": str(method_result_path),
                    })

                    # Rebuild the file-level checkpoint after every method.
                    file_result_df = pd.concat(
                        method_frames,
                        ignore_index=True,
                    )
                    _atomic_write_csv(file_result_df, result_path)
                    _atomic_write_csv(pd.DataFrame(progress_rows), status_path)

                    # Global progress is also saved after every method.
                    global_progress = []
                    for status_file in sorted(
                        per_file_dir.glob("*/status.csv"),
                        key=_natural_path_key,
                    ):
                        try:
                            global_progress.append(pd.read_csv(status_file))
                        except Exception:
                            continue
                    if global_progress:
                        _atomic_write_csv(
                            pd.concat(global_progress, ignore_index=True),
                            progress_path,
                        )

                    scanner._null_stats_cache.clear()
                    gc.collect()

            if not _result_has_all_methods(file_result_df, methods, analysis_signature):
                completed_names = sorted(
                    set(file_result_df.get("method", pd.Series(dtype=str)))
                )
                raise RuntimeError(
                    "The file checkpoint is incomplete after processing. "
                    f"Completed methods: {completed_names}."
                )

            # Replace any earlier manifest row for this simulation.
            manifest_rows = [
                row for row in manifest_rows
                if int(row.get("sim", -1)) != int(simulation_id)
            ]
            manifest_rows.append({
                "sim": int(simulation_id),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "n_rows": int(len(data)),
                "n_counties": int(data["county"].nunique()),
                "n_years": int(data["year"].nunique()),
                "population_min": float(data["population"].min()),
                "population_max": float(data["population"].max()),
                "population_mean": float(data["population"].mean()),
                "population_varies": bool(data["population"].nunique() > 1),
                "pooled_R0": scanner.pooled_R0,
                "total_observed_cases": scanner.total_observed_cases,
                "total_person_time": scanner.total_person_time,
                "ratio_name": scanner.ratio_name,
                "expected_count_definition": EXPECTED_COUNT_DEFINITION,
                "implementation_version": IMPLEMENTATION_VERSION,
                "diagnostic_path": str(diagnostic_path),
                "methods_completed": len(methods),
                "status": "completed",
                "result_path": str(result_path),
            })
            _atomic_write_csv(pd.DataFrame(manifest_rows), manifest_path)

            # Remove an older failure entry when this file later succeeds.
            failed_rows = [
                row for row in failed_rows
                if int(row.get("sim", -1)) != int(simulation_id)
            ]
            if failed_rows:
                _atomic_write_csv(pd.DataFrame(failed_rows), failed_path)
            elif failed_path.exists():
                failed_path.unlink()

            live = _save_live_batch_outputs(
                scanner=scanner,
                per_file_dir=per_file_dir,
                output_dir=output_dir,
                methods=methods,
            )

            completed_file_count = (
                int(live["results"]["sim"].nunique())
                if not live["results"].empty
                else 0
            )
            print(
                "File complete. Cumulative completed files: "
                f"{completed_file_count}/{len(files)}"
            )
            print(
                "Population range: "
                f"{data['population'].min():,.2f}--"
                f"{data['population'].max():,.2f}"
            )

        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            print(f"FAILED: {error_text}")

            failed_rows = [
                row for row in failed_rows
                if int(row.get("sim", -1)) != int(simulation_id)
            ]
            failed_rows.append({
                "sim": int(simulation_id),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "error": error_text,
                "traceback": traceback.format_exc(),
            })
            _atomic_write_csv(pd.DataFrame(failed_rows), failed_path)

            manifest_rows = [
                row for row in manifest_rows
                if int(row.get("sim", -1)) != int(simulation_id)
            ]
            manifest_rows.append({
                "sim": int(simulation_id),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "status": "failed",
                "error": error_text,
                "result_path": str(result_path),
            })
            _atomic_write_csv(pd.DataFrame(manifest_rows), manifest_path)

            if stop_on_error:
                raise

        finally:
            scanner._null_stats_cache.clear()
            gc.collect()

    final_outputs = _save_live_batch_outputs(
        scanner=scanner,
        per_file_dir=per_file_dir,
        output_dir=output_dir,
        methods=methods,
    )

    all_results = final_outputs["results"]
    summary = final_outputs["summary"]
    rank_summary = final_outputs["rank_summary"]

    if all_results.empty:
        raise RuntimeError(
            "No file completed successfully. See failed_files.csv."
        )

    _atomic_write_csv(
        all_results,
        output_dir / "all_100_files_mc_results.csv",
    )
    _atomic_write_csv(
        summary,
        output_dir / "power_and_cluster_metrics_summary.csv",
    )
    _atomic_write_csv(
        rank_summary,
        output_dir / "rank_specific_summary.csv",
    )

    batch_complete = all_results["sim"].nunique() == len(files)
    if batch_complete:
        marker_path = output_dir / "_BATCH_COMPLETE.txt"
        marker_path.write_text(
            "All requested files and methods completed successfully.\n",
            encoding="utf-8",
        )

    print("\n" + "=" * 90)
    print("BATCH FINISHED")
    print("=" * 90)
    print(f"Successful files: {all_results['sim'].nunique()}")
    print(f"Requested files : {len(files)}")
    print(f"Failed files    : {len(failed_rows)}")
    print(f"Output directory: {output_dir}")
    print("\nPower and cluster-recovery summary:")
    print(summary.to_string(index=False))

    return {
        "scanner": scanner,
        "files": files,
        "results": all_results,
        "summary": summary,
        "rank_summary": rank_summary,
        "manifest": pd.DataFrame(manifest_rows),
        "failed": pd.DataFrame(failed_rows),
        "output_dir": output_dir,
        "batch_complete": batch_complete,
    }


# ======================================================================

# Jupyter usage:
# %run "C:/Users/ansarims/Desktop/Sem3_Fall2027/Code/Oct_1/svtt_pooled_r0_oct1.py"
# In run_100_saved_files(...), leave output_dir=None to use Oct_1.
# Each dataset is fitted using its own pooled R0 across ALL counties and years.
