
### Data Generation
### Start Date : 30 Sep, 2026
### 



# Generate 100 datasets for each of 18 single-spatial-cluster settings.
# Y_it ~ Poisson(m_i * theta_it), independently across counties and years.
# Constant: population_i = 10000 and m_i = 100.
# Original: population_i from the mapping file and m_i = 0.01 * population_i.
# Population is fixed over time; nonlinear RR applies inside the true cluster.
# Saved CSV columns are county, year, cases, pop (compatible with prior files).

# Portable SVTT simulation generation and pooled-SIR visualization.
# Save as Code/generate_simulations.R.
# Run from the project root: Rscript Code/generate_simulations.R
# Or in R/RStudio: source("Code/generate_simulations.R")
# For a notebook, start it within this repository.
# Requires ggplot2 only when make_plots is TRUE.
#
# Existing datasets are used by default. To regenerate, set generate_data TRUE.
generate_data <- FALSE
make_plots <- TRUE
project_root_override <- NULL  # Optional repository root for unusual launch setups.

find_project_root <- function() {
  starts <- character()
  if (!is.null(project_root_override)) {
    starts <- normalizePath(project_root_override, winslash = "/", mustWork = TRUE)
  } else {
    # source() supplies the script path in an execution frame.
    for (frame in sys.frames()) {
      if (!is.null(frame$ofile)) {
        starts <- c(starts, dirname(normalizePath(frame$ofile,
                         winslash = "/", mustWork = TRUE)))
      }
    }
    # Rscript supplies --file=...; the script can be launched from elsewhere.
    script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(script_arg)) {
      starts <- c(starts, dirname(normalizePath(sub("^--file=", "", script_arg[1]),
                         winslash = "/", mustWork = TRUE)))
    }
    starts <- c(starts, normalizePath(getwd(), winslash = "/", mustWork = TRUE))
  }
  for (start in unique(starts)) {
    candidate <- start
    repeat {
      if (dir.exists(file.path(candidate, "Code")) &&
          dir.exists(file.path(candidate, "Data"))) return(candidate)
      parent <- dirname(candidate)
      if (identical(parent, candidate)) break
      candidate <- parent
    }
  }
  stop("Cannot locate the repository containing Code/ and Data/. ",
       "Launch from within the repository or set project_root_override.")
}

project_root <- find_project_root()
data_dir <- file.path(project_root, "Data")
mapping_file <- file.path(data_dir, "mapping_population_coordinates_2021.csv")
# Accept saved simulations directly in Data/ or inside Data/Sim_Data/.
if (dir.exists(file.path(data_dir, "Single_Cluster")) ||
    dir.exists(file.path(data_dir, "Two_Cluster"))) {
  Output_folder <- data_dir
} else {
  Output_folder <- file.path(data_dir, "Sim_Data")
}
results_dir <- file.path(project_root, "Results")
cat("Repository:", project_root, "\n")
cat("Simulation data:", Output_folder, "\n")

simulation_years <- 1980:2000
n_simulations <- 100L
base_seed <- 20261001L
# FALSE prevents accidentally replacing previously generated datasets.
overwrite_existing <- FALSE

if (generate_data) {
if (!file.exists(mapping_file)) {
  stop("Mapping file not found: ", mapping_file,
       "\nPlace the mapping CSV inside the repository Data/ folder.")
}

# 2. Read one fixed population per county; identifiers must be 1,...,33.
mapping_df <- read.csv(
  mapping_file,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

original_names <- names(mapping_df)
normalized_names <- tolower(trimws(original_names))
normalized_names <- gsub("[^a-z0-9]+", "_", normalized_names)
normalized_names <- gsub("^_+|_+$", "", normalized_names)

choose_column <- function(label, preferred_names, fallback_pattern) {
  preferred_positions <- match(preferred_names, normalized_names)
  preferred_positions <- preferred_positions[!is.na(preferred_positions)]
  
  if (length(preferred_positions) > 0L) {
    return(original_names[preferred_positions[1L]])
  }
  
  fallback_positions <- grep(fallback_pattern, normalized_names)
  
  if (length(fallback_positions) == 1L) {
    return(original_names[fallback_positions])
  }
  
  if (length(fallback_positions) > 1L) {
    stop(
      "More than one possible ", label, " column was found: ",
      paste(original_names[fallback_positions], collapse = ", ")
    )
  }
  
  stop(
    "Could not identify the ", label, " column. Available columns: ",
    paste(original_names, collapse = ", ")
  )
}

county_column <- choose_column(
  label = "county identifier",
  preferred_names = c("county_id", "county_code", "county"),
  fallback_pattern = "county|fips"
)

population_column <- choose_column(
  label = "2021 population",
  preferred_names = c(
    "population_2021",
    "population2021",
    "pop_2021",
    "pop2021",
    "population",
    "pop"
  ),
  fallback_pattern = "population|(^|_)pop($|_)"
)

if (identical(county_column, population_column)) {
  stop("The county and population columns cannot be the same column.")
}

mapping_df$county <- suppressWarnings(
  as.integer(mapping_df[[county_column]])
)

mapping_df$population_2021 <- suppressWarnings(
  as.numeric(
    gsub(
      ",",
      "",
      trimws(as.character(mapping_df[[population_column]])),
      fixed = TRUE
    )
  )
)

if (anyNA(mapping_df$county)) {
  stop("The county identifier column contains missing or non-integer values.")
}

if (anyDuplicated(mapping_df$county)) {
  stop("The population mapping contains duplicate county identifiers.")
}

county_ids <- sort(unique(mapping_df$county))
n_counties <- length(county_ids)

if (!identical(county_ids, seq_len(n_counties))) {
  stop(
    "County identifiers must be consecutive integers from 1 to ",
    n_counties,
    ". Found: ",
    paste(county_ids, collapse = ", ")
  )
}

if (n_counties != 33L) {
  stop("Expected 33 New Mexico counties but found ", n_counties, ".")
}

original_population <- mapping_df$population_2021[
  match(seq_len(n_counties), mapping_df$county)
]

if (
  any(!is.finite(original_population)) ||
  any(original_population <= 0)
) {
  stop("The original 2021 population vector contains invalid values.")
}

# 3. Expected counts and true cluster memberships.
population_settings <- list(
  Constant = list(
    population = rep(10000, n_counties),
    expected_counts = rep(100, n_counties)
  ),
  Original_Population = list(
    population = original_population,
    expected_counts = 0.01 * original_population
  )
)
clusters <- list(
  Single_County = c(15L),
  Double_County = c(15L, 31L),
  Triple_County = c(3L, 15L, 31L)
)

} # End generation-only population setup.

# 4. Residual scenarios: log RR = 0.5 * standardized residual inside,
# RR = 1 outside; intercept, linear and quadratic components removed.
build_residual_components <- function(scenario) {
  if (
    length(scenario) != 1L ||
    is.na(scenario) ||
    !(scenario %in% 1:3)
  ) {
    stop("scenario must be an integer from 1 to 3.")
  }
  
  reference_t <- 1:21
  t_min <- min(reference_t)
  t_max <- max(reference_t)
  t_range <- t_max - t_min
  centre <- (t_min + t_max) / 2
  
  if (scenario == 1L) {
    steepness <- 0.55
    logistic_min <- 1 / (1 + exp(-steepness * (t_min - centre)))
    logistic_max <- 1 / (1 + exp(-steepness * (t_max - centre)))
    logistic_range <- logistic_max - logistic_min
    
    original_fun <- function(t) {
      y <- 1 / (1 + exp(-steepness * (as.numeric(t) - centre)))
      (y - logistic_min) / logistic_range
    }
    
    scenario_name <- "Logistic residual"
  } else if (scenario == 2L) {
    original_fun <- function(t) {
      scaled_time <- 2 * pi * (as.numeric(t) - t_min) / t_range
      sin(scaled_time)
    }
    
    scenario_name <- "Sinusoidal residual"
  } else {
    width <- 0.06 * t_range
    
    original_fun <- function(t) {
      exp(-0.5 * ((as.numeric(t) - centre) / width)^2)
    }
    
    scenario_name <- "Gaussian-bump residual"
  }
  
  scale_time <- function(t) {
    (as.numeric(t) - t_min) / t_range
  }
  
  reference_u <- scale_time(reference_t)
  y_original <- original_fun(reference_t)
  
  X_reference <- cbind(
    intercept = 1,
    linear = reference_u,
    quadratic = reference_u^2
  )
  
  beta_hat <- qr.solve(X_reference, y_original)
  fitted_reference <- as.vector(X_reference %*% beta_hat)
  residual_reference <- y_original - fitted_reference
  residual_scale <- max(abs(residual_reference))
  
  if (
    !is.finite(residual_scale) ||
    residual_scale <= .Machine$double.eps
  ) {
    stop("The nonlinear residual is identically zero.")
  }
  
  standardized_residual_fun <- function(t) {
    t <- as.numeric(t)
    u <- scale_time(t)
    
    X_new <- cbind(
      intercept = 1,
      linear = u,
      quadratic = u^2
    )
    
    residual <- original_fun(t) - as.vector(X_new %*% beta_hat)
    standardized_residual <- residual / residual_scale
    
    if (any(!is.finite(standardized_residual))) {
      stop("The standardized residual contains invalid values.")
    }
    
    standardized_residual
  }
  
  list(
    scenario_name = scenario_name,
    reference_t = reference_t,
    X_reference = X_reference,
    standardized_residual_fun = standardized_residual_fun
  )
}

# Non-slope setting:
# log RR_inside(t) = 0.5 r(t), log RR_outside(t) = 0.
get_residual_scenario <- function(scenario) {
  components <- build_residual_components(scenario)
  
  inside_fun <- function(t) {
    exp(0.5 * components$standardized_residual_fun(t))
  }
  
  outside_fun <- function(t) {
    rep(1, length(as.numeric(t)))
  }
  
  list(
    scenario = scenario,
    scenario_name = components$scenario_name,
    inside_fun = inside_fun,
    outside_fun = outside_fun
  )
}

# 5. Poisson dataset generator.
generate_single_cluster_data <- function(
    n_counties = 33L,
    cluster,
    inside_fun,
    outside_fun,
    years = 1980:2000,
    pop = 10000,
    E_it = 100,
    seed = NULL,
    keep_parameters = FALSE
) {
  if (length(n_counties) != 1L || !is.finite(n_counties) ||
      n_counties < 1 || n_counties != as.integer(n_counties)) {
    stop("n_counties must be a positive integer.")
  }
  n_counties <- as.integer(n_counties)
  counties <- seq_len(n_counties)
  if (!all(cluster %in% counties)) {
    stop("Invalid cluster counties: ",
         paste(setdiff(cluster, counties), collapse = ", "))
  }
  if (!is.function(inside_fun) || !is.function(outside_fun)) {
    stop("inside_fun and outside_fun must be functions.")
  }
  if (length(years) < 1L || anyNA(years) || anyDuplicated(years)) {
    stop("years must contain unique, non-missing values.")
  }
  if (!(length(E_it) %in% c(1L, n_counties)) ||
      any(!is.finite(E_it)) || any(E_it <= 0)) {
    stop("E_it must contain one positive value or one per county.")
  }
  if (!(length(pop) %in% c(1L, n_counties)) ||
      any(!is.finite(pop)) || any(pop <= 0)) {
    stop("pop must contain one positive value or one per county.")
  }
  if (!is.null(seed)) set.seed(seed)
  n_years <- length(years)
  time_points <- seq_along(years)
  data <- expand.grid(county = counties, year = years,
                      KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  data$t <- match(data$year, years)
  inside_values <- inside_fun(time_points)
  outside_values <- outside_fun(time_points)
  if (length(inside_values) == 1L) inside_values <- rep(inside_values, n_years)
  if (length(outside_values) == 1L) outside_values <- rep(outside_values, n_years)
  if (length(inside_values) != n_years || length(outside_values) != n_years ||
      any(!is.finite(inside_values)) || any(inside_values <= 0) ||
      any(!is.finite(outside_values)) || any(outside_values <= 0)) {
    stop("Risk functions must return one positive finite RR or one per year.")
  }
  data$theta <- ifelse(data$county %in% cluster,
                       inside_values[data$t], outside_values[data$t])
  population_values <- rep(pop, length.out = n_counties)
  expected_values <- rep(E_it, length.out = n_counties)
  data$pop <- population_values[data$county]
  data$lambda <- expected_values[data$county] * data$theta
  if (any(!is.finite(data$lambda))) stop("Non-finite Poisson means.")
  data$cases <- rpois(nrow(data), lambda = data$lambda)
  columns <- c("county", "year", "cases", "pop")
  if (keep_parameters) columns <- c(columns, "t", "theta", "lambda")
  data <- data[order(data$county, data$year), columns]
  rownames(data) <- NULL
  data
}

# 6. Generate 100 CSV files in every leaf folder.
# Sim_Data/Single_Cluster/{Single_County,Double_County,Triple_County}/
#         {Constant,Original_Population}/{Res1,Res2,Res3}/sim_001.csv...

if (generate_data) {
# Preflight before writing any datasets.
for (cluster_name in names(clusters)) {
  for (population_name in names(population_settings)) {
    for (scenario in 1:3) {
      folder <- file.path(Output_folder, "Single_Cluster", cluster_name,
                          population_name, paste0("Res", scenario))
      targets <- file.path(folder, sprintf("sim_%03d.csv", seq_len(n_simulations)))
      if (!overwrite_existing && any(file.exists(targets))) {
        stop("Existing simulation files found in: ", folder,
             "\nUse a new output folder or set overwrite_existing <- TRUE.")
      }
    }
  }
}

manifest <- list()
folder_index <- 0L
for (cluster_index in seq_along(clusters)) {
  cluster_name <- names(clusters)[cluster_index]
  cluster <- clusters[[cluster_index]]
  for (population_index in seq_along(population_settings)) {
    population_name <- names(population_settings)[population_index]
    setting <- population_settings[[population_index]]
    for (scenario in 1:3) {
      residual <- get_residual_scenario(scenario)
      folder <- file.path(Output_folder, "Single_Cluster", cluster_name,
                          population_name, paste0("Res", scenario))
      dir.create(folder, recursive = TRUE, showWarnings = FALSE)
      if (!dir.exists(folder)) stop("Could not create folder: ", folder)
      folder_index <- folder_index + 1L
      for (replicate in seq_len(n_simulations)) {
        # Distinct, reproducible seed for every dataset.
        seed <- base_seed + (folder_index - 1L) * n_simulations + replicate
        simulated <- generate_single_cluster_data(
          n_counties = n_counties,
          cluster = cluster,
          inside_fun = residual$inside_fun,
          outside_fun = residual$outside_fun,
          years = simulation_years,
          pop = setting$population,
          E_it = setting$expected_counts,
          seed = seed,
          keep_parameters = FALSE
        )
        destination <- file.path(folder, sprintf("sim_%03d.csv", replicate))
        write.csv(simulated, destination, row.names = FALSE)
        manifest[[length(manifest) + 1L]] <- data.frame(
          configuration = cluster_name,
          population_setting = population_name,
          residual_scenario = scenario,
          true_cluster = paste(cluster, collapse = ","),
          replicate = replicate,
          seed = seed,
          n_rows = nrow(simulated),
          file = substring(destination, nchar(project_root) + 2L),
          stringsAsFactors = FALSE
        )
      }
      cat("Saved", n_simulations, "datasets:", folder, "\n")
    }
  }
}

write.csv(do.call(rbind, manifest),
          file.path(Output_folder, "simulation_manifest.csv"), row.names = FALSE)
saveRDS(list(
  mapping_file = file.path("Data", basename(mapping_file)),
  county_ids = county_ids,
  population_settings = population_settings,
  clusters = clusters,
  years = simulation_years,
  n_simulations = n_simulations,
  base_seed = base_seed,
  effect_size = 0.5,
  model = "Y_it ~ Poisson(m_i * theta_it)",
  R_version = R.version.string,
  RNG = RNGkind()
), file.path(Output_folder, "simulation_settings.rds"))

cat("\nCompleted:", folder_index, "folders;",
    length(manifest), "simulation CSV files.\n")
cat("Rows per dataset:", n_counties * length(simulation_years), "\n")
cat("Output:", Output_folder, "\n")




} # End single-cluster generation.

## Double cluster 




C1 <- c(3L, 15L, 31L)
C2 <- c(12L, 21L, 32L)

# Scenario 1: previous Residual 1 and Residual 3
res1 <- get_residual_scenario(1)
res3 <- get_residual_scenario(3)

# Scenario 2: curve from the image, with quadratic component removed
get_opposite_curves <- function() {
  
  t <- 1:21
  u <- (t - 1) / 20
  
  f <- function(u) {
    plogis((u - 0.30) / 0.08) -
      plogis((u - 0.65) / 0.06)
  }
  
  X <- cbind(1, u, u^2)
  beta <- qr.solve(X, f(u))
  
  residual <- f(u) - as.vector(X %*% beta)
  residual_scale <- max(abs(residual))
  
  if (!is.finite(residual_scale) ||
      residual_scale <= .Machine$double.eps) {
    stop("The residual is identically zero.")
  }
  
  r_star <- function(t) {
    u_new <- (as.numeric(t) - 1) / 20
    X_new <- cbind(1, u_new, u_new^2)
    
    (f(u_new) - as.vector(X_new %*% beta)) / residual_scale
  }
  
  list(
    cluster1_fun = function(t) exp(0.5 * r_star(t)),
    cluster2_fun = function(t) exp(-0.5 * r_star(t))
  )
}

two_scenarios <- list(
  Scenario_1 = list(
    cluster1_fun = res1$inside_fun,
    cluster2_fun = res3$inside_fun
  ),
  Scenario_2 = get_opposite_curves()
)

# Generate independent Poisson counts
generate_two_cluster_data <- function(
    cluster1_fun,
    cluster2_fun,
    pop,
    E_it,
    seed,
    years = 1980:2000
) {
  
  n_counties <- length(pop)
  
  if (n_counties != 33L ||
      length(E_it) != n_counties ||
      any(!is.finite(pop)) ||
      any(pop <= 0) ||
      any(!is.finite(E_it)) ||
      any(E_it <= 0)) {
    stop("Provide 33 positive populations and expected counts.")
  }
  
  rr1 <- cluster1_fun(seq_along(years))
  rr2 <- cluster2_fun(seq_along(years))
  
  if (length(rr1) != length(years) ||
      length(rr2) != length(years) ||
      any(!is.finite(c(rr1, rr2))) ||
      any(c(rr1, rr2) <= 0)) {
    stop("Invalid relative-risk curves.")
  }
  
  data <- expand.grid(
    county = seq_len(n_counties),
    year = years,
    KEEP.OUT.ATTRS = FALSE
  )
  
  time_index <- match(data$year, years)
  
  # Outside both clusters, RR = 1
  theta <- rep(1, nrow(data))
  
  in1 <- data$county %in% C1
  in2 <- data$county %in% C2
  
  theta[in1] <- rr1[time_index[in1]]
  theta[in2] <- rr2[time_index[in2]]
  
  data$pop <- pop[data$county]
  lambda <- E_it[data$county] * theta
  
  set.seed(seed)
  data$cases <- rpois(nrow(data), lambda = lambda)
  
  data <- data[
    order(data$county, data$year),
    c("county", "year", "cases", "pop")
  ]
  
  rownames(data) <- NULL
  data
}

# Uses the existing population_settings:
# Constant: population = 10000, expected_counts = 100
# Original_Population: expected_counts = 0.01 * population

# Save n_simulations datasets per combination.
if (generate_data) {
# Check every two-cluster destination before writing any of them.
for (scenario_name in names(two_scenarios)) {
  for (population_name in names(population_settings)) {
    folder <- file.path(Output_folder, "Two_Cluster", scenario_name, population_name)
    targets <- file.path(folder, sprintf("sim_%03d.csv", seq_len(n_simulations)))
    if (!overwrite_existing && any(file.exists(targets))) {
      stop("Existing simulation files found in: ", folder,
           "\nUse a new output folder or set overwrite_existing <- TRUE.")
    }
  }
}
for (scenario_index in seq_along(two_scenarios)) {
  
  scenario_name <- names(two_scenarios)[scenario_index]
  scenario <- two_scenarios[[scenario_index]]
  
  for (population_index in seq_along(population_settings)) {
    
    population_name <- names(population_settings)[population_index]
    setting <- population_settings[[population_index]]
    
    folder <- file.path(
      Output_folder,
      "Two_Cluster",
      scenario_name,
      population_name
    )
    
    dir.create(folder, recursive = TRUE, showWarnings = FALSE)
    
    if (!dir.exists(folder)) {
      stop("Could not create folder: ", folder)
    }
    
    for (b in seq_len(n_simulations)) {
      
      seed <- 20271001L +
        (scenario_index - 1L) * 2L * n_simulations +
        (population_index - 1L) * n_simulations + b
      
      simulated <- generate_two_cluster_data(
        cluster1_fun = scenario$cluster1_fun,
        cluster2_fun = scenario$cluster2_fun,
        pop = setting$population,
        E_it = setting$expected_counts,
        seed = seed,
        years = simulation_years
      )
      
      write.csv(
        simulated,
        file.path(folder, sprintf("sim_%03d.csv", b)),
        row.names = FALSE
      )
    }
    
    cat("Saved", n_simulations, "datasets:", folder, "\n")
  }
}







#------------------------------------------------------


} # End two-cluster generation.

if (make_plots) {
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Plotting requires ggplot2; install it before running.")
}
library(ggplot2)

replicates_to_plot <- 1L  # Change to 1:100 for mean SIR

sir_plot_dir <- file.path(results_dir, "SIR_Plots")
dir.create(sir_plot_dir, recursive = TRUE, showWarnings = FALSE)

# Load datasets and calculate SIR using one overall rate per dataset
load_sir <- function(
    folder,
    panel,
    cluster1,
    cluster2 = integer(0)
) {
  
  files <- file.path(
    folder,
    sprintf("sim_%03d.csv", replicates_to_plot)
  )
  
  if (any(!file.exists(files))) {
    stop(
      "Missing simulation files:\n",
      paste(files[!file.exists(files)], collapse = "\n")
    )
  }
  
  simulations <- lapply(files, function(file) {
    
    d <- read.csv(file)
    
    required <- c("county", "year", "cases", "pop")
    
    if (!all(required %in% names(d))) {
      stop("Missing required columns in: ", file)
    }
    
    d <- d[required]
    
    if (any(!vapply(d, is.numeric, logical(1))) ||
        any(!is.finite(as.matrix(d))) ||
        any(d$cases < 0) ||
        any(d$pop <= 0)) {
      stop("Invalid data in: ", file)
    }
    
    if (nrow(d) != 33L * 21L ||
        !setequal(d$county, 1:33) ||
        !setequal(d$year, 1980:2000) ||
        anyDuplicated(d[c("county", "year")])) {
      stop("Expected 33 counties × 21 unique years in: ", file)
    }
    
    # One rate pooled over ALL counties and ALL years.
    # Repeated annual populations contribute person-years.
    overall_rate <- sum(d$cases) / sum(d$pop)
    
    if (overall_rate <= 0) {
      stop("Cannot calculate SIR when all counts are zero.")
    }
    
    d$expected <- d$pop * overall_rate
    d$SIR <- d$cases / d$expected
    d$overall_rate <- overall_rate
    
    d$replicate <- as.integer(
      sub("sim_([0-9]+)\\.csv$", "\\1", basename(file))
    )
    
    d
  })
  
  raw <- do.call(rbind, simulations)
  
  # One replicate: observed SIR.
  # Multiple replicates: mean of separately calculated SIRs.
  curves <- aggregate(
    SIR ~ county + year,
    data = raw,
    FUN = mean
  )
  
  curves$panel <- panel
  curves$group <- "Other counties"
  curves$group[curves$county %in% cluster1] <- "Cluster 1"
  curves$group[curves$county %in% cluster2] <- "Cluster 2"
  
  write.csv(
    raw,
    file.path(sir_plot_dir, paste0(gsub("[/\\\\]", "_",
      substring(folder, nchar(Output_folder) + 2L)), "_SIR_pooled_rate.csv")),
    row.names = FALSE
  )
  
  curves
}

# Plot all county trajectories and highlight true cluster counties
draw_sir <- function(d, title, filename) {
  
  y_label <- if (length(replicates_to_plot) == 1L) {
    "SIR"
  } else {
    "Mean SIR"
  }
  
  p <- ggplot(
    d,
    aes(year, SIR, group = interaction(panel, county))
  ) +
    geom_hline(
      yintercept = 1,
      linetype = "dashed",
      colour = "#626C75"
    ) +
    geom_line(
      data = d[d$group == "Other counties", ],
      colour = "#B0B7BF",
      linewidth = 0.45,
      alpha = 0.7
    ) +
    geom_line(
      data = d[d$group != "Other counties", ],
      aes(colour = group),
      linewidth = 1.05
    ) +
    facet_wrap(~panel, nrow = 1) +
    scale_colour_manual(
      values = c(
        "Cluster 1" = "#007A78",
        "Cluster 2" = "#D85B45"
      )
    ) +
    scale_x_continuous(breaks = seq(1980, 2000, 5)) +
    labs(
      title = title,
      x = "Year",
      y = y_label,
      colour = NULL
    ) +
    theme_minimal(base_size = 13) +
    theme(
      panel.background = element_rect(
        fill = "#F3F6F8",
        colour = NA
      ),
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
  
  print(p)
  
  for (extension in c("png", "pdf")) {
    ggsave(
      file.path(
        sir_plot_dir,
        paste0(filename, ".", extension)
      ),
      plot = p,
      width = 5 * length(unique(d$panel)),
      height = 4.5,
      dpi = 300
    )
  }
  
  p
}

# Single-cluster configurations
single_clusters <- list(
  Single_County = c(15L),
  Double_County = c(15L, 31L),
  Triple_County = c(3L, 15L, 31L)
)

population_folders <- c("Constant", "Original_Population")
sir_plots <- list()

# Six figures: three residual panels in each figure
for (population_name in population_folders) {
  for (configuration in names(single_clusters)) {
    
    d <- do.call(rbind, lapply(1:3, function(residual_number) {
      
      load_sir(
        folder = file.path(
          Output_folder,
          "Single_Cluster",
          configuration,
          population_name,
          paste0("Res", residual_number)
        ),
        panel = paste("Residual", residual_number),
        cluster1 = single_clusters[[configuration]]
      )
    }))
    
    key <- paste(configuration, population_name, sep = "_")
    
    sir_plots[[key]] <- draw_sir(
      d,
      title = gsub("_", " ", key),
      filename = key
    )
  }
}

# Two additional figures: two-cluster scenarios
for (population_name in population_folders) {
  
  d <- do.call(rbind, lapply(1:2, function(scenario_number) {
    
    load_sir(
      folder = file.path(
        Output_folder,
        "Two_Cluster",
        paste0("Scenario_", scenario_number),
        population_name
      ),
      panel = if (scenario_number == 1L) {
        "Scenario 1: Res1 and Res3"
      } else {
        "Scenario 2: opposite curves"
      },
      cluster1 = c(3L, 15L, 31L),
      cluster2 = c(12L, 21L, 32L)
    )
  }))
  
  key <- paste0("Two_Cluster_", population_name)
  
  sir_plots[[key]] <- draw_sir(
    d,
    title = gsub("_", " ", key),
    filename = key
  )
}

cat("Saved 8 figures as PNG and PDF in:", sir_plot_dir, "\n")











} # End optional plotting.
