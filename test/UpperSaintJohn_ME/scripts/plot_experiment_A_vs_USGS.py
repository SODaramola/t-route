# ============================================================
# EXPERIMENT A
# STANDARD MUSKINGUM-CUNGE BASELINE
#
# Compare:
#   1. USGS observed streamflow
#   2. NWM v3 retrospective streamflow
#   3. t-route standard Muskingum-Cunge streamflow
#
# Stations:
#   Dickey    : USGS 01010500 / NWM feature 4288603
#   Fort Kent : USGS 01014000 / NWM feature 4287759
#
# Period:
#   2016-11-01 00:00
#       through
#   2016-11-03 00:00
#
# Hydrographs retain native temporal resolution:
#   USGS    = 15 min
#   NWM     = 1 hour
#   t-route = 5 min
#
# Statistics are calculated at common hourly timestamps:
#   2016-11-01 01:00 through 2016-11-03 00:00
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OBS_FILE = (
    BASE_DIR
    / "obs"
    / "USGS_Dickey_FortKent_20161101_20161103.csv"
)

NWM_FILE = (
    BASE_DIR
    / "forcing"
    / "NWM_v3_Dickey_FortKent_20161101_20161103.csv"
)

BASELINE_FILE = (
    BASE_DIR
    / "output"
    / "baseline_MC_vs_NWM_v3_hourly.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. STATION INFORMATION
# ============================================================

STATIONS = {
    "Dickey": {
        "site_no": "01010500",
        "feature_id": 4288603,
        "label": "Dickey, ME",
    },

    "Fort_Kent": {
        "site_no": "01014000",
        "feature_id": 4287759,
        "label": "Fort Kent, ME",
    },
}


# ============================================================
# 3. PLOT SETTINGS
# ============================================================

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    }
)


# ============================================================
# 4. READ USGS OBSERVATIONS
# ============================================================

print()
print("=" * 80)
print("READING USGS OBSERVATIONS")
print("=" * 80)

usgs = pd.read_csv(
    OBS_FILE,
)

usgs["time"] = pd.to_datetime(
    usgs["time"]
)

usgs["streamflow_cms"] = pd.to_numeric(
    usgs["streamflow_cms"],
    errors="coerce",
)

print(
    "USGS rows:",
    len(usgs),
)


# ============================================================
# 5. READ NWM V3
# ============================================================

print()
print("=" * 80)
print("READING NWM V3")
print("=" * 80)

nwm = pd.read_csv(
    NWM_FILE,
)

nwm["time"] = pd.to_datetime(
    nwm["time"]
)

nwm["feature_id"] = pd.to_numeric(
    nwm["feature_id"],
)

nwm["streamflow"] = pd.to_numeric(
    nwm["streamflow"],
    errors="coerce",
)

print(
    "NWM rows:",
    len(nwm),
)


# ============================================================
# 6. LOCATE T-ROUTE PARQUET FILE
# ============================================================

parquet_files = sorted(
    OUTPUT_DIR.glob(
        "flowveldepth_*.parquet"
    ),
    key=lambda p: p.stat().st_mtime,
)

if not parquet_files:

    raise FileNotFoundError(
        "No t-route flowveldepth parquet file found."
    )

PARQUET_FILE = parquet_files[-1]

print()
print("=" * 80)
print("READING T-ROUTE OUTPUT")
print("=" * 80)

print(
    "Parquet:",
    PARQUET_FILE.name,
)


# ============================================================
# 7. READ ONLY REQUIRED T-ROUTE RECORDS
# ============================================================

location_ids = [
    f"nex-{info['feature_id']}"
    for info in STATIONS.values()
]

try:

    troute = pd.read_parquet(
        PARQUET_FILE,
        filters=[
            (
                "variable_name",
                "==",
                "streamflow",
            ),
            (
                "location_id",
                "in",
                location_ids,
            ),
        ],
    )

except Exception as exc:

    print()
    print(
        "Filtered parquet read failed."
    )

    print(
        "Falling back to full parquet read."
    )

    print(
        "Reason:",
        exc,
    )

    troute = pd.read_parquet(
        PARQUET_FILE
    )

    troute = troute[
        (
            troute["variable_name"]
            == "streamflow"
        )
        &
        (
            troute["location_id"]
            .isin(location_ids)
        )
    ].copy()


troute["value_time"] = pd.to_datetime(
    troute["value_time"]
)

troute["value"] = pd.to_numeric(
    troute["value"],
    errors="coerce",
)

print(
    "Selected t-route rows:",
    len(troute),
)


# ============================================================
# 8. MAP T-ROUTE LOCATION IDS
# ============================================================

feature_to_location = {
    info["feature_id"]: location
    for location, info in STATIONS.items()
}

location_id_to_name = {
    f"nex-{feature_id}": location
    for feature_id, location
    in feature_to_location.items()
}

troute["location"] = (
    troute["location_id"]
    .map(location_id_to_name)
)

if troute["location"].isna().any():

    raise RuntimeError(
        "Unable to map one or more t-route "
        "location IDs to stations."
    )


# ============================================================
# 9. DEFINE MODEL PERIOD
# ============================================================

START = pd.Timestamp(
    "2016-11-01 00:00:00"
)

END = pd.Timestamp(
    "2016-11-03 00:00:00"
)


usgs = usgs[
    (usgs["time"] >= START)
    &
    (usgs["time"] <= END)
].copy()


nwm = nwm[
    (nwm["time"] >= START)
    &
    (nwm["time"] <= END)
].copy()


troute = troute[
    (troute["value_time"] >= START)
    &
    (troute["value_time"] <= END)
].copy()


# ============================================================
# 10. VALIDATE NATIVE TIME SERIES
# ============================================================

print()
print("=" * 80)
print("NATIVE TIME-SERIES SUMMARY")
print("=" * 80)

for location in STATIONS:

    obs_sub = usgs[
        usgs["location"] == location
    ]

    nwm_sub = nwm[
        nwm["location"] == location
    ]

    tr_sub = troute[
        troute["location"] == location
    ]

    print()
    print(location)
    print("-" * 50)

    print(
        "USGS records   :",
        len(obs_sub),
    )

    print(
        "NWM records    :",
        len(nwm_sub),
    )

    print(
        "t-route records:",
        len(tr_sub),
    )

    print(
        "USGS range:",
        obs_sub["streamflow_cms"].min(),
        "to",
        obs_sub["streamflow_cms"].max(),
    )

    print(
        "NWM range:",
        nwm_sub["streamflow"].min(),
        "to",
        nwm_sub["streamflow"].max(),
    )

    print(
        "t-route range:",
        tr_sub["value"].min(),
        "to",
        tr_sub["value"].max(),
    )


# ============================================================
# 11. COMMON HOURLY COMPARISON TIMES
# ============================================================
#
# t-route output begins at 00:05.
#
# Therefore:
#   00:00 = initialization
#
# and the routed comparison begins:
#   01:00
#
# through:
#   2016-11-03 00:00
#
# This gives exactly 48 common hourly routed timestamps.
# ============================================================

comparison_times = pd.date_range(
    start="2016-11-01 01:00:00",
    end="2016-11-03 00:00:00",
    freq="1h",
)

print()
print("=" * 80)
print("COMMON COMPARISON TIMES")
print("=" * 80)

print(
    "Number of hourly times:",
    len(comparison_times),
)

print(
    "First:",
    comparison_times[0],
)

print(
    "Last :",
    comparison_times[-1],
)


# ============================================================
# 12. PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    observed,
    simulated,
):

    observed = np.asarray(
        observed,
        dtype=float,
    )

    simulated = np.asarray(
        simulated,
        dtype=float,
    )

    valid = (
        np.isfinite(observed)
        &
        np.isfinite(simulated)
    )

    observed = observed[valid]
    simulated = simulated[valid]

    if len(observed) == 0:

        raise RuntimeError(
            "No valid observations available."
        )

    error = (
        simulated
        - observed
    )

    bias = np.mean(
        error
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    mean_obs = np.mean(
        observed
    )

    if mean_obs != 0:

        relative_bias = (
            100.0
            * bias
            / mean_obs
        )

    else:

        relative_bias = np.nan

    if len(observed) > 1:

        correlation = np.corrcoef(
            observed,
            simulated,
        )[0, 1]

    else:

        correlation = np.nan

    nse_denominator = np.sum(
        (
            observed
            - mean_obs
        ) ** 2
    )

    if nse_denominator > 0:

        nse = (
            1.0
            -
            np.sum(
                (
                    simulated
                    - observed
                ) ** 2
            )
            / nse_denominator
        )

    else:

        nse = np.nan

    return {
        "n": len(observed),
        "mean_observed": mean_obs,
        "mean_simulated":
            np.mean(simulated),
        "bias": bias,
        "relative_bias_percent":
            relative_bias,
        "mae": mae,
        "rmse": rmse,
        "correlation": correlation,
        "nse": nse,
    }


# ============================================================
# 13. BUILD HOURLY COMPARISON DATASET
# ============================================================

comparison_rows = []

metric_rows = []

for location, info in STATIONS.items():

    # --------------------------------------------------------
    # USGS
    # --------------------------------------------------------

    obs_sub = (
        usgs[
            usgs["location"] == location
        ][
            [
                "time",
                "streamflow_cms",
            ]
        ]
        .copy()
        .set_index("time")
    )

    # Because USGS data are at exact 15-minute intervals,
    # every hourly timestamp should already exist.
    obs_hourly = (
        obs_sub
        .reindex(comparison_times)
        .rename(
            columns={
                "streamflow_cms":
                    "usgs_observed"
            }
        )
    )

    # --------------------------------------------------------
    # NWM
    # --------------------------------------------------------

    nwm_sub = (
        nwm[
            nwm["location"] == location
        ][
            [
                "time",
                "streamflow",
            ]
        ]
        .copy()
        .set_index("time")
    )

    nwm_hourly = (
        nwm_sub
        .reindex(comparison_times)
        .rename(
            columns={
                "streamflow":
                    "nwm_v3"
            }
        )
    )

    # --------------------------------------------------------
    # T-ROUTE
    # --------------------------------------------------------

    tr_sub = (
        troute[
            troute["location"] == location
        ][
            [
                "value_time",
                "value",
            ]
        ]
        .copy()
        .set_index("value_time")
    )

    tr_hourly = (
        tr_sub
        .reindex(comparison_times)
        .rename(
            columns={
                "value":
                    "troute_mc"
            }
        )
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    merged = pd.concat(
        [
            obs_hourly,
            nwm_hourly,
            tr_hourly,
        ],
        axis=1,
    )

    merged.index.name = "time"

    merged = (
        merged
        .reset_index()
    )

    merged["location"] = location
    merged["site_no"] = info["site_no"]
    merged["feature_id"] = info["feature_id"]

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print(
        f"Hourly comparison: {location}"
    )

    print(
        "Rows:",
        len(merged),
    )

    print(
        "Missing USGS:",
        merged["usgs_observed"]
        .isna()
        .sum(),
    )

    print(
        "Missing NWM:",
        merged["nwm_v3"]
        .isna()
        .sum(),
    )

    print(
        "Missing t-route:",
        merged["troute_mc"]
        .isna()
        .sum(),
    )

    comparison_rows.append(
        merged
    )

    # --------------------------------------------------------
    # NWM METRICS
    # --------------------------------------------------------

    nwm_metrics = calculate_metrics(
        merged["usgs_observed"],
        merged["nwm_v3"],
    )

    nwm_metrics.update(
        {
            "location": location,
            "site_no": info["site_no"],
            "feature_id":
                info["feature_id"],
            "model": "NWM v3",
        }
    )

    metric_rows.append(
        nwm_metrics
    )

    # --------------------------------------------------------
    # T-ROUTE METRICS
    # --------------------------------------------------------

    troute_metrics = calculate_metrics(
        merged["usgs_observed"],
        merged["troute_mc"],
    )

    troute_metrics.update(
        {
            "location": location,
            "site_no": info["site_no"],
            "feature_id":
                info["feature_id"],
            "model":
                "t-route standard MC",
        }
    )

    metric_rows.append(
        troute_metrics
    )


comparison_df = pd.concat(
    comparison_rows,
    ignore_index=True,
)

metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# 14. REORDER METRIC COLUMNS
# ============================================================

metrics_df = metrics_df[
    [
        "location",
        "site_no",
        "feature_id",
        "model",
        "n",
        "mean_observed",
        "mean_simulated",
        "bias",
        "relative_bias_percent",
        "mae",
        "rmse",
        "correlation",
        "nse",
    ]
]


# ============================================================
# 15. PRINT METRICS
# ============================================================

print()
print("=" * 80)
print("MODEL PERFORMANCE AGAINST USGS OBSERVATIONS")
print("=" * 80)

with pd.option_context(
    "display.max_columns",
    None,
    "display.width",
    200,
):

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.3f}",
        )
    )


# ============================================================
# 16. SAVE COMPARISON TABLES
# ============================================================

COMPARISON_CSV = (
    OUTPUT_DIR
    / "ExperimentA_USGS_NWM_troute_hourly.csv"
)

METRICS_CSV = (
    OUTPUT_DIR
    / "ExperimentA_USGS_NWM_troute_metrics.csv"
)

comparison_df.to_csv(
    COMPARISON_CSV,
    index=False,
)

metrics_df.to_csv(
    METRICS_CSV,
    index=False,
)


# ============================================================
# 17. CREATE TWO-PANEL HYDROGRAPH
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(12, 9),
    sharex=True,
)

panel_labels = {
    "Dickey": "(a)",
    "Fort_Kent": "(b)",
}


for ax, (
    location,
    info,
) in zip(
    axes,
    STATIONS.items(),
):

    # --------------------------------------------------------
    # Native-resolution data
    # --------------------------------------------------------

    obs_sub = (
        usgs[
            usgs["location"] == location
        ]
        .sort_values("time")
    )

    nwm_sub = (
        nwm[
            nwm["location"] == location
        ]
        .sort_values("time")
    )

    tr_sub = (
        troute[
            troute["location"] == location
        ]
        .sort_values("value_time")
    )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    ax.plot(
        obs_sub["time"],
        obs_sub["streamflow_cms"],
        linewidth=2.4,
        label="USGS observed",
        zorder=4,
    )

    ax.plot(
        nwm_sub["time"],
        nwm_sub["streamflow"],
        linewidth=2.0,
        linestyle="--",
        label="NWM v3 retrospective",
        zorder=3,
    )

    ax.plot(
        tr_sub["value_time"],
        tr_sub["value"],
        linewidth=1.8,
        linestyle="-.",
        label="t-route standard MC",
        zorder=2,
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    ax.set_ylabel(
        r"Streamflow ($\mathrm{m^3\,s^{-1}}$)"
    )

    ax.set_title(
        f"{panel_labels[location]} "
        f"{info['label']} "
        f"(USGS {info['site_no']})",
        loc="left",
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.7,
        alpha=0.6,
    )

    ax.set_xlim(
        START,
        END,
    )

    # --------------------------------------------------------
    # METRICS TEXT
    # --------------------------------------------------------

    station_metrics = metrics_df[
        metrics_df["location"]
        == location
    ]

    nwm_row = station_metrics[
        station_metrics["model"]
        == "NWM v3"
    ].iloc[0]

    tr_row = station_metrics[
        station_metrics["model"]
        == "t-route standard MC"
    ].iloc[0]

    metrics_text = (
        "Hourly comparison vs USGS\n"
        f"NWM: RMSE={nwm_row['rmse']:.2f}, "
        f"Bias={nwm_row['bias']:.2f}, "
        f"r={nwm_row['correlation']:.3f}\n"
        f"t-route: RMSE={tr_row['rmse']:.2f}, "
        f"Bias={tr_row['bias']:.2f}, "
        f"r={tr_row['correlation']:.3f}"
    )

    ax.text(
        0.985,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.5",
            "alpha": 0.90,
        },
    )


# ============================================================
# 18. SHARED LEGEND
# ============================================================

handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.985),
    ncol=3,
    frameon=False,
)


# ============================================================
# 19. X AXIS
# ============================================================

axes[-1].set_xlabel(
    "Date and time (UTC)"
)

axes[-1].xaxis.set_major_locator(
    mdates.HourLocator(
        interval=6
    )
)

axes[-1].xaxis.set_major_formatter(
    mdates.DateFormatter(
        "%b %d\n%H:%M"
    )
)


# ============================================================
# 20. FIGURE TITLE
# ============================================================

fig.suptitle(
    "Experiment A: Standard Muskingum–Cunge Baseline "
    "Compared with NWM v3 and USGS Observations",
    fontsize=16,
    fontweight="bold",
    y=1.015,
)


# ============================================================
# 21. LAYOUT
# ============================================================

fig.tight_layout(
    rect=[
        0.03,
        0.03,
        0.98,
        0.94,
    ]
)


# ============================================================
# 22. SAVE FIGURE
# ============================================================

PNG_FILE = (
    OUTPUT_DIR
    / "ExperimentA_USGS_NWM_troute_comparison.png"
)

PDF_FILE = (
    OUTPUT_DIR
    / "ExperimentA_USGS_NWM_troute_comparison.pdf"
)

fig.savefig(
    PNG_FILE,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 23. FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("EXPERIMENT A COMPARISON COMPLETE")
print("=" * 80)

print()
print(
    "Figure PNG:"
)

print(
    PNG_FILE.resolve()
)

print()
print(
    "Figure PDF:"
)

print(
    PDF_FILE.resolve()
)

print()
print(
    "Hourly comparison CSV:"
)

print(
    COMPARISON_CSV.resolve()
)

print()
print(
    "Metrics CSV:"
)

print(
    METRICS_CSV.resolve()
)
