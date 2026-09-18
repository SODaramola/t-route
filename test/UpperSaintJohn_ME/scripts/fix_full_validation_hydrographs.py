from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams, font_manager


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUT_DIR = BASE_DIR / "output" / "troute_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)

USGS_FILE = (
    BASE_DIR
    / "obs"
    / "USGS_Dickey_FortKent_20161101_20170501.csv"
)

NWM_FILE = (
    BASE_DIR
    / "forcing"
    / "NWM_v3_Dickey_FortKent_20161101_20170501.csv"
)

TROUTE_FILE = (
    BASE_DIR
    / "output"
    / "UpperSaintJohn_MC_20161101_20170501_CHANOBS.nc"
)

METRICS_FILE = (
    BASE_DIR
    / "output"
    / "standard_MC_NWM_USGS_metrics_20161101_20170501.csv"
)


# ============================================================
# 2. STATIONS
# ============================================================

STATIONS = {
    "Dickey": 4288603,
    "Fort_Kent": 4287759,
}

DISPLAY_NAMES = {
    "Dickey": "Dickey",
    "Fort_Kent": "Fort Kent",
}


# ============================================================
# 3. FIXED STYLE
# ============================================================

font_manager._load_fontmanager(
    try_read_cache=False
)

rcParams["font.family"] = "Times New Roman"

rcParams["font.size"] = 16
rcParams["axes.titlesize"] = 20
rcParams["axes.labelsize"] = 18
rcParams["xtick.labelsize"] = 15
rcParams["ytick.labelsize"] = 15
rcParams["legend.fontsize"] = 15

rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300


# Same colors everywhere in the project
COLOR_USGS = "#1f77b4"
COLOR_EST = "#cc33cc"
COLOR_NWM = "#ff7f0e"
COLOR_TROUTE = "#2ca02c"


# ============================================================
# 4. READ DATA
# ============================================================

usgs = pd.read_csv(
    USGS_FILE
)

usgs["time"] = pd.to_datetime(
    usgs["time"]
)

usgs["is_estimated"] = (
    usgs["qualifier"]
    .fillna("")
    .astype(str)
    .str.contains(
        "ESTIMATED",
        case=False,
        na=False,
    )
)


nwm = pd.read_csv(
    NWM_FILE
)

nwm["time"] = pd.to_datetime(
    nwm["time"]
)


metrics = pd.read_csv(
    METRICS_FILE
)


ds = xr.open_dataset(
    TROUTE_FILE
)


# ============================================================
# 5. METRIC TEXT
# ============================================================

def metric_text(location):

    m = metrics[
        (metrics["location"] == location)
        &
        (metrics["subset"] == "all_USGS_reported")
    ].copy()

    tr = m[
        m["model"]
        == "t-route_standard_MC"
    ].iloc[0]

    nw = m[
        m["model"]
        == "NWM_v3"
    ].iloc[0]

    text = (
        "Common-hour validation\n\n"

        "t-route (MC) vs USGS\n"
        f"n = {int(tr['n'])}\n"
        f"Bias = {tr['bias_cms']:.1f} m³/s\n"
        f"MAE = {tr['mae_cms']:.1f} m³/s\n"
        f"RMSE = {tr['rmse_cms']:.1f} m³/s\n"
        f"PBIAS = {tr['percent_bias']:.1f}%\n"
        f"r = {tr['correlation']:.3f}\n\n"

        "NWM v3 vs USGS\n"
        f"n = {int(nw['n'])}\n"
        f"Bias = {nw['bias_cms']:.1f} m³/s\n"
        f"MAE = {nw['mae_cms']:.1f} m³/s\n"
        f"RMSE = {nw['rmse_cms']:.1f} m³/s\n"
        f"PBIAS = {nw['percent_bias']:.1f}%\n"
        f"r = {nw['correlation']:.3f}"
    )

    return text


# ============================================================
# 6. PLOT CONTINUOUS USGS OBSERVATION SEGMENTS
# ============================================================

def plot_usgs_observed_segments(
    ax,
    station_df,
):

    station_df = (
        station_df
        .sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    # A regular USGS record here is normally 15 min.
    #
    # Break the line whenever:
    #   1. observation is estimated
    #   2. the time gap exceeds 30 minutes
    #
    # This prevents Matplotlib from connecting observations
    # across missing winter intervals.
    gaps = (
        station_df["time"]
        .diff()
        .gt(pd.Timedelta("30min"))
    )

    breaks = (
        gaps
        |
        station_df["is_estimated"]
    )

    station_df["_segment"] = (
        breaks
        .cumsum()
    )

    observed = station_df[
        ~station_df["is_estimated"]
    ].copy()

    first_label = True

    for _, segment in observed.groupby(
        "_segment"
    ):

        segment = segment.sort_values(
            "time"
        )

        # Do not manufacture a line from a single point.
        if len(segment) >= 2:

            ax.plot(
                segment["time"],
                segment["streamflow_cms"],
                color=COLOR_USGS,
                linewidth=2.4,
                label=(
                    "USGS"
                    if first_label
                    else None
                ),
                zorder=4,
            )

            first_label = False

        elif len(segment) == 1:

            ax.scatter(
                segment["time"],
                segment["streamflow_cms"],
                color=COLOR_USGS,
                s=20,
                label=(
                    "USGS"
                    if first_label
                    else None
                ),
                zorder=4,
            )

            first_label = False


# ============================================================
# 7. CREATE HYDROGRAPHS
# ============================================================

for figure_number, (
    location,
    feature_id,
) in enumerate(
    STATIONS.items(),
    start=2,
):

    u = usgs[
        usgs["location"] == location
    ].copy()

    n = nwm[
        nwm["location"] == location
    ].copy()

    t = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ds["time"].values
            ),

            "streamflow": (
                ds["streamflow"]
                .sel(
                    feature_id=feature_id
                )
                .values
                .astype(float)
            ),
        }
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(18, 8.5)
    )


    # --------------------------------------------------------
    # USGS OBSERVED
    # --------------------------------------------------------

    plot_usgs_observed_segments(
        ax,
        u,
    )


    # --------------------------------------------------------
    # USGS ESTIMATED
    #
    # MARKERS ONLY.
    # NO CONNECTING LINE.
    # --------------------------------------------------------

    estimated = u[
        u["is_estimated"]
    ].copy()

    ax.scatter(
        estimated["time"],
        estimated["streamflow_cms"],
        color=COLOR_EST,
        marker="x",
        s=42,
        linewidths=1.8,
        label="USGS estimated",
        zorder=6,
    )


    # --------------------------------------------------------
    # NWM V3
    # --------------------------------------------------------

    ax.plot(
        n["time"],
        n["streamflow"],
        color=COLOR_NWM,
        linewidth=2.2,
        label="NWM v3",
        zorder=2,
    )


    # --------------------------------------------------------
    # T-ROUTE
    # --------------------------------------------------------

    ax.plot(
        t["time"],
        t["streamflow"],
        color=COLOR_TROUTE,
        linewidth=2.2,
        label="t-route (MC)",
        zorder=3,
    )


    # --------------------------------------------------------
    # TITLE / AXES
    # --------------------------------------------------------

    ax.set_title(
        f"{DISPLAY_NAMES[location]}: "
        "t-route (MC) vs NWM v3 vs USGS\n"
        "November 2016–April 2017"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Streamflow (m³/s)"
    )

    ax.set_xlim(
        pd.Timestamp(
            "2016-11-01"
        ),
        pd.Timestamp(
            "2017-05-01"
        ),
    )


    # --------------------------------------------------------
    # DATE TICKS
    # --------------------------------------------------------

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b\n%Y"
        )
    )


    # --------------------------------------------------------
    # GRID
    # --------------------------------------------------------

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.8,
    )

    ax.set_axisbelow(
        True
    )


    # --------------------------------------------------------
    # METRICS OUTSIDE AXES
    # --------------------------------------------------------

    fig.text(
        0.79,
        0.83,
        metric_text(location),
        ha="left",
        va="top",
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
        ),
    )


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    # Force logical order
    wanted_order = [
        "USGS",
        "USGS estimated",
        "NWM v3",
        "t-route (MC)",
    ]

    ordered_handles = []
    ordered_labels = []

    for wanted in wanted_order:

        if wanted in labels:

            i = labels.index(
                wanted
            )

            ordered_handles.append(
                handles[i]
            )

            ordered_labels.append(
                labels[i]
            )


    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.13,
        ),
        ncol=4,
        frameon=True,
    )


    # Leave actual space for metrics and legend
    fig.subplots_adjust(
        left=0.08,
        right=0.76,
        top=0.86,
        bottom=0.20,
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    outfile = (
        OUT_DIR
        /
        (
            f"{figure_number:02d}_"
            f"{location}_"
            "t-route_MC_vs_NWM_v3_vs_USGS.png"
        )
    )

    fig.savefig(
        outfile,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        outfile,
    )


ds.close()

print()
print("=" * 70)
print("CORRECTED HYDROGRAPHS COMPLETE")
print("=" * 70)
