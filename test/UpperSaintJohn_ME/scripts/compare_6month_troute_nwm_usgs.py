from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

CHANOBS_FILE = (
    BASE_DIR
    / "output"
    / "UpperSaintJohn_MC_20161101_20170501_CHANOBS.nc"
)

NWM_FILE = (
    BASE_DIR
    / "forcing"
    / "NWM_v3_Dickey_FortKent_20161101_20170501.csv"
)

USGS_FILE = (
    BASE_DIR
    / "obs"
    / "USGS_Dickey_FortKent_20161101_20170501.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. STATIONS
# ============================================================

STATIONS = {
    "Dickey": {
        "feature_id": 4288603,
        "usgs": "01010500",
    },
    "Fort_Kent": {
        "feature_id": 4287759,
        "usgs": "01014000",
    },
}


# ============================================================
# 3. READ T-ROUTE CHANOBS
# ============================================================

ds = xr.open_dataset(CHANOBS_FILE)

troute_frames = []

for location, info in STATIONS.items():

    q = (
        ds["streamflow"]
        .sel(feature_id=info["feature_id"])
        .values
        .astype(float)
    )

    troute_frames.append(
        pd.DataFrame(
            {
                "time": pd.to_datetime(ds["time"].values),
                "location": location,
                "troute_streamflow": q,
            }
        )
    )

troute = pd.concat(
    troute_frames,
    ignore_index=True,
)

ds.close()


# ============================================================
# 4. READ NWM
# ============================================================

nwm = pd.read_csv(NWM_FILE)

nwm["time"] = pd.to_datetime(
    nwm["time"]
)

nwm = nwm[
    [
        "time",
        "location",
        "feature_id",
        "streamflow",
    ]
].copy()

nwm = nwm.rename(
    columns={
        "streamflow": "nwm_streamflow"
    }
)


# ============================================================
# 5. READ USGS
# ============================================================

usgs = pd.read_csv(USGS_FILE)

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

usgs = usgs[
    [
        "time",
        "location",
        "site_no",
        "streamflow_cms",
        "is_estimated",
    ]
].copy()

usgs = usgs.rename(
    columns={
        "streamflow_cms": "usgs_streamflow"
    }
)


# ============================================================
# 6. EXACT HOURLY T-ROUTE VALUES
#
# No interpolation.
# T-route is sampled only where an exact top-of-hour
# 5-minute model timestamp exists.
# ============================================================

troute_hourly = troute[
    (troute["time"].dt.minute == 0)
    & (troute["time"].dt.second == 0)
].copy()


# ============================================================
# 7. METRIC FUNCTION
# ============================================================

def calculate_metrics(model, obs):

    model = np.asarray(
        model,
        dtype=float,
    )

    obs = np.asarray(
        obs,
        dtype=float,
    )

    valid = (
        np.isfinite(model)
        & np.isfinite(obs)
    )

    model = model[valid]
    obs = obs[valid]

    n = len(obs)

    if n == 0:
        return {
            "n": 0,
            "bias_cms": np.nan,
            "mae_cms": np.nan,
            "rmse_cms": np.nan,
            "percent_bias": np.nan,
            "correlation": np.nan,
        }

    error = model - obs

    bias = np.mean(error)
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    if np.sum(obs) != 0:
        percent_bias = (
            100.0
            * np.sum(error)
            / np.sum(obs)
        )
    else:
        percent_bias = np.nan

    if n > 1:
        correlation = np.corrcoef(
            model,
            obs,
        )[0, 1]
    else:
        correlation = np.nan

    return {
        "n": n,
        "bias_cms": bias,
        "mae_cms": mae,
        "rmse_cms": rmse,
        "percent_bias": percent_bias,
        "correlation": correlation,
    }


# ============================================================
# 8. BUILD COMPARISONS
# ============================================================

metric_rows = []

for location, info in STATIONS.items():

    print("\n" + "=" * 78)
    print(location)
    print("=" * 78)

    t = troute[
        troute["location"] == location
    ].copy()

    th = troute_hourly[
        troute_hourly["location"] == location
    ].copy()

    n = nwm[
        nwm["location"] == location
    ].copy()

    u = usgs[
        usgs["location"] == location
    ].copy()

    # --------------------------------------------------------
    # Common hourly USGS/NWM/t-route timestamps
    # --------------------------------------------------------

    u_hourly = u[
        (u["time"].dt.minute == 0)
        & (u["time"].dt.second == 0)
    ].copy()

    common = (
        u_hourly
        .merge(
            n[
                [
                    "time",
                    "nwm_streamflow",
                ]
            ],
            on="time",
            how="inner",
        )
        .merge(
            th[
                [
                    "time",
                    "troute_streamflow",
                ]
            ],
            on="time",
            how="inner",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(
        "Native USGS records:",
        len(u),
    )

    print(
        "USGS exact-hour records:",
        len(u_hourly),
    )

    print(
        "Three-way common hourly records:",
        len(common),
    )

    print(
        "Common non-estimated:",
        (~common["is_estimated"]).sum(),
    )

    print(
        "Common estimated:",
        common["is_estimated"].sum(),
    )

    # Save common comparison table
    comparison_file = (
        OUTPUT_DIR
        / f"{location}_three_way_hourly_20161101_20170501.csv"
    )

    common.to_csv(
        comparison_file,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics against USGS
    # --------------------------------------------------------

    subsets = {
        "all_USGS_reported": common,
        "non_estimated_USGS": common[
            ~common["is_estimated"]
        ],
        "estimated_USGS": common[
            common["is_estimated"]
        ],
    }

    for subset_name, subset in subsets.items():

        for model_name, model_col in [
            (
                "t-route_standard_MC",
                "troute_streamflow",
            ),
            (
                "NWM_v3",
                "nwm_streamflow",
            ),
        ]:

            metrics = calculate_metrics(
                subset[model_col],
                subset["usgs_streamflow"],
            )

            metric_rows.append(
                {
                    "location": location,
                    "subset": subset_name,
                    "model": model_name,
                    **metrics,
                }
            )

    # --------------------------------------------------------
    # t-route versus NWM over every common model hour
    # --------------------------------------------------------

    model_common = (
        n[
            [
                "time",
                "nwm_streamflow",
            ]
        ]
        .merge(
            th[
                [
                    "time",
                    "troute_streamflow",
                ]
            ],
            on="time",
            how="inner",
        )
        .sort_values("time")
    )

    model_metrics = calculate_metrics(
        model_common["troute_streamflow"],
        model_common["nwm_streamflow"],
    )

    metric_rows.append(
        {
            "location": location,
            "subset": "all_common_model_hours",
            "model": "t-route_vs_NWM",
            **model_metrics,
        }
    )

    print(
        "t-route/NWM common model hours:",
        len(model_common),
    )

    # ========================================================
    # 9. HYDROGRAPH
    #
    # Model lines use native model resolution:
    #   t-route = 5 min
    #   NWM     = 1 hour
    #
    # USGS is shown only at timestamps actually reported.
    # Estimated values are distinguished from non-estimated.
    # No USGS interpolation is performed.
    # ========================================================

    observed = u[
        ~u["is_estimated"]
    ]

    estimated = u[
        u["is_estimated"]
    ]

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    ax.plot(
        t["time"],
        t["troute_streamflow"],
        linewidth=1.1,
        label="t-route standard MC",
    )

    ax.plot(
        n["time"],
        n["nwm_streamflow"],
        linewidth=1.1,
        label="NWM v3 retrospective",
    )

    ax.scatter(
        observed["time"],
        observed["usgs_streamflow"],
        s=8,
        label="USGS reported",
        zorder=4,
    )

    ax.scatter(
        estimated["time"],
        estimated["usgs_streamflow"],
        s=22,
        marker="x",
        label="USGS estimated",
        zorder=5,
    )

    display_name = (
        "Fort Kent"
        if location == "Fort_Kent"
        else location
    )

    ax.set_title(
        f"{display_name}: Standard MC vs NWM v3 vs USGS\n"
        "November 2016–April 2017"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Streamflow (m³/s)"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    ax.xaxis.set_major_locator(
        mdates.MonthLocator()
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b\n%Y"
        )
    )

    fig.tight_layout()

    figure_file = (
        OUTPUT_DIR
        / f"{location}_standard_MC_NWM_USGS_20161101_20170501.png"
    )

    fig.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved comparison:",
        comparison_file,
    )

    print(
        "Saved hydrograph:",
        figure_file,
    )


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    metric_rows
)

metrics_file = (
    OUTPUT_DIR
    / "standard_MC_NWM_USGS_metrics_20161101_20170501.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False,
)

print("\n" + "=" * 78)
print("METRICS")
print("=" * 78)

print(
    metrics_df.to_string(
        index=False
    )
)

print("\nSaved metrics:")
print(metrics_file)

print("\n" + "=" * 78)
print("SIX-MONTH COMPARISON COMPLETE")
print("=" * 78)
