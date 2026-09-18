from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MC_FILE = (
    BASE_DIR
    / "output"
    / "UpperSaintJohn_MC_20161101_20170501_CHANOBS.nc"
)

DW_FILE = (
    BASE_DIR
    / "output"
    / "UpperSaintJohn_MC_DW_20161101_20170501.parquet"
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
        "location_id": "nex-4288603",
        "usgs": "01010500",
    },
    "Fort_Kent": {
        "feature_id": 4287759,
        "location_id": "nex-4287759",
        "usgs": "01014000",
    },
}


# ============================================================
# 3. METRIC FUNCTION
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

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    if np.sum(obs) != 0:

        percent_bias = (
            100.0
            * np.sum(error)
            / np.sum(obs)
        )

    else:

        percent_bias = np.nan

    if (
        n > 1
        and np.std(model) > 0
        and np.std(obs) > 0
    ):

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
# 4. READ T-ROUTE MC
# ============================================================

ds = xr.open_dataset(
    MC_FILE
)

mc_frames = []

for location, info in STATIONS.items():

    q = ds["streamflow"].sel(
        feature_id=info["feature_id"]
    )

    if "reference_time" in q.dims:
        q = q.squeeze(
            "reference_time",
            drop=True,
        )

    q = np.asarray(
        q.values,
        dtype=float,
    ).reshape(-1)

    mc_frames.append(
        pd.DataFrame(
            {
                "time": pd.to_datetime(
                    ds["time"].values
                ),
                "location": location,
                "mc_streamflow": q,
            }
        )
    )


mc = pd.concat(
    mc_frames,
    ignore_index=True,
)

ds.close()


# ============================================================
# 5. READ T-ROUTE MC-DW
# ============================================================

dw_raw = pd.read_parquet(
    DW_FILE
)

dw_raw["value_time"] = pd.to_datetime(
    dw_raw["value_time"]
)

dw_raw = dw_raw[
    dw_raw["variable_name"]
    == "streamflow"
].copy()

dw_frames = []

for location, info in STATIONS.items():

    d = dw_raw[
        dw_raw["location_id"]
        == info["location_id"]
    ].copy()

    d = d.sort_values(
        "value_time"
    )

    dw_frames.append(
        pd.DataFrame(
            {
                "time": d["value_time"].values,
                "location": location,
                "dw_streamflow": pd.to_numeric(
                    d["value"],
                    errors="coerce",
                ).values,
            }
        )
    )


dw = pd.concat(
    dw_frames,
    ignore_index=True,
)


# ============================================================
# 6. READ NWM V3
# ============================================================

nwm = pd.read_csv(
    NWM_FILE
)

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
# 7. READ USGS
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
# 8. BASIC STRUCTURAL CHECKS
# ============================================================

print("=" * 82)
print("STRUCTURAL CHECKS")
print("=" * 82)

for location in STATIONS:

    m = mc[
        mc["location"] == location
    ]

    d = dw[
        dw["location"] == location
    ]

    print(
        f"{location:12s}  "
        f"MC={len(m):5d}  "
        f"MC-DW={len(d):5d}"
    )

    assert len(m) == 52128
    assert len(d) == 52128

    assert m["time"].is_unique
    assert d["time"].is_unique

    assert np.isfinite(
        d["dw_streamflow"]
    ).all()


# ============================================================
# 9. EXACT HOURLY MODEL VALUES
#
# No interpolation.
# ============================================================

mc_hourly = mc[
    (mc["time"].dt.minute == 0)
    & (mc["time"].dt.second == 0)
].copy()

dw_hourly = dw[
    (dw["time"].dt.minute == 0)
    & (dw["time"].dt.second == 0)
].copy()


# ============================================================
# 10. EXACT-HOUR USGS VALUES
#
# Preserve only observations that actually occurred at the
# top of the hour. No resampling and no interpolation.
# ============================================================

usgs_hourly = usgs[
    (usgs["time"].dt.minute == 0)
    & (usgs["time"].dt.second == 0)
].copy()


# ============================================================
# 11. FOUR-WAY COMMON HOURLY DATASET
# ============================================================

common_frames = []

for location in STATIONS:

    a = mc_hourly[
        mc_hourly["location"] == location
    ].copy()

    b = dw_hourly[
        dw_hourly["location"] == location
    ].copy()

    c = nwm[
        nwm["location"] == location
    ].copy()

    d = usgs_hourly[
        usgs_hourly["location"] == location
    ].copy()

    x = (
        a[
            [
                "time",
                "location",
                "mc_streamflow",
            ]
        ]
        .merge(
            b[
                [
                    "time",
                    "location",
                    "dw_streamflow",
                ]
            ],
            on=[
                "time",
                "location",
            ],
            how="inner",
        )
        .merge(
            c[
                [
                    "time",
                    "location",
                    "nwm_streamflow",
                ]
            ],
            on=[
                "time",
                "location",
            ],
            how="inner",
        )
        .merge(
            d[
                [
                    "time",
                    "location",
                    "usgs_streamflow",
                    "is_estimated",
                ]
            ],
            on=[
                "time",
                "location",
            ],
            how="inner",
        )
    )

    x = x.sort_values(
        "time"
    ).reset_index(drop=True)

    common_frames.append(x)


common = pd.concat(
    common_frames,
    ignore_index=True,
)

common.to_csv(
    OUTPUT_DIR
    / "MC_DW_NWM_USGS_common_hourly.csv",
    index=False,
)


# ============================================================
# 12. USGS VALIDATION METRICS
# ============================================================

metric_rows = []

for location in STATIONS:

    x = common[
        common["location"] == location
    ].copy()

    subsets = {
        "all": x,
        "non_estimated": x[
            ~x["is_estimated"]
        ].copy(),
        "estimated": x[
            x["is_estimated"]
        ].copy(),
    }

    print("\n" + "=" * 82)
    print(location.upper())
    print("=" * 82)

    print(
        "Four-way common exact-hour records:",
        len(x),
    )

    for subset_name, s in subsets.items():

        print(
            f"\n{subset_name.upper()} "
            f"(n={len(s)})"
        )

        for model_name, column in [
            (
                "t-route (MC)",
                "mc_streamflow",
            ),
            (
                "t-route (MC-DW)",
                "dw_streamflow",
            ),
            (
                "NWM v3",
                "nwm_streamflow",
            ),
        ]:

            metrics = calculate_metrics(
                s[column],
                s["usgs_streamflow"],
            )

            metric_rows.append(
                {
                    "location": location,
                    "subset": subset_name,
                    "model": model_name,
                    **metrics,
                }
            )

            print(
                f"{model_name:17s} "
                f"bias={metrics['bias_cms']:10.3f}  "
                f"MAE={metrics['mae_cms']:10.3f}  "
                f"RMSE={metrics['rmse_cms']:10.3f}  "
                f"PBIAS={metrics['percent_bias']:9.3f}%  "
                f"r={metrics['correlation']:8.4f}"
            )


metrics_df = pd.DataFrame(
    metric_rows
)

metrics_df.to_csv(
    OUTPUT_DIR
    / "MC_DW_NWM_USGS_metrics.csv",
    index=False,
)


# ============================================================
# 13. FULL 5-MINUTE MC VS MC-DW
# ============================================================

model_rows = []

for location in STATIONS:

    m = mc[
        mc["location"] == location
    ][
        [
            "time",
            "mc_streamflow",
        ]
    ].copy()

    d = dw[
        dw["location"] == location
    ][
        [
            "time",
            "dw_streamflow",
        ]
    ].copy()

    x = m.merge(
        d,
        on="time",
        how="inner",
    )

    metrics = calculate_metrics(
        x["dw_streamflow"],
        x["mc_streamflow"],
    )

    model_rows.append(
        {
            "location": location,
            "comparison": "MC-DW minus MC",
            **metrics,
        }
    )

    print("\n" + "=" * 82)
    print(
        f"{location}: FULL 5-MINUTE MC-DW VS MC"
    )
    print("=" * 82)

    print(
        f"n       = {metrics['n']}"
    )

    print(
        f"Bias    = {metrics['bias_cms']:.6f} m3/s"
    )

    print(
        f"MAE     = {metrics['mae_cms']:.6f} m3/s"
    )

    print(
        f"RMSE    = {metrics['rmse_cms']:.6f} m3/s"
    )

    print(
        f"PBIAS   = {metrics['percent_bias']:.6f} %"
    )

    print(
        f"r       = {metrics['correlation']:.8f}"
    )


# ============================================================
# 14. HOURLY DW VS NWM
# ============================================================

for location in STATIONS:

    d = dw_hourly[
        dw_hourly["location"] == location
    ][
        [
            "time",
            "dw_streamflow",
        ]
    ].copy()

    n = nwm[
        nwm["location"] == location
    ][
        [
            "time",
            "nwm_streamflow",
        ]
    ].copy()

    x = d.merge(
        n,
        on="time",
        how="inner",
    )

    metrics = calculate_metrics(
        x["dw_streamflow"],
        x["nwm_streamflow"],
    )

    model_rows.append(
        {
            "location": location,
            "comparison": "MC-DW minus NWM v3",
            **metrics,
        }
    )


model_metrics_df = pd.DataFrame(
    model_rows
)

model_metrics_df.to_csv(
    OUTPUT_DIR
    / "MC_DW_model_to_model_metrics.csv",
    index=False,
)


# ============================================================
# 15. PEAK MAGNITUDE AND TIMING
#
# Peaks are calculated on the FOUR-WAY COMMON EXACT-HOUR
# dataset so every series is compared using identical times.
# ============================================================

peak_rows = []

for location in STATIONS:

    x = common[
        common["location"] == location
    ].copy()

    if x.empty:
        continue

    usgs_peak_idx = (
        x["usgs_streamflow"]
        .astype(float)
        .idxmax()
    )

    usgs_peak_time = x.loc[
        usgs_peak_idx,
        "time",
    ]

    for model_name, column in [
        (
            "USGS",
            "usgs_streamflow",
        ),
        (
            "t-route (MC)",
            "mc_streamflow",
        ),
        (
            "t-route (MC-DW)",
            "dw_streamflow",
        ),
        (
            "NWM v3",
            "nwm_streamflow",
        ),
    ]:

        idx = (
            x[column]
            .astype(float)
            .idxmax()
        )

        peak_time = x.loc[
            idx,
            "time",
        ]

        peak_value = float(
            x.loc[
                idx,
                column,
            ]
        )

        lag_hours = (
            peak_time
            - usgs_peak_time
        ).total_seconds() / 3600.0

        peak_rows.append(
            {
                "location": location,
                "series": model_name,
                "peak_cms": peak_value,
                "peak_time": peak_time,
                "lag_from_USGS_peak_hours": lag_hours,
                "basis": "four-way common exact-hour timestamps",
            }
        )


peaks_df = pd.DataFrame(
    peak_rows
)

peaks_df.to_csv(
    OUTPUT_DIR
    / "MC_DW_NWM_USGS_peaks_common_hourly.csv",
    index=False,
)


# ============================================================
# 16. MONTHLY NOVEMBER-APRIL DIAGNOSTICS
#
# Again use four-way common exact-hour timestamps.
# These are diagnostic sample means, not monthly flow volumes.
# ============================================================

monthly_rows = []

for location in STATIONS:

    x = common[
        common["location"] == location
    ].copy()

    x["month"] = (
        x["time"]
        .dt.to_period("M")
        .astype(str)
    )

    for month, g in x.groupby(
        "month"
    ):

        if month not in {
            "2016-11",
            "2016-12",
            "2017-01",
            "2017-02",
            "2017-03",
            "2017-04",
        }:
            continue

        usgs_peak_idx = (
            g["usgs_streamflow"]
            .astype(float)
            .idxmax()
        )

        mc_peak_idx = (
            g["mc_streamflow"]
            .astype(float)
            .idxmax()
        )

        dw_peak_idx = (
            g["dw_streamflow"]
            .astype(float)
            .idxmax()
        )

        nwm_peak_idx = (
            g["nwm_streamflow"]
            .astype(float)
            .idxmax()
        )

        monthly_rows.append(
            {
                "location": location,
                "month": month,
                "n_common": len(g),
                "n_estimated_usgs": int(
                    g["is_estimated"].sum()
                ),

                "usgs_mean_cms":
                    g["usgs_streamflow"].mean(),

                "mc_mean_cms":
                    g["mc_streamflow"].mean(),

                "dw_mean_cms":
                    g["dw_streamflow"].mean(),

                "nwm_mean_cms":
                    g["nwm_streamflow"].mean(),

                "mc_bias_cms":
                    (
                        g["mc_streamflow"]
                        - g["usgs_streamflow"]
                    ).mean(),

                "dw_bias_cms":
                    (
                        g["dw_streamflow"]
                        - g["usgs_streamflow"]
                    ).mean(),

                "nwm_bias_cms":
                    (
                        g["nwm_streamflow"]
                        - g["usgs_streamflow"]
                    ).mean(),

                "usgs_peak_cms":
                    g.loc[
                        usgs_peak_idx,
                        "usgs_streamflow",
                    ],

                "usgs_peak_time":
                    g.loc[
                        usgs_peak_idx,
                        "time",
                    ],

                "mc_peak_cms":
                    g.loc[
                        mc_peak_idx,
                        "mc_streamflow",
                    ],

                "mc_peak_time":
                    g.loc[
                        mc_peak_idx,
                        "time",
                    ],

                "dw_peak_cms":
                    g.loc[
                        dw_peak_idx,
                        "dw_streamflow",
                    ],

                "dw_peak_time":
                    g.loc[
                        dw_peak_idx,
                        "time",
                    ],

                "nwm_peak_cms":
                    g.loc[
                        nwm_peak_idx,
                        "nwm_streamflow",
                    ],

                "nwm_peak_time":
                    g.loc[
                        nwm_peak_idx,
                        "time",
                    ],
            }
        )


monthly_df = pd.DataFrame(
    monthly_rows
)

monthly_df.to_csv(
    OUTPUT_DIR
    / "MC_DW_NWM_USGS_monthly.csv",
    index=False,
)


# ============================================================
# 17. PRINT PEAK SUMMARY
# ============================================================

print("\n" + "=" * 82)
print("PEAK SUMMARY — FOUR-WAY COMMON EXACT-HOUR DATA")
print("=" * 82)

for location in STATIONS:

    p = peaks_df[
        peaks_df["location"] == location
    ]

    print("\n" + location)

    for _, row in p.iterrows():

        print(
            f"{row['series']:17s} "
            f"{row['peak_cms']:10.3f} m3/s  "
            f"{row['peak_time']}  "
            f"lag={row['lag_from_USGS_peak_hours']:8.1f} h"
        )


# ============================================================
# 18. PRINT MONTHLY MEANS
# ============================================================

print("\n" + "=" * 82)
print("MONTHLY COMMON-HOUR MEANS")
print("=" * 82)

for location in STATIONS:

    m = monthly_df[
        monthly_df["location"]
        == location
    ]

    print("\n" + location)

    print(
        m[
            [
                "month",
                "n_common",
                "n_estimated_usgs",
                "usgs_mean_cms",
                "mc_mean_cms",
                "dw_mean_cms",
                "nwm_mean_cms",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# ============================================================
# 19. OUTPUT FILES
# ============================================================

print("\n" + "=" * 82)
print("OUTPUT FILES")
print("=" * 82)

for p in [
    OUTPUT_DIR
    / "MC_DW_NWM_USGS_common_hourly.csv",

    OUTPUT_DIR
    / "MC_DW_NWM_USGS_metrics.csv",

    OUTPUT_DIR
    / "MC_DW_model_to_model_metrics.csv",

    OUTPUT_DIR
    / "MC_DW_NWM_USGS_peaks_common_hourly.csv",

    OUTPUT_DIR
    / "MC_DW_NWM_USGS_monthly.csv",
]:

    print(p)


print("\nPASS: MC / MC-DW / NWM v3 / USGS comparison complete.")
