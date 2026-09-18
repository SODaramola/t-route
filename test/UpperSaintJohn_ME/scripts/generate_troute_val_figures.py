from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "output"
VAL_DIR = OUTPUT_DIR / "troute_val"
VAL_DIR.mkdir(parents=True, exist_ok=True)

STUDY_REACH_SRC = OUTPUT_DIR / "UpperSaintJohn_study_reach.png"
STUDY_REACH_DST = VAL_DIR / "01_UpperSaintJohn_study_reach.png"

METRICS_FILE = OUTPUT_DIR / "standard_MC_NWM_USGS_metrics_20161101_20170501.csv"

THREE_WAY_FILES = {
    "Dickey": OUTPUT_DIR / "Dickey_three_way_hourly_20161101_20170501.csv",
    "Fort_Kent": OUTPUT_DIR / "Fort_Kent_three_way_hourly_20161101_20170501.csv",
}

MONTHLY_FILES = {
    "Dickey": OUTPUT_DIR / "Dickey_monthly_baseline_diagnostics_201611_201704.csv",
    "Fort_Kent": OUTPUT_DIR / "Fort_Kent_monthly_baseline_diagnostics_201611_201704.csv",
}

INCREMENTAL_FILE = OUTPUT_DIR / "Dickey_FortKent_incremental_flow_diagnostics.csv"


# ============================================================
# 2. PLOTTING STYLE
# ============================================================

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 12
rcParams["axes.titlesize"] = 14
rcParams["axes.labelsize"] = 12
rcParams["legend.fontsize"] = 11
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11


# ============================================================
# 3. HELPERS
# ============================================================

def display_name(location):
    return location.replace("_", " ")


def choose_column(df, exact=None, contains_all=None, contains_any=None, exclude=None):
    cols = list(df.columns)

    if exact:
        for name in exact:
            if name in cols:
                return name

    contains_all = contains_all or []
    contains_any = contains_any or []
    exclude = exclude or []

    lowered = {c: c.lower() for c in cols}

    for c in cols:
        lc = lowered[c]

        if contains_all and not all(x in lc for x in contains_all):
            continue

        if contains_any and not any(x in lc for x in contains_any):
            continue

        if exclude and any(x in lc for x in exclude):
            continue

        return c

    raise RuntimeError(
        "Could not identify expected column.\n"
        f"Available columns: {cols}"
    )


def format_metric_row(row, label):
    return (
        f"{label}: "
        f"n={int(row['n'])}, "
        f"Bias={row['bias_cms']:.1f}, "
        f"MAE={row['mae_cms']:.1f}, "
        f"RMSE={row['rmse_cms']:.1f}, "
        f"r={row['correlation']:.3f}"
    )


def load_three_way(location):
    df = pd.read_csv(THREE_WAY_FILES[location])
    df["time"] = pd.to_datetime(df["time"])

    usgs_col = choose_column(
        df,
        exact=["usgs_streamflow", "USGS_streamflow", "usgs_cms", "USGS_cms"],
        contains_all=["usgs"],
    )

    nwm_col = choose_column(
        df,
        exact=["nwm_streamflow", "NWM_streamflow", "nwm_cms", "NWM_cms"],
        contains_all=["nwm"],
    )

    troute_col = choose_column(
        df,
        exact=[
            "troute_streamflow",
            "tRoute_streamflow",
            "troute_cms",
            "tRoute_cms",
        ],
        contains_any=["troute", "t-route"],
    )

    estimated_col = None
    for c in df.columns:
        if "estimated" in c.lower():
            estimated_col = c
            break

    if estimated_col is not None:
        df[estimated_col] = df[estimated_col].fillna(False).astype(bool)

    return df, usgs_col, nwm_col, troute_col, estimated_col


def get_metric_text(metrics_df, location):
    m = metrics_df[
        (metrics_df["location"] == location)
        & (metrics_df["subset"] == "all_USGS_reported")
    ].copy()

    rows = []

    tr = m[m["model"] == "t-route_standard_MC"]
    if not tr.empty:
        rows.append(format_metric_row(tr.iloc[0], "t-route (MC)"))

    nwm = m[m["model"] == "NWM_v3"]
    if not nwm.empty:
        rows.append(format_metric_row(nwm.iloc[0], "NWM v3"))

    return "All USGS-reported common hours\n" + "\n".join(rows)


def add_metric_box(ax, text):
    ax.text(
        0.015,
        0.985,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="black"),
    )


# ============================================================
# 4. COPY STUDY REACH IMAGE
# ============================================================

if not STUDY_REACH_SRC.exists():
    raise FileNotFoundError(f"Study reach image not found: {STUDY_REACH_SRC}")

shutil.copy2(STUDY_REACH_SRC, STUDY_REACH_DST)
print(f"Saved: {STUDY_REACH_DST}")


# ============================================================
# 5. LOAD METRICS
# ============================================================

metrics_df = pd.read_csv(METRICS_FILE)


# ============================================================
# 6. FULL HYDROGRAPHS
# ============================================================

figure_counter = 2

for location in ["Dickey", "Fort_Kent"]:
    df, usgs_col, nwm_col, troute_col, estimated_col = load_three_way(location)

    fig, ax = plt.subplots(figsize=(14, 6.5))

    ax.plot(
        df["time"],
        df[usgs_col],
        label="USGS",
        linewidth=1.8,
    )

    ax.plot(
        df["time"],
        df[nwm_col],
        label="NWM v3",
        linewidth=1.6,
    )

    ax.plot(
        df["time"],
        df[troute_col],
        label="t-route (MC)",
        linewidth=1.6,
    )

    if estimated_col is not None:
        est = df[df[estimated_col]].copy()
        if not est.empty:
            ax.scatter(
                est["time"],
                est[usgs_col],
                s=12,
                label="USGS estimated",
                zorder=5,
            )

    ax.set_title(
        f"{display_name(location)}: t-route (MC) vs NWM v3 vs USGS\n"
        "2016-11-01 to 2017-05-01"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Streamflow (m$^3$/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=4)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    metric_text = get_metric_text(metrics_df, location)
    add_metric_box(ax, metric_text)

    fig.autofmt_xdate()
    fig.tight_layout()

    out = VAL_DIR / f"{figure_counter:02d}_{location}_t-route_MC_vs_NWM_v3_vs_USGS.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 7. MONTHLY MEAN FLOW FIGURES
# ============================================================

for location in ["Dickey", "Fort_Kent"]:
    df = pd.read_csv(MONTHLY_FILES[location])
    df["month"] = pd.to_datetime(df["month"].astype(str))

    fig, ax = plt.subplots(figsize=(12, 5.8))

    ax.plot(df["month"], df["USGS_mean"], marker="o", linewidth=1.8, label="USGS")
    ax.plot(df["month"], df["NWM_mean"], marker="o", linewidth=1.6, label="NWM v3")
    ax.plot(df["month"], df["tRoute_mean"], marker="o", linewidth=1.6, label="t-route (MC)")

    ax.set_title(f"{display_name(location)}: Monthly Mean Flow Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly mean streamflow (m$^3$/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    mean_nwm_bias = df["NWM_bias_pct"].mean()
    mean_troute_bias = df["tRoute_bias_pct"].mean()
    total_est = int(df["estimated_n"].sum())

    metric_text = (
        "Monthly diagnostics\n"
        f"Mean NWM v3 bias = {mean_nwm_bias:.1f}%\n"
        f"Mean t-route (MC) bias = {mean_troute_bias:.1f}%\n"
        f"Estimated USGS points in common-hour sample = {total_est}"
    )
    add_metric_box(ax, metric_text)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()

    out = VAL_DIR / f"{figure_counter:02d}_{location}_monthly_mean_flow.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 8. MONTHLY PEAK FLOW FIGURES
# ============================================================

for location in ["Dickey", "Fort_Kent"]:
    df = pd.read_csv(MONTHLY_FILES[location])
    df["month"] = pd.to_datetime(df["month"].astype(str))
    df["USGS_peak_time"] = pd.to_datetime(df["USGS_peak_time"])
    df["NWM_peak_time"] = pd.to_datetime(df["NWM_peak_time"])
    df["tRoute_peak_time"] = pd.to_datetime(df["tRoute_peak_time"])

    nwm_peak_mae = np.mean(np.abs(df["NWM_max"] - df["USGS_max"]))
    troute_peak_mae = np.mean(np.abs(df["tRoute_max"] - df["USGS_max"]))

    nwm_peak_lag = np.mean(np.abs((df["NWM_peak_time"] - df["USGS_peak_time"]).dt.total_seconds() / 3600.0))
    troute_peak_lag = np.mean(np.abs((df["tRoute_peak_time"] - df["USGS_peak_time"]).dt.total_seconds() / 3600.0))

    fig, ax = plt.subplots(figsize=(12, 5.8))

    ax.plot(df["month"], df["USGS_max"], marker="o", linewidth=1.8, label="USGS")
    ax.plot(df["month"], df["NWM_max"], marker="o", linewidth=1.6, label="NWM v3")
    ax.plot(df["month"], df["tRoute_max"], marker="o", linewidth=1.6, label="t-route (MC)")

    ax.set_title(f"{display_name(location)}: Monthly Peak Flow Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly peak streamflow (m$^3$/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    metric_text = (
        "Monthly peak diagnostics\n"
        f"NWM v3 peak-flow MAE = {nwm_peak_mae:.1f} m$^3$/s\n"
        f"t-route (MC) peak-flow MAE = {troute_peak_mae:.1f} m$^3$/s\n"
        f"NWM v3 mean peak-timing error = {nwm_peak_lag:.1f} h\n"
        f"t-route (MC) mean peak-timing error = {troute_peak_lag:.1f} h"
    )
    add_metric_box(ax, metric_text)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.autofmt_xdate()
    fig.tight_layout()

    out = VAL_DIR / f"{figure_counter:02d}_{location}_monthly_peak_flow.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 9. INCREMENTAL FLOW FIGURE
# ============================================================

inc = pd.read_csv(INCREMENTAL_FILE)
inc["month"] = pd.to_datetime(inc["month"].astype(str))

fig, ax = plt.subplots(figsize=(12.5, 6))

ax.plot(
    inc["month"],
    inc["USGS_increment_mean"],
    marker="o",
    linewidth=1.8,
    label="USGS incremental flow",
)

ax.plot(
    inc["month"],
    inc["NWM_increment_mean"],
    marker="o",
    linewidth=1.6,
    label="NWM v3 incremental flow",
)

ax.plot(
    inc["month"],
    inc["tRoute_increment_mean"],
    marker="o",
    linewidth=1.6,
    label="t-route (MC) incremental flow",
)

ax.set_title("Dickey to Fort Kent: Monthly Mean Incremental Flow Comparison")
ax.set_xlabel("Month")
ax.set_ylabel("Incremental flow (m$^3$/s)")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")

metric_text = (
    "Incremental-flow diagnostics\n"
    f"Mean NWM v3 increment bias = {inc['NWM_increment_bias_pct'].mean():.1f}%\n"
    f"Mean t-route (MC) increment bias = {inc['tRoute_increment_bias_pct'].mean():.1f}%\n"
    f"Max USGS incremental monthly mean = {inc['USGS_increment_mean'].max():.1f} m$^3$/s"
)
add_metric_box(ax, metric_text)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

fig.autofmt_xdate()
fig.tight_layout()

out = VAL_DIR / f"{figure_counter:02d}_Dickey_to_FortKent_incremental_flow.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

figure_counter += 1


# ============================================================
# 10. SUMMARY FILE
# ============================================================

summary_file = VAL_DIR / "figure_inventory.txt"
with open(summary_file, "w") as f:
    for p in sorted(VAL_DIR.glob("*.png")):
        f.write(p.name + "\n")

print(f"Saved: {summary_file}")
print("\nAll validation figures created in:")
print(VAL_DIR)
