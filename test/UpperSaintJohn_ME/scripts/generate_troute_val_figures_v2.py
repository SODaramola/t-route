from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams, font_manager


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "output"
VAL_DIR = OUTPUT_DIR / "troute_val"
VAL_DIR.mkdir(parents=True, exist_ok=True)

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
# 2. FONT / STYLE
# ============================================================

# Force Times New Roman
font_manager._load_fontmanager(try_read_cache=False)

rcParams["font.family"] = "Times New Roman"
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300

rcParams["font.size"] = 15
rcParams["axes.titlesize"] = 18
rcParams["axes.labelsize"] = 16
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["legend.fontsize"] = 14

# Consistent colors across all figures
COLORS = {
    "USGS": "#1f77b4",            # blue
    "USGS_estimated": "#cc33cc",  # magenta
    "NWM v3": "#ff7f0e",          # orange
    "t-route (MC)": "#2ca02c",    # green
}

LINEWIDTH_MAIN = 2.5
MARKER_SIZE = 24


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

    for c in cols:
        lc = c.lower()

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


def metric_row_to_text(row, label):
    return (
        f"{label}\n"
        f"n = {int(row['n'])}\n"
        f"Bias = {row['bias_cms']:.1f} m$^3$/s\n"
        f"MAE = {row['mae_cms']:.1f} m$^3$/s\n"
        f"RMSE = {row['rmse_cms']:.1f} m$^3$/s\n"
        f"r = {row['correlation']:.3f}"
    )


def add_right_metric_box(fig, ax, text):
    fig.text(
        0.80,
        0.84,
        text,
        ha="left",
        va="top",
        fontsize=13,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="black",
            alpha=0.95
        ),
    )


def finalize_figure(fig, ax):
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.77, bottom=0.20, top=0.88)


# ============================================================
# 4. LOAD METRICS
# ============================================================

metrics_df = pd.read_csv(METRICS_FILE)


# ============================================================
# 5. FIGURE 2 & 3: FULL HYDROGRAPHS
# ============================================================

figure_counter = 2

for location in ["Dickey", "Fort_Kent"]:
    df, usgs_col, nwm_col, troute_col, estimated_col = load_three_way(location)

    fig, ax = plt.subplots(figsize=(17, 8))

    # Non-estimated USGS line only
    if estimated_col is not None:
        non_est = df.loc[~df[estimated_col]].copy()
        est = df.loc[df[estimated_col]].copy()
    else:
        non_est = df.copy()
        est = pd.DataFrame(columns=df.columns)

    ax.plot(
        non_est["time"],
        non_est[usgs_col],
        color=COLORS["USGS"],
        linewidth=LINEWIDTH_MAIN,
        label="USGS",
    )

    ax.plot(
        df["time"],
        df[nwm_col],
        color=COLORS["NWM v3"],
        linewidth=LINEWIDTH_MAIN,
        label="NWM v3",
    )

    ax.plot(
        df["time"],
        df[troute_col],
        color=COLORS["t-route (MC)"],
        linewidth=LINEWIDTH_MAIN,
        label="t-route (MC)",
    )

    # Estimated USGS markers only
    if not est.empty:
        ax.scatter(
            est["time"],
            est[usgs_col],
            color=COLORS["USGS_estimated"],
            s=MARKER_SIZE,
            label="USGS estimated",
            zorder=5,
        )

    ax.set_title(
        f"{display_name(location)}: t-route (MC) vs NWM v3 vs USGS\n"
        "2016-11-01 to 2017-05-01"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Streamflow (m$^3$/s)")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        frameon=True,
    )

    m = metrics_df[
        (metrics_df["location"] == location)
        & (metrics_df["subset"] == "all_USGS_reported")
    ].copy()

    text_lines = ["All USGS-reported common hours"]

    tr = m[m["model"] == "t-route_standard_MC"]
    if not tr.empty:
        text_lines.append(metric_row_to_text(tr.iloc[0], "t-route (MC)"))

    nwm = m[m["model"] == "NWM_v3"]
    if not nwm.empty:
        text_lines.append(metric_row_to_text(nwm.iloc[0], "NWM v3"))

    metric_text = "\n\n".join(text_lines)

    finalize_figure(fig, ax)
    add_right_metric_box(fig, ax, metric_text)

    out = VAL_DIR / f"{figure_counter:02d}_{location}_t-route_MC_vs_NWM_v3_vs_USGS.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 6. FIGURE 4 & 5: MONTHLY MEAN FLOW
# ============================================================

for location in ["Dickey", "Fort_Kent"]:
    df = pd.read_csv(MONTHLY_FILES[location])
    df["month"] = pd.to_datetime(df["month"].astype(str))

    fig, ax = plt.subplots(figsize=(15.5, 7.5))

    ax.plot(
        df["month"], df["USGS_mean"],
        color=COLORS["USGS"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="USGS"
    )
    ax.plot(
        df["month"], df["NWM_mean"],
        color=COLORS["NWM v3"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="NWM v3"
    )
    ax.plot(
        df["month"], df["tRoute_mean"],
        color=COLORS["t-route (MC)"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="t-route (MC)"
    )

    ax.set_title(f"{display_name(location)}: Monthly Mean Flow Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly mean streamflow (m$^3$/s)")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
    )

    mean_nwm_bias = df["NWM_bias_pct"].mean()
    mean_troute_bias = df["tRoute_bias_pct"].mean()
    total_est = int(df["estimated_n"].sum())

    metric_text = (
        "Monthly diagnostics\n\n"
        f"Mean NWM v3 bias = {mean_nwm_bias:.1f}%\n"
        f"Mean t-route (MC) bias = {mean_troute_bias:.1f}%\n"
        f"Estimated USGS points = {total_est}"
    )

    finalize_figure(fig, ax)
    add_right_metric_box(fig, ax, metric_text)

    out = VAL_DIR / f"{figure_counter:02d}_{location}_monthly_mean_flow.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 7. FIGURE 6 & 7: MONTHLY PEAK FLOW
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

    fig, ax = plt.subplots(figsize=(15.5, 7.5))

    ax.plot(
        df["month"], df["USGS_max"],
        color=COLORS["USGS"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="USGS"
    )
    ax.plot(
        df["month"], df["NWM_max"],
        color=COLORS["NWM v3"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="NWM v3"
    )
    ax.plot(
        df["month"], df["tRoute_max"],
        color=COLORS["t-route (MC)"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
        label="t-route (MC)"
    )

    ax.set_title(f"{display_name(location)}: Monthly Peak Flow Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly peak streamflow (m$^3$/s)")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=True,
    )

    metric_text = (
        "Monthly peak diagnostics\n\n"
        f"NWM v3 peak-flow MAE = {nwm_peak_mae:.1f} m$^3$/s\n"
        f"t-route (MC) peak-flow MAE = {troute_peak_mae:.1f} m$^3$/s\n"
        f"NWM v3 mean peak-timing error = {nwm_peak_lag:.1f} h\n"
        f"t-route (MC) mean peak-timing error = {troute_peak_lag:.1f} h"
    )

    finalize_figure(fig, ax)
    add_right_metric_box(fig, ax, metric_text)

    out = VAL_DIR / f"{figure_counter:02d}_{location}_monthly_peak_flow.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    figure_counter += 1


# ============================================================
# 8. FIGURE 8: INCREMENTAL FLOW
# ============================================================

inc = pd.read_csv(INCREMENTAL_FILE)
inc["month"] = pd.to_datetime(inc["month"].astype(str))

fig, ax = plt.subplots(figsize=(15.5, 7.5))

ax.plot(
    inc["month"], inc["USGS_increment_mean"],
    color=COLORS["USGS"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
    label="USGS incremental flow"
)

ax.plot(
    inc["month"], inc["NWM_increment_mean"],
    color=COLORS["NWM v3"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
    label="NWM v3 incremental flow"
)

ax.plot(
    inc["month"], inc["tRoute_increment_mean"],
    color=COLORS["t-route (MC)"], marker="o", linewidth=LINEWIDTH_MAIN, markersize=8,
    label="t-route (MC) incremental flow"
)

ax.set_title("Dickey to Fort Kent: Monthly Mean Incremental Flow Comparison")
ax.set_xlabel("Month")
ax.set_ylabel("Incremental flow (m$^3$/s)")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    ncol=3,
    frameon=True,
)

metric_text = (
    "Incremental-flow diagnostics\n\n"
    f"Mean NWM v3 increment bias = {inc['NWM_increment_bias_pct'].mean():.1f}%\n"
    f"Mean t-route (MC) increment bias = {inc['tRoute_increment_bias_pct'].mean():.1f}%\n"
    f"Max USGS incremental monthly mean = {inc['USGS_increment_mean'].max():.1f} m$^3$/s"
)

finalize_figure(fig, ax)
add_right_metric_box(fig, ax, metric_text)

out = VAL_DIR / f"{figure_counter:02d}_Dickey_to_FortKent_incremental_flow.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

print("\nAll revised validation figures saved in:")
print(VAL_DIR)
