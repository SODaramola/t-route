# ============================================================
# VISUALIZE T-ROUTE LOWER COLORADO TEST
#
# Creates:
#   1. Lower Colorado routing network colored by peak flow
#   2. Hydrograph for the segment with the largest peak flow
#
# Inputs:
#   domain/RouteLink.nc
#   output/flowveldepth_*.parquet
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ============================================================
# 1. PATHS
# ============================================================

base_dir = Path(__file__).resolve().parent

domain_file = base_dir / "domain" / "RouteLink.nc"
output_dir = base_dir / "output"

print("Test directory:")
print(base_dir)

print("\nRouteLink file:")
print(domain_file)


# ============================================================
# 2. FIND MOST RECENT T-ROUTE PARQUET OUTPUT
# ============================================================

parquet_files = list(output_dir.glob("flowveldepth_*.parquet"))

if not parquet_files:
    raise FileNotFoundError(
        f"No flowveldepth parquet file found in:\n{output_dir}"
    )

parquet_file = max(
    parquet_files,
    key=lambda p: p.stat().st_mtime
)

print("\nUsing output:")
print(parquet_file)


# ============================================================
# 3. READ T-ROUTE RESULTS
# ============================================================

results = pd.read_parquet(parquet_file)

print("\nOutput columns:")
print(results.columns.tolist())

print("\nFirst rows:")
print(results.head())

print("\nVariables:")
print(results["variable_name"].value_counts())

results["value_time"] = pd.to_datetime(results["value_time"])


# ============================================================
# 4. EXTRACT NHD LINK ID FROM LOCATION ID
#
# Example:
# nex-1234567 -> 1234567
# ============================================================

results["link"] = (
    results["location_id"]
    .astype(str)
    .str.extract(r"(\d+)$")[0]
)

results["link"] = pd.to_numeric(
    results["link"],
    errors="coerce"
).astype("Int64")

results = results.dropna(subset=["link"]).copy()

results["link"] = results["link"].astype(np.int64)


# ============================================================
# 5. SEPARATE STREAMFLOW RESULTS
# ============================================================

flow = results[
    results["variable_name"] == "streamflow"
].copy()

if flow.empty:
    raise RuntimeError(
        "No streamflow records found in the parquet output."
    )

print("\nNumber of routed streamflow records:")
print(len(flow))

print("\nNumber of routed links:")
print(flow["link"].nunique())


# ============================================================
# 6. COMPUTE PEAK STREAMFLOW FOR EACH LINK
# ============================================================

peak_flow = (
    flow
    .groupby("link", as_index=False)["value"]
    .max()
    .rename(columns={"value": "peak_streamflow"})
)

print("\nPeak-flow summary:")
print(peak_flow["peak_streamflow"].describe())


# ============================================================
# 7. READ ROUTELINK NETWORK
# ============================================================

ds = xr.open_dataset(domain_file)

print("\nRouteLink variables:")
print(list(ds.variables))


# ============================================================
# 8. GET NETWORK VARIABLES
# ============================================================

required = ["link", "to", "lon", "lat"]

missing = [
    name for name in required
    if name not in ds.variables
]

if missing:
    raise KeyError(
        "The following expected RouteLink variables "
        f"were not found: {missing}"
    )

network = pd.DataFrame(
    {
        "link": np.asarray(ds["link"].values).squeeze(),
        "to": np.asarray(ds["to"].values).squeeze(),
        "lon": np.asarray(ds["lon"].values).squeeze(),
        "lat": np.asarray(ds["lat"].values).squeeze(),
    }
)

network["link"] = network["link"].astype(np.int64)
network["to"] = network["to"].astype(np.int64)

network = network.replace(
    [np.inf, -np.inf],
    np.nan
)

network = network.dropna(
    subset=["lon", "lat"]
)

print("\nNumber of RouteLink segments:")
print(len(network))


# ============================================================
# 9. ADD PEAK STREAMFLOW TO NETWORK
# ============================================================

network = network.merge(
    peak_flow,
    on="link",
    how="left"
)


# ============================================================
# 10. BUILD LINK COORDINATE LOOKUP
# ============================================================

coords = {
    int(row.link): (float(row.lon), float(row.lat))
    for row in network.itertuples()
}


# ============================================================
# 11. BUILD NETWORK LINE SEGMENTS
#
# Each t-route link is connected to its downstream link.
# ============================================================

segments = []
segment_values = []

for row in network.itertuples():

    upstream_id = int(row.link)
    downstream_id = int(row.to)

    if (
        upstream_id in coords
        and downstream_id in coords
        and pd.notna(row.peak_streamflow)
    ):

        x1, y1 = coords[upstream_id]
        x2, y2 = coords[downstream_id]

        segments.append(
            [(x1, y1), (x2, y2)]
        )

        segment_values.append(
            float(row.peak_streamflow)
        )


if not segments:
    raise RuntimeError(
        "No connected network segments could be constructed."
    )


# ============================================================
# 12. MAP PEAK STREAMFLOW
# ============================================================

fig, ax = plt.subplots(figsize=(11, 8))

line_collection = LineCollection(
    segments,
    array=np.asarray(segment_values),
    linewidths=1.6
)

ax.add_collection(line_collection)

ax.autoscale()

colorbar = fig.colorbar(
    line_collection,
    ax=ax
)

colorbar.set_label(
    "Peak streamflow (m³/s)",
    fontsize=12
)

ax.set_title(
    "t-route Lower Colorado River Test\n"
    "Peak Simulated Streamflow",
    fontsize=14
)

ax.set_xlabel(
    "Longitude",
    fontsize=12
)

ax.set_ylabel(
    "Latitude",
    fontsize=12
)

ax.grid(
    alpha=0.25
)

plt.tight_layout()

map_file = (
    output_dir
    / "LowerColorado_peak_streamflow_map.png"
)

plt.savefig(
    map_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved map:")
print(map_file)

plt.show()


# ============================================================
# 13. FIND LINK WITH LARGEST PEAK FLOW
# ============================================================

max_row = peak_flow.loc[
    peak_flow["peak_streamflow"].idxmax()
]

selected_link = int(
    max_row["link"]
)

selected_peak = float(
    max_row["peak_streamflow"]
)

print("\nLink with largest peak flow:")
print(selected_link)

print("\nLargest simulated peak streamflow:")
print(f"{selected_peak:.2f} m3/s")


# ============================================================
# 14. HYDROGRAPH FOR THAT LINK
# ============================================================

hydrograph = flow[
    flow["link"] == selected_link
].copy()

hydrograph = hydrograph.sort_values(
    "value_time"
)


fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    hydrograph["value_time"],
    hydrograph["value"],
    linewidth=1.8
)

ax.set_title(
    f"t-route Streamflow Hydrograph\n"
    f"NHD Link {selected_link}",
    fontsize=14
)

ax.set_xlabel(
    "Time",
    fontsize=12
)

ax.set_ylabel(
    "Streamflow (m³/s)",
    fontsize=12
)

ax.grid(
    alpha=0.25
)

fig.autofmt_xdate()

plt.tight_layout()

hydrograph_file = (
    output_dir
    / f"hydrograph_link_{selected_link}.png"
)

plt.savefig(
    hydrograph_file,
    dpi=300,
    bbox_inches="tight"
)

print("\nSaved hydrograph:")
print(hydrograph_file)

plt.show()


# ============================================================
# 15. CLOSE NETCDF
# ============================================================

ds.close()


print("\n========================================")
print("VISUALIZATION COMPLETE")
print("========================================")