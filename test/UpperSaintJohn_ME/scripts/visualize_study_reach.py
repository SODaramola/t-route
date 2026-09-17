# ============================================================
# VISUALIZE UPPER SAINT JOHN STUDY REACH
#
# Inputs:
#   domain/Dickey_downstream_150km.geojson
#   domain/Dickey_USGS_01010500.geojson
#   domain/Fort_Kent_USGS_01014000.geojson
#
# Output:
#   output/UpperSaintJohn_study_reach.png
# ============================================================

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOMAIN_DIR = BASE_DIR / "domain"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

flowline_file = DOMAIN_DIR / "Dickey_downstream_150km.geojson"
dickey_file = DOMAIN_DIR / "Dickey_USGS_01010500.geojson"
fort_kent_file = DOMAIN_DIR / "Fort_Kent_USGS_01014000.geojson"


# ============================================================
# 2. READ DATA
# ============================================================

flowlines = gpd.read_file(flowline_file)
dickey = gpd.read_file(dickey_file)
fort_kent = gpd.read_file(fort_kent_file)

print("Flowline features:", len(flowlines))

print("\nFlowline CRS:")
print(flowlines.crs)

print("\nDickey:")
print(dickey)

print("\nFort Kent:")
print(fort_kent)


# ============================================================
# 3. FORCE COMMON CRS
# ============================================================

flowlines = flowlines.to_crs("EPSG:4326")
dickey = dickey.to_crs("EPSG:4326")
fort_kent = fort_kent.to_crs("EPSG:4326")


# ============================================================
# 4. PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

flowlines.plot(
    ax=ax,
    linewidth=1.2,
)

dickey.plot(
    ax=ax,
    marker="o",
    markersize=80,
)

fort_kent.plot(
    ax=ax,
    marker="o",
    markersize=80,
)


# ============================================================
# 5. LABEL GAGES
# ============================================================

dx = float(dickey.geometry.x.iloc[0])
dy = float(dickey.geometry.y.iloc[0])

fx = float(fort_kent.geometry.x.iloc[0])
fy = float(fort_kent.geometry.y.iloc[0])


ax.annotate(
    "Dickey\nUSGS 01010500\nCOMID 4288603",
    xy=(dx, dy),
    xytext=(8, -35),
    textcoords="offset points",
    fontsize=10,
)

ax.annotate(
    "Fort Kent\nUSGS 01014000\nCOMID 4287975",
    xy=(fx, fy),
    xytext=(8, 10),
    textcoords="offset points",
    fontsize=10,
)


# ============================================================
# 6. FORMATTING
# ============================================================

ax.set_title(
    "Upper Saint John River\n"
    "Dickey → Fort Kent Routing Study Area",
    fontsize=15,
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.grid(alpha=0.25)

plt.tight_layout()


# ============================================================
# 7. SAVE
# ============================================================

output_file = OUTPUT_DIR / "UpperSaintJohn_study_reach.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

print("\nSaved map:")
print(output_file)

plt.show()
