# ============================================================
# TEST ACCESS TO NOAA NWM V3 RETROSPECTIVE
#
# Study area:
# Upper Saint John River, Maine
#
# Gages:
# Dickey
#   USGS 01010500
#   NWM v3 feature_id 4288603
#
# Fort Kent
#   USGS 01014000
#   NWM v3 feature_id 4287759
#
# This script:
#   1. Opens the NOAA NWM v3 retrospective Zarr dataset
#   2. Verifies that both NWM feature IDs exist
#   3. Verifies streamflow and q_lateral are available
#   4. Extracts a small 48-hour test period
#   5. Saves the extracted data to CSV
#
# Test period:
#   2016-11-01 00:00
#   through
#   2016-11-03 00:00
# ============================================================


# ============================================================
# 1. IMPORT PACKAGES
# ============================================================

from pathlib import Path

import pandas as pd
import s3fs
import xarray as xr


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FORCING_DIR = BASE_DIR / "forcing"

FORCING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. NWM V3 RETROSPECTIVE ZARR LOCATION
# ============================================================

BUCKET = "noaa-nwm-retrospective-3-0-pds"

ZARR_PATH = (
    f"{BUCKET}/CONUS/zarr/chrtout.zarr"
)


# ============================================================
# 4. VERIFIED NWM V3 FEATURE IDS
#
# IMPORTANT:
#
# Fort Kent NLDI COMID = 4287975
#
# But the correct NWM v3 feature_id is:
#
# Fort Kent NWM v3 = 4287759
#
# For NWM/t-route work use 4287759.
# ============================================================

gage_ids = {
    "Dickey": 4288603,
    "Fort_Kent": 4287759,
}


# ============================================================
# 5. CONNECT TO PUBLIC NOAA S3 BUCKET
# ============================================================

print("=" * 70)
print("CONNECTING TO NWM V3 RETROSPECTIVE")
print("=" * 70)

fs = s3fs.S3FileSystem(
    anon=True
)

mapper = fs.get_mapper(
    ZARR_PATH
)


# ============================================================
# 6. OPEN NWM V3 ZARR DATASET
# ============================================================

try:

    ds = xr.open_zarr(
        mapper,
        consolidated=True,
    )

except Exception as exc:

    print("\nConsolidated metadata failed.")

    print("Reason:")
    print(exc)

    print(
        "\nTrying unconsolidated "
        "Zarr metadata..."
    )

    ds = xr.open_zarr(
        mapper,
        consolidated=False,
    )


print("\nDataset opened successfully.")


# ============================================================
# 7. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDimensions:")
print(ds.sizes)

print("\nCoordinates:")
print(list(ds.coords))

print("\nVariables:")
print(list(ds.data_vars))


# ============================================================
# 8. VERIFY REQUIRED VARIABLES
# ============================================================

required_variables = [
    "streamflow",
    "q_lateral",
]


print("\nChecking required variables...")

for variable in required_variables:

    if variable not in ds.data_vars:

        raise KeyError(
            f"Required NWM variable "
            f"not found: {variable}"
        )

    print(
        f"{variable:15s}: FOUND"
    )


# ============================================================
# 9. VERIFY FEATURE_ID COORDINATE
# ============================================================

if "feature_id" not in ds.coords:

    raise KeyError(
        "feature_id coordinate "
        "not found in NWM dataset."
    )


# ============================================================
# 10. VERIFY STUDY REACH IDS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING UPPER SAINT JOHN FEATURE IDS")
print("=" * 70)

valid_ids = []


for name, feature_id in gage_ids.items():

    try:

        # Select only the feature_id.
        # Do not use method='nearest'.
        ds.sel(
            feature_id=feature_id
        )

        print(
            f"{name:10s}: "
            f"feature_id {feature_id} FOUND"
        )

        valid_ids.append(
            feature_id
        )

    except Exception as exc:

        print(
            f"{name:10s}: "
            f"feature_id {feature_id} NOT FOUND"
        )

        print(exc)


# ============================================================
# 11. STOP IF ANY FEATURE ID IS MISSING
# ============================================================

if len(valid_ids) != len(gage_ids):

    raise RuntimeError(
        "One or more Upper Saint John "
        "NWM v3 feature IDs were not found. "
        "Stop before extracting forcing."
    )


print(
    "\nAll required Upper Saint John "
    "feature IDs were found."
)


# ============================================================
# 12. DEFINE SMALL 48-HOUR TEST PERIOD
# ============================================================

start_time = "2016-11-01 00:00:00"

end_time = "2016-11-03 00:00:00"


print("\n" + "=" * 70)
print("EXTRACTING 48-HOUR NWM TEST")
print("=" * 70)

print(
    "Start:",
    start_time,
)

print(
    "End  :",
    end_time,
)


# ============================================================
# 13. SELECT VARIABLES
# ============================================================

variables = [
    "streamflow",
    "q_lateral",
]


# Include velocity if available
if "velocity" in ds.data_vars:

    variables.append(
        "velocity"
    )


print("\nVariables to extract:")

for variable in variables:

    print(
        "  -",
        variable,
    )


# ============================================================
# 14. EXTRACT THE TWO STUDY LOCATIONS
# ============================================================

subset = (
    ds[variables]
    .sel(
        time=slice(
            start_time,
            end_time,
        ),
        feature_id=valid_ids,
    )
)


print("\nSubset structure:")
print(subset)


# ============================================================
# 15. DOWNLOAD ONLY THE SMALL SUBSET
# ============================================================

print(
    "\nDownloading 48-hour subset..."
)

subset = subset.load()

print(
    "Download complete."
)


# ============================================================
# 16. CONVERT TO PANDAS DATAFRAME
# ============================================================

df = (
    subset
    .to_dataframe()
    .reset_index()
)


# ============================================================
# 17. ADD LOCATION NAMES
# ============================================================

name_lookup = {
    4288603: "Dickey",
    4287759: "Fort_Kent",
}


df["location"] = (
    df["feature_id"]
    .map(name_lookup)
)


# ============================================================
# 18. REORDER COLUMNS
# ============================================================

preferred_columns = [
    "time",
    "location",
    "feature_id",
    "streamflow",
    "q_lateral",
]


if "velocity" in df.columns:

    preferred_columns.append(
        "velocity"
    )


other_columns = [
    column
    for column in df.columns
    if column not in preferred_columns
]


df = df[
    preferred_columns
    + other_columns
]


# ============================================================
# 19. SAVE CSV
# ============================================================

output_file = (
    FORCING_DIR
    / "NWM_v3_Dickey_FortKent_20161101_20161103.csv"
)


df.to_csv(
    output_file,
    index=False,
)


print("\nSaved:")
print(output_file)


# ============================================================
# 20. DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("NWM V3 SAMPLE")
print("=" * 70)


print(
    df.head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. DISPLAY DATA COUNTS
# ============================================================

print("\nNumber of rows:")
print(len(df))


print("\nRows by location:")

print(
    df["location"]
    .value_counts()
)


# ============================================================
# 22. DISPLAY STREAMFLOW SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STREAMFLOW SUMMARY")
print("=" * 70)


streamflow_summary = (
    df.groupby("location")
    ["streamflow"]
    .agg(
        [
            "count",
            "min",
            "mean",
            "max",
        ]
    )
)


print(
    streamflow_summary
    .to_string()
)


# ============================================================
# 23. DISPLAY Q_LATERAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("Q_LATERAL SUMMARY")
print("=" * 70)


qlateral_summary = (
    df.groupby("location")
    ["q_lateral"]
    .agg(
        [
            "count",
            "min",
            "mean",
            "max",
        ]
    )
)


print(
    qlateral_summary
    .to_string()
)


# ============================================================
# 24. CLOSE DATASET
# ============================================================

ds.close()


# ============================================================
# 25. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("NWM V3 ACCESS TEST SUCCESSFUL")
print("=" * 70)

print("\nVerified NWM v3 feature IDs:")

print(
    "Dickey    : 4288603"
)

print(
    "Fort Kent : 4287759"
)

print(
    "\nThe next step is to build the "
    "Dickey-to-Fort-Kent NWM/t-route network."
)