# ============================================================
# FIND NWM V3 FEATURE IDs FROM USGS GAGE IDs
#
# Upper Saint John River
#
# Dickey    USGS 01010500
# Fort Kent USGS 01014000
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import s3fs
import xarray as xr


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOMAIN_DIR = BASE_DIR / "domain"

DOMAIN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. NWM V3 RETROSPECTIVE
# ============================================================

BUCKET = "noaa-nwm-retrospective-3-0-pds"

ZARR_PATH = (
    f"{BUCKET}/CONUS/zarr/chrtout.zarr"
)


# ============================================================
# 3. TARGET USGS GAGES
# ============================================================

TARGET_GAGES = {
    "Dickey": "01010500",
    "Fort_Kent": "01014000",
}


# ============================================================
# 4. CONNECT
# ============================================================

print("=" * 70)
print("OPENING NWM V3 CHRTOUT")
print("=" * 70)

fs = s3fs.S3FileSystem(
    anon=True
)

mapper = fs.get_mapper(
    ZARR_PATH
)

ds = xr.open_zarr(
    mapper,
    consolidated=True,
)

print("\nDataset opened successfully.")

print("\nFeature count:")
print(ds.sizes["feature_id"])


# ============================================================
# 5. LOAD NWM NETWORK COORDINATES
# ============================================================

print("\nLoading NWM network metadata...")

feature_ids = np.asarray(
    ds["feature_id"].values
)

gage_raw = np.asarray(
    ds["gage_id"].values
)

latitudes = np.asarray(
    ds["latitude"].values
)

longitudes = np.asarray(
    ds["longitude"].values
)

stream_order = np.asarray(
    ds["order"].values
)


# ============================================================
# 6. DECODE GAGE IDs
#
# NWM stores gage_id as fixed-width byte strings.
# ============================================================

if gage_raw.dtype.kind == "S":

    gage_ids = np.char.decode(
        gage_raw,
        "utf-8",
        errors="ignore",
    )

else:

    gage_ids = gage_raw.astype("U")


gage_ids = np.char.strip(
    gage_ids
)


# ============================================================
# 7. FIND TARGET GAGES
# ============================================================

records = []

print("\n" + "=" * 70)
print("NWM V3 GAGE LOOKUP")
print("=" * 70)


for location, target_gage in TARGET_GAGES.items():

    indices = np.where(
        gage_ids == target_gage
    )[0]

    print(f"\n{location}")
    print("-" * 50)

    print(
        "USGS gage ID       :",
        target_gage,
    )

    if len(indices) == 0:

        print(
            "NWM feature       : NOT FOUND"
        )

        records.append(
            {
                "location": location,
                "usgs_gage": target_gage,
                "nwm_feature_id": None,
                "latitude": None,
                "longitude": None,
                "stream_order": None,
            }
        )

        continue


    for index in indices:

        feature_id = int(
            feature_ids[index]
        )

        latitude = float(
            latitudes[index]
        )

        longitude = float(
            longitudes[index]
        )

        order = int(
            stream_order[index]
        )

        print(
            "NWM feature_id    :",
            feature_id,
        )

        print(
            "Latitude          :",
            latitude,
        )

        print(
            "Longitude         :",
            longitude,
        )

        print(
            "Stream order      :",
            order,
        )

        records.append(
            {
                "location": location,
                "usgs_gage": target_gage,
                "nwm_feature_id": feature_id,
                "latitude": latitude,
                "longitude": longitude,
                "stream_order": order,
            }
        )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(
    records
)

output_file = (
    DOMAIN_DIR
    / "nwm_v3_gage_feature_ids.csv"
)

result_df.to_csv(
    output_file,
    index=False,
)


print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    result_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(output_file)


# ============================================================
# 9. CLOSE DATASET
# ============================================================

ds.close()

print("\nDONE")
