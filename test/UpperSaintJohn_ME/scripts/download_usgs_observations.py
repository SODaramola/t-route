# ============================================================
# DOWNLOAD USGS OBSERVED STREAMFLOW
#
# Upper Saint John River
#
# Stations:
#   Dickey    : USGS 01010500
#   Fort Kent : USGS 01014000
#
# Period:
#   2016-11-01 00:00 UTC
#       through
#   2016-11-03 00:00 UTC
#
# Parameter:
#   00060 = discharge
#
# Source:
#   USGS Water Data OGC API
# ============================================================

from pathlib import Path

import pandas as pd
import requests


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    BASE_DIR
    / "obs"
    / "USGS_Dickey_FortKent_20161101_20161103.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. USGS SETTINGS
# ============================================================

API_URL = (
    "https://api.waterdata.usgs.gov/"
    "ogcapi/v0/collections/continuous/items"
)

PARAMETER_CODE = "00060"

START_TIME = "2016-11-01T00:00:00Z"
END_TIME = "2016-11-03T00:00:00Z"

SITES = {
    "Dickey": "01010500",
    "Fort_Kent": "01014000",
}

# Exact conversion
CFS_TO_CMS = 0.028316846592


# ============================================================
# 3. DOWNLOAD FUNCTION
# ============================================================

def download_site(location, site_no):

    monitoring_location_id = f"USGS-{site_no}"

    params = {
        "f": "json",
        "monitoring_location_id": monitoring_location_id,
        "parameter_code": PARAMETER_CODE,
        "datetime": f"{START_TIME}/{END_TIME}",
        "limit": 10000,
    }

    print()
    print("=" * 80)
    print(f"DOWNLOADING {location}")
    print("=" * 80)

    print("USGS station:", site_no)

    response = requests.get(
        API_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    features = data.get(
        "features",
        [],
    )

    if not features:
        raise RuntimeError(
            f"No USGS observations returned for {site_no}"
        )

    rows = []

    for feature in features:

        p = feature["properties"]

        rows.append(
            {
                "time": p.get("time"),
                "location": location,
                "site_no": site_no,
                "monitoring_location_id":
                    p.get("monitoring_location_id"),
                "time_series_id":
                    p.get("time_series_id"),
                "parameter_code":
                    p.get("parameter_code"),
                "value_original":
                    p.get("value"),
                "unit_original":
                    p.get("unit_of_measure"),
                "approval_status":
                    p.get("approval_status"),
                "qualifier":
                    p.get("qualifier"),
            }
        )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Parse time
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
    )

    # Convert to UTC-naive timestamps so they align directly
    # with the existing t-route/NWM timestamps.
    df["time"] = (
        df["time"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    # --------------------------------------------------------
    # Numeric discharge
    # --------------------------------------------------------

    df["value_original"] = pd.to_numeric(
        df["value_original"],
        errors="coerce",
    )

    if df["value_original"].isna().any():

        bad = df[
            df["value_original"].isna()
        ]

        print("\nWARNING:")
        print(
            "Non-numeric observations:",
            len(bad),
        )

    # --------------------------------------------------------
    # Convert discharge to m3/s
    # --------------------------------------------------------

    units = (
        df["unit_original"]
        .dropna()
        .astype(str)
        .unique()
    )

    print("Returned units:", units)

    if len(units) != 1:

        raise RuntimeError(
            f"Unexpected USGS units: {units}"
        )

    unit = units[0].lower()

    if (
        "ft3/s" in unit
        or "ft^3/s" in unit
        or "cubic feet" in unit
    ):

        df["streamflow_cms"] = (
            df["value_original"]
            * CFS_TO_CMS
        )

    elif (
        "m3/s" in unit
        or "m^3/s" in unit
    ):

        df["streamflow_cms"] = (
            df["value_original"]
        )

    else:

        raise RuntimeError(
            "Unrecognized discharge unit: "
            f"{units[0]}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    print("Observations:", len(df))
    print(
        "First time:",
        df["time"].min(),
    )
    print(
        "Last time :",
        df["time"].max(),
    )

    print(
        "Minimum Q:",
        df["streamflow_cms"].min(),
        "m3/s",
    )

    print(
        "Mean Q   :",
        df["streamflow_cms"].mean(),
        "m3/s",
    )

    print(
        "Maximum Q:",
        df["streamflow_cms"].max(),
        "m3/s",
    )

    print(
        "Time-series IDs:",
        df["time_series_id"]
        .dropna()
        .unique(),
    )

    print(
        "Approval status:",
        df["approval_status"]
        .astype(str)
        .unique(),
    )

    print(
        "Qualifiers:",
        df["qualifier"]
        .astype(str)
        .unique(),
    )

    # Duplicate timestamp check
    duplicate_times = (
        df["time"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate timestamps:",
        duplicate_times,
    )

    return df


# ============================================================
# 4. DOWNLOAD BOTH STATIONS
# ============================================================

frames = []

for location, site_no in SITES.items():

    frames.append(
        download_site(
            location,
            site_no,
        )
    )


# ============================================================
# 5. COMBINE
# ============================================================

obs = pd.concat(
    frames,
    ignore_index=True,
)

obs = obs.sort_values(
    [
        "location",
        "time",
    ]
).reset_index(drop=True)


# ============================================================
# 6. FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("COMBINED OBSERVATION SUMMARY")
print("=" * 80)

for location in SITES:

    sub = obs[
        obs["location"] == location
    ]

    print()
    print(location)
    print("-" * 40)

    print(
        "Records:",
        len(sub),
    )

    print(
        "First:",
        sub["time"].min(),
    )

    print(
        "Last :",
        sub["time"].max(),
    )


# ============================================================
# 7. SAVE
# ============================================================

obs.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("=" * 80)
print("USGS OBSERVATIONS SAVED")
print("=" * 80)

print(OUTPUT_FILE.resolve())
