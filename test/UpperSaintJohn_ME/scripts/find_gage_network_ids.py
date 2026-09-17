# ============================================================
# IDENTIFY DICKEY AND FORT KENT ON THE USGS NLDI NETWORK
#
# Study:
# Integration of river ice information in hydrodynamic models
# for enhanced and continuous streamflow forecast
#
# Gages:
#   Dickey, Maine    USGS 01010500
#   Fort Kent, Maine USGS 01014000
#
# Outputs:
#   domain/gage_network_ids.csv
#   domain/Dickey_USGS_01010500.geojson
#   domain/Fort_Kent_USGS_01014000.geojson
#   domain/Dickey_downstream_150km.geojson
#
# ============================================================

from pathlib import Path

import pandas as pd
import requests


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DOMAIN_DIR = BASE_DIR / "domain"

DOMAIN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. USGS NLDI BASE URL
# ============================================================

NLDI = "https://api.water.usgs.gov/nldi/linked-data"


# ============================================================
# 3. STUDY GAGES
# ============================================================

sites = {
    "Dickey": "01010500",
    "Fort Kent": "01014000",
}


# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def get_json(url, params=None):

    print("\nRequesting:")
    print(url)

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    return response.json()


# ============================================================
# 5. QUERY EACH STREAMGAGE
# ============================================================

records = []

print("=" * 70)
print("UPPER SAINT JOHN RIVER")
print("USGS / NHDPLUS NETWORK IDENTIFICATION")
print("=" * 70)


for name, site_no in sites.items():

    feature_id = f"USGS-{site_no}"

    url = f"{NLDI}/nwissite/{feature_id}"

    data = get_json(
        url,
        params={"f": "json"},
    )

    if not data.get("features"):

        print(f"\nERROR: No NLDI feature returned for {name}")
        continue

    feature = data["features"][0]

    properties = feature.get("properties", {})
    geometry = feature.get("geometry", {})

    coordinates = geometry.get("coordinates", [None, None])

    lon = coordinates[0]
    lat = coordinates[1]

    comid = properties.get("comid")
    reachcode = properties.get("reachcode")
    measure = properties.get("measure")
    station_name = properties.get("name")
    identifier = properties.get("identifier")

    print("\n" + "-" * 70)
    print(name.upper())
    print("-" * 70)

    print("USGS station       :", site_no)
    print("Identifier         :", identifier)
    print("Station name       :", station_name)
    print("NHDPlus COMID      :", comid)
    print("Reach code         :", reachcode)
    print("Measure            :", measure)
    print("Longitude          :", lon)
    print("Latitude           :", lat)

    records.append(
        {
            "location": name,
            "usgs_site": site_no,
            "identifier": identifier,
            "station_name": station_name,
            "comid": comid,
            "reachcode": reachcode,
            "measure": measure,
            "longitude": lon,
            "latitude": lat,
        }
    )

    # ------------------------------------------
    # Save original GeoJSON response
    # ------------------------------------------

    safe_name = name.replace(" ", "_")

    output_file = (
        DOMAIN_DIR
        / f"{safe_name}_USGS_{site_no}.geojson"
    )

    import json

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved             :", output_file)


# ============================================================
# 6. SAVE GAGE NETWORK TABLE
# ============================================================

gage_df = pd.DataFrame(records)

csv_file = DOMAIN_DIR / "gage_network_ids.csv"

gage_df.to_csv(
    csv_file,
    index=False,
)

print("\n" + "=" * 70)
print("GAGE NETWORK TABLE")
print("=" * 70)

print(gage_df.to_string(index=False))

print("\nSaved:")
print(csv_file)


# ============================================================
# 7. TEST DICKEY -> DOWNSTREAM NETWORK
#
# Retrieve downstream-mainstem NHDPlus flowlines starting
# from Dickey for 150 km.
# ============================================================

dickey_id = "USGS-01010500"

navigation_url = (
    f"{NLDI}/nwissite/{dickey_id}"
    "/navigation/DM/flowlines"
)

downstream = get_json(
    navigation_url,
    params={
        "f": "json",
        "distance": 150,
        "trimStart": "true",
    },
)


downstream_file = (
    DOMAIN_DIR
    / "Dickey_downstream_150km.geojson"
)

import json

with open(downstream_file, "w") as f:
    json.dump(downstream, f, indent=2)

print("\nSaved downstream flowlines:")
print(downstream_file)

print(
    "\nNumber of downstream flowline features:",
    len(downstream.get("features", [])),
)


# ============================================================
# 8. CHECK FOR DOWNSTREAM NWIS GAGES
#
# This is an independent connectivity check. Fort Kent should
# occur downstream from Dickey if both gages are indexed to the
# expected Saint John River mainstem.
# ============================================================

gage_navigation_url = (
    f"{NLDI}/nwissite/{dickey_id}"
    "/navigation/DM/nwissite"
)

downstream_gages = get_json(
    gage_navigation_url,
    params={
        "f": "json",
        "distance": 150,
    },
)

print("\n" + "=" * 70)
print("DOWNSTREAM NWIS SITES FROM DICKEY")
print("=" * 70)

fort_kent_found = False

for feature in downstream_gages.get("features", []):

    props = feature.get("properties", {})

    identifier = str(
        props.get("identifier", "")
    )

    name = props.get("name", "")

    print(identifier, "-", name)

    if "01014000" in identifier:

        fort_kent_found = True


print("\n" + "=" * 70)

if fort_kent_found:

    print(
        "SUCCESS: Fort Kent (01014000) was found "
        "downstream of Dickey."
    )

else:

    print(
        "WARNING: Fort Kent was not found within "
        "the 150-km downstream search."
    )

print("=" * 70)


# ============================================================
# 9. FINISHED
# ============================================================

print("\nNETWORK IDENTIFICATION COMPLETE")
