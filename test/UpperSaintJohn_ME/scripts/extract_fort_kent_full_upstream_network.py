# ============================================================
# EXTRACT FULL NWM V3.0 UPSTREAM NETWORK TO FORT KENT
#
# Outlet:
#   Fort Kent, Maine
#   USGS 01014000
#   NWM feature_id = 4287759
#
# Purpose:
#   Build a stock t-route Muskingum-Cunge test domain without
#   requiring an externally imposed time-varying upstream
#   boundary at Dickey.
#
# The script recursively traces every NWM reach upstream of
# Fort Kent.
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOMAIN_DIR = BASE_DIR / "domain"

ROUTELINK_FILE = (
    DOMAIN_DIR
    / "RouteLink_CONUS_v3.0.nc"
)


# ============================================================
# 2. IMPORTANT FEATURE IDS
# ============================================================

DICKEY = 4288603

FORT_KENT = 4287759


# ============================================================
# 3. OPEN FULL NWM ROUTELINK
# ============================================================

print("=" * 80)
print("OPENING NWM V3.0 ROUTELINK")
print("=" * 80)

ds = xr.open_dataset(
    ROUTELINK_FILE
)

print(
    "Total NWM reaches:",
    ds.sizes["feature_id"],
)


# ============================================================
# 4. LOAD TOPOLOGY
# ============================================================

print("\nLoading network topology...")

links = np.asarray(
    ds["link"].values,
    dtype=np.int64,
)

downstream = np.asarray(
    ds["to"].values,
    dtype=np.int64,
)

print("Topology loaded.")


# ============================================================
# 5. BUILD LINK -> INDEX LOOKUP
# ============================================================

print("\nBuilding feature lookup...")

link_sort_order = np.argsort(
    links
)

sorted_links = links[
    link_sort_order
]


def get_index(feature_id):

    position = np.searchsorted(
        sorted_links,
        feature_id,
    )

    if position >= len(sorted_links):

        return None

    if sorted_links[position] != feature_id:

        return None

    return int(
        link_sort_order[position]
    )


print("Feature lookup ready.")


# ============================================================
# 6. VERIFY IMPORTANT FEATURES
# ============================================================

for name, feature_id in {
    "Dickey": DICKEY,
    "Fort Kent": FORT_KENT,
}.items():

    index = get_index(
        feature_id
    )

    if index is None:

        raise RuntimeError(
            f"{name} feature_id "
            f"{feature_id} not found."
        )

    print(
        f"{name:10s}: "
        f"{feature_id} FOUND"
    )


# ============================================================
# 7. BUILD DOWNSTREAM -> UPSTREAM LOOKUP
# ============================================================

print("\nBuilding reverse network lookup...")

to_sort_order = np.argsort(
    downstream
)

sorted_to = downstream[
    to_sort_order
]


def get_upstream_features(feature_id):

    left = np.searchsorted(
        sorted_to,
        feature_id,
        side="left",
    )

    right = np.searchsorted(
        sorted_to,
        feature_id,
        side="right",
    )

    indices = to_sort_order[
        left:right
    ]

    return links[
        indices
    ]


print("Reverse lookup ready.")


# ============================================================
# 8. RECURSIVELY EXTRACT EVERYTHING UPSTREAM OF FORT KENT
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING FULL UPSTREAM NETWORK TO FORT KENT")
print("=" * 80)

network_features = set()

stack = [
    FORT_KENT
]


while stack:

    feature_id = int(
        stack.pop()
    )

    if feature_id in network_features:

        continue

    network_features.add(
        feature_id
    )

    upstream_features = (
        get_upstream_features(
            feature_id
        )
    )

    for upstream_feature in upstream_features:

        upstream_feature = int(
            upstream_feature
        )

        if upstream_feature not in network_features:

            stack.append(
                upstream_feature
            )


print(
    "Upstream network reaches:",
    len(network_features),
)


# ============================================================
# 9. VERIFY DICKEY IS INCLUDED
# ============================================================

if DICKEY in network_features:

    print(
        "SUCCESS: Dickey is contained "
        "in the Fort Kent upstream network."
    )

else:

    raise RuntimeError(
        "Dickey is NOT contained in the "
        "Fort Kent upstream network."
    )


# ============================================================
# 10. CHECK NETWORK CONNECTIVITY
# ============================================================

broken_connections = []


for feature_id in network_features:

    if feature_id == FORT_KENT:

        continue

    index = get_index(
        feature_id
    )

    to_feature = int(
        downstream[index]
    )

    if to_feature not in network_features:

        broken_connections.append(
            (
                feature_id,
                to_feature,
            )
        )


print(
    "Broken downstream connections:",
    len(broken_connections),
)


if broken_connections:

    print("\nFirst broken connections:")

    for pair in broken_connections[:20]:

        print(
            pair[0],
            "->",
            pair[1],
        )


# ============================================================
# 11. FIND HEADWATER REACHES
# ============================================================

headwaters = []


for feature_id in network_features:

    upstream = get_upstream_features(
        feature_id
    )

    upstream_in_network = [
        int(x)
        for x in upstream
        if int(x) in network_features
    ]

    if len(upstream_in_network) == 0:

        headwaters.append(
            feature_id
        )


print(
    "Headwater reaches:",
    len(headwaters),
)


# ============================================================
# 12. BUILD TABLE
# ============================================================

variables_to_save = [
    "link",
    "from",
    "to",
    "lon",
    "lat",
    "alt",
    "order",
    "Qi",
    "MusK",
    "MusX",
    "Length",
    "n",
    "So",
    "ChSlp",
    "BtmWdth",
    "NHDWaterbodyComID",
    "gages",
    "Kchan",
    "ascendingIndex",
    "nCC",
    "TopWdthCC",
    "TopWdth",
]


feature_ids = sorted(
    network_features
)


indices = np.asarray(
    [
        get_index(feature_id)
        for feature_id in feature_ids
    ],
    dtype=np.int64,
)


data = {}


for variable in variables_to_save:

    if variable not in ds.variables:

        continue

    values = ds[
        variable
    ].values[
        indices
    ]

    if values.dtype.kind == "S":

        values = np.char.decode(
            values,
            "utf-8",
            errors="ignore",
        )

        values = np.char.strip(
            values
        )

    data[variable] = values


network_df = pd.DataFrame(
    data
)


# ============================================================
# 13. SORT USING NWM ROUTING ORDER
# ============================================================

if "ascendingIndex" in network_df.columns:

    network_df = (
        network_df
        .sort_values(
            "ascendingIndex"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 14. NETWORK ROLE
# ============================================================

network_df["network_role"] = "channel"

network_df.loc[
    network_df["link"] == DICKEY,
    "network_role",
] = "Dickey_gage"


network_df.loc[
    network_df["link"] == FORT_KENT,
    "network_role",
] = "Fort_Kent_outlet"


network_df.loc[
    network_df["link"].isin(
        headwaters
    ),
    "network_role",
] = "headwater"


# Restore important labels if one happens to be a headwater
network_df.loc[
    network_df["link"] == DICKEY,
    "network_role",
] = "Dickey_gage"

network_df.loc[
    network_df["link"] == FORT_KENT,
    "network_role",
] = "Fort_Kent_outlet"


# ============================================================
# 15. BASIC STATISTICS
# ============================================================

total_length_km = (
    network_df["Length"].sum()
    / 1000.0
)


waterbody_mask = (
    network_df["NHDWaterbodyComID"]
    .fillna(-9999)
    .astype(np.int64)
    != -9999
)


waterbody_reaches = int(
    waterbody_mask.sum()
)


gage_strings = (
    network_df["gages"]
    .astype(str)
    .str.strip()
)


gage_count = int(
    (gage_strings != "").sum()
)


# ============================================================
# 16. SAVE CSV
# ============================================================

csv_file = (
    DOMAIN_DIR
    / "FortKent_full_upstream_network.csv"
)


network_df.to_csv(
    csv_file,
    index=False,
)


# ============================================================
# 17. SAVE FEATURE IDS
# ============================================================

feature_file = (
    DOMAIN_DIR
    / "FortKent_full_upstream_feature_ids.txt"
)


with open(
    feature_file,
    "w",
) as file:

    for feature_id in network_df["link"]:

        file.write(
            f"{int(feature_id)}\n"
        )


# ============================================================
# 18. SAVE HEADWATER IDS
# ============================================================

headwater_file = (
    DOMAIN_DIR
    / "FortKent_headwater_feature_ids.txt"
)


with open(
    headwater_file,
    "w",
) as file:

    for feature_id in sorted(
        headwaters
    ):

        file.write(
            f"{int(feature_id)}\n"
        )


# ============================================================
# 19. SAVE RAW NETCDF SUBSET
#
# Fort Kent retains its original downstream connection here.
# We will modify it only when creating the final t-route file.
# ============================================================

network_indices = np.asarray(
    [
        get_index(
            int(feature_id)
        )
        for feature_id
        in network_df["link"]
    ],
    dtype=np.int64,
)


subset = ds.isel(
    feature_id=network_indices
)


netcdf_file = (
    DOMAIN_DIR
    / "RouteLink_FortKent_full_upstream_raw.nc"
)


subset.to_netcdf(
    netcdf_file
)


# ============================================================
# 20. DISPLAY DICKEY AND FORT KENT
# ============================================================

print("\n" + "=" * 80)
print("DICKEY / FORT KENT")
print("=" * 80)


display_columns = [
    "link",
    "to",
    "Length",
    "order",
    "n",
    "So",
    "BtmWdth",
    "gages",
    "network_role",
]


print(
    network_df[
        network_df["link"].isin(
            [
                DICKEY,
                FORT_KENT,
            ]
        )
    ][
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 21. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FORT KENT FULL UPSTREAM NETWORK SUMMARY")
print("=" * 80)

print(
    "Outlet              :",
    FORT_KENT,
)

print(
    "Dickey included     :",
    DICKEY in network_features,
)

print(
    "Total reaches       :",
    len(network_features),
)

print(
    "Headwater reaches   :",
    len(headwaters),
)

print(
    "Total channel length:",
    f"{total_length_km:.2f} km",
)

print(
    "Waterbody reaches   :",
    waterbody_reaches,
)

print(
    "USGS-gaged reaches  :",
    gage_count,
)

print(
    "Broken connections  :",
    len(broken_connections),
)


# ============================================================
# 22. FILES
# ============================================================

print("\nSaved:")

print(csv_file)
print(feature_file)
print(headwater_file)
print(netcdf_file)


# ============================================================
# 23. CLOSE
# ============================================================

ds.close()


print("\n" + "=" * 80)
print("FULL UPSTREAM NETWORK EXTRACTION COMPLETE")
print("=" * 80)
