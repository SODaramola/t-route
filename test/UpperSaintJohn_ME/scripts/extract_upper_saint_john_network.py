# ============================================================
# EXTRACT UPPER SAINT JOHN NWM V3.0 ROUTING NETWORK
#
# Study reach:
#
#   Dickey, Maine
#   USGS 01010500
#   NWM feature_id = 4288603
#
#           ↓
#      Saint John River
#           ↓
#
#   Fort Kent, Maine
#   USGS 01014000
#   NWM feature_id = 4287759
#
#
# This script:
#
#   1. Reads the full NWM v3.0 RouteLink
#   2. Traces the mainstem from Dickey to Fort Kent
#   3. Finds all tributary networks entering between
#      Dickey and Fort Kent
#   4. Stops upstream traversal at Dickey so that the
#      basin upstream of the Dickey boundary is excluded
#   5. Saves:
#
#      - mainstem reach table
#      - full incremental routing network
#      - direct tributary mouths
#      - raw RouteLink NetCDF subset
#
# IMPORTANT:
# The raw NetCDF subset is NOT YET the final t-route
# RouteLink file. Fort Kent still retains its original
# downstream link. We will prepare the final t-route
# domain after checking the extracted topology.
#
# ============================================================


# ============================================================
# 1. IMPORT PACKAGES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOMAIN_DIR = BASE_DIR / "domain"

ROUTELINK_FILE = (
    DOMAIN_DIR
    / "RouteLink_CONUS_v3.0.nc"
)


# ============================================================
# 3. STUDY BOUNDARIES
# ============================================================

DICKEY = 4288603

FORT_KENT = 4287759


# ============================================================
# 4. OPEN ROUTELINK
# ============================================================

print("=" * 80)
print("OPENING NWM V3.0 ROUTELINK")
print("=" * 80)

ds = xr.open_dataset(
    ROUTELINK_FILE
)

print(
    "Number of NWM reaches:",
    ds.sizes["feature_id"],
)


# ============================================================
# 5. LOAD NETWORK TOPOLOGY
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
# 6. CREATE FAST LINK -> ARRAY INDEX LOOKUP
#
# Avoid creating a huge Python dictionary for 2.7 million
# reaches. Instead, sort the link IDs once and use binary
# search.
# ============================================================

print("\nBuilding link lookup...")

link_sort_order = np.argsort(
    links
)

sorted_links = links[
    link_sort_order
]


def get_index(feature_id):
    """
    Return the RouteLink array index corresponding to an
    NWM feature_id.
    """

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


print("Link lookup ready.")


# ============================================================
# 7. VERIFY DICKEY AND FORT KENT
# ============================================================

dickey_index = get_index(
    DICKEY
)

fort_kent_index = get_index(
    FORT_KENT
)


if dickey_index is None:

    raise RuntimeError(
        f"Dickey feature_id {DICKEY} not found."
    )


if fort_kent_index is None:

    raise RuntimeError(
        f"Fort Kent feature_id {FORT_KENT} not found."
    )


print("\nVerified study boundaries:")

print(
    "Dickey    :",
    DICKEY,
)

print(
    "Fort Kent :",
    FORT_KENT,
)


# ============================================================
# 8. TRACE MAINSTEM FROM DICKEY TO FORT KENT
# ============================================================

print("\n" + "=" * 80)
print("TRACING DICKEY -> FORT KENT MAINSTEM")
print("=" * 80)


mainstem = []

visited_mainstem = set()

current = DICKEY

MAX_STEPS = 10000


for step in range(MAX_STEPS):

    if current in visited_mainstem:

        raise RuntimeError(
            "Loop detected while tracing mainstem "
            f"at feature_id {current}."
        )

    visited_mainstem.add(
        current
    )

    mainstem.append(
        current
    )

    if current == FORT_KENT:

        break

    index = get_index(
        current
    )

    if index is None:

        raise RuntimeError(
            f"Mainstem feature {current} "
            "not found in RouteLink."
        )

    next_feature = int(
        downstream[index]
    )

    if next_feature in (
        0,
        -9999,
    ):

        raise RuntimeError(
            "Reached network terminal before "
            "Fort Kent.\n"
            f"Current feature: {current}\n"
            f"Downstream: {next_feature}"
        )

    current = next_feature


else:

    raise RuntimeError(
        "Maximum mainstem traversal steps exceeded."
    )


if mainstem[-1] != FORT_KENT:

    raise RuntimeError(
        "Fort Kent was not reached from Dickey."
    )


print(
    "SUCCESS: Fort Kent reached from Dickey."
)

print(
    "Mainstem NWM reaches:",
    len(mainstem),
)


# ============================================================
# 9. BUILD FAST DOWNSTREAM -> UPSTREAM LOOKUP
#
# For each feature, we need to know which reaches flow into it.
#
# Again, use sorted NumPy arrays rather than a huge Python
# dictionary.
# ============================================================

print("\nBuilding reverse network lookup...")

to_sort_order = np.argsort(
    downstream
)

sorted_to = downstream[
    to_sort_order
]


def get_upstream_features(feature_id):
    """
    Return all NWM reaches whose downstream 'to' value is
    feature_id.
    """

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
# 10. EXTRACT INCREMENTAL NETWORK
#
# Work upstream from Fort Kent.
#
# IMPORTANT:
#
# When Dickey is reached, DO NOT continue upstream.
#
# Therefore:
#
# - Dickey becomes the upstream mainstem boundary
# - tributaries entering downstream of Dickey are retained
# - the enormous basin upstream of Dickey is excluded
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING INCREMENTAL DICKEY -> FORT KENT NETWORK")
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

    # --------------------------------------------------------
    # Stop upstream traversal at Dickey
    # --------------------------------------------------------

    if feature_id == DICKEY:

        continue

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
    "Total incremental network reaches:",
    len(network_features),
)


# ============================================================
# 11. VERIFY MAINSTEM IS INSIDE INCREMENTAL NETWORK
# ============================================================

missing_mainstem = (
    set(mainstem)
    - network_features
)


if missing_mainstem:

    raise RuntimeError(
        "Some mainstem features are missing from "
        "the incremental network:\n"
        f"{sorted(missing_mainstem)}"
    )


print(
    "All mainstem reaches are contained "
    "in incremental network."
)


# ============================================================
# 12. IDENTIFY DIRECT TRIBUTARY MOUTHS
#
# These are non-mainstem reaches whose downstream 'to'
# feature is a mainstem reach.
# ============================================================

mainstem_set = set(
    mainstem
)


tributary_mouths = []


for feature_id in network_features:

    if feature_id in mainstem_set:

        continue

    index = get_index(
        feature_id
    )

    if index is None:

        continue

    to_feature = int(
        downstream[index]
    )

    if to_feature in mainstem_set:

        tributary_mouths.append(
            feature_id
        )


tributary_mouths = sorted(
    tributary_mouths
)


print(
    "Direct tributary mouths:",
    len(tributary_mouths),
)


# ============================================================
# 13. VERIFY NETWORK CONNECTIVITY
#
# Every selected feature should flow to another selected
# feature except Fort Kent, which exits the study domain.
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


if broken_connections:

    print(
        "\nWARNING: Broken downstream connections:"
    )

    for feature_id, to_feature in broken_connections[:20]:

        print(
            feature_id,
            "->",
            to_feature,
        )

else:

    print(
        "Network connectivity check: PASSED"
    )


# ============================================================
# 14. HELPER FOR BUILDING TABLES
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


def build_table(feature_ids):

    indices = []

    for feature_id in feature_ids:

        index = get_index(
            int(feature_id)
        )

        if index is None:

            raise RuntimeError(
                f"Feature {feature_id} "
                "was not found."
            )

        indices.append(
            index
        )

    indices = np.asarray(
        indices,
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

        # ----------------------------------------------------
        # Decode fixed-width byte strings such as gage IDs
        # ----------------------------------------------------

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

    return pd.DataFrame(
        data
    )


# ============================================================
# 15. MAINSTEM TABLE
# ============================================================

mainstem_df = build_table(
    mainstem
)

mainstem_df.insert(
    0,
    "mainstem_sequence",
    np.arange(
        1,
        len(mainstem_df) + 1,
    ),
)


# ============================================================
# 16. CALCULATE MAINSTEM LENGTH
# ============================================================

mainstem_length_km = (
    mainstem_df["Length"].sum()
    / 1000.0
)


print(
    "\nMainstem modeled length:",
    f"{mainstem_length_km:.2f} km",
)


# ============================================================
# 17. FULL NETWORK TABLE
# ============================================================

network_df = build_table(
    sorted(network_features)
)


# ============================================================
# 18. ADD NETWORK ROLE
# ============================================================

network_df["network_role"] = "tributary"

network_df.loc[
    network_df["link"].isin(
        mainstem_set
    ),
    "network_role",
] = "mainstem"


network_df.loc[
    network_df["link"] == DICKEY,
    "network_role",
] = "upstream_boundary"


network_df.loc[
    network_df["link"] == FORT_KENT,
    "network_role",
] = "outlet"


# ============================================================
# 19. SORT NETWORK BY NWM ASCENDING INDEX
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
# 20. TRIBUTARY MOUTH TABLE
# ============================================================

tributary_df = build_table(
    tributary_mouths
)


if not tributary_df.empty:

    tributary_df = (
        tributary_df
        .sort_values(
            "link"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# 21. SAVE CSV TABLES
# ============================================================

mainstem_csv = (
    DOMAIN_DIR
    / "Dickey_to_FortKent_mainstem.csv"
)

network_csv = (
    DOMAIN_DIR
    / "UpperSaintJohn_incremental_network.csv"
)

tributary_csv = (
    DOMAIN_DIR
    / "UpperSaintJohn_tributary_mouths.csv"
)


mainstem_df.to_csv(
    mainstem_csv,
    index=False,
)

network_df.to_csv(
    network_csv,
    index=False,
)

tributary_df.to_csv(
    tributary_csv,
    index=False,
)


# ============================================================
# 22. SAVE FEATURE ID LIST
# ============================================================

feature_id_file = (
    DOMAIN_DIR
    / "UpperSaintJohn_feature_ids.txt"
)


with open(
    feature_id_file,
    "w",
) as file:

    for feature_id in network_df["link"]:

        file.write(
            f"{int(feature_id)}\n"
        )


# ============================================================
# 23. CREATE RAW ROUTELINK NETCDF SUBSET
#
# IMPORTANT:
#
# This remains an unmodified subset of the NOAA RouteLink.
# It is NOT YET the final t-route domain.
# ============================================================

network_indices = []


for feature_id in network_df["link"]:

    index = get_index(
        int(feature_id)
    )

    network_indices.append(
        index
    )


network_indices = np.asarray(
    network_indices,
    dtype=np.int64,
)


subset = ds.isel(
    feature_id=network_indices
)


raw_subset_file = (
    DOMAIN_DIR
    / "RouteLink_UpperSaintJohn_raw.nc"
)


subset.to_netcdf(
    raw_subset_file,
)


# ============================================================
# 24. PRINT MAINSTEM
# ============================================================

print("\n" + "=" * 80)
print("DICKEY -> FORT KENT MAINSTEM")
print("=" * 80)


display_columns = [
    "mainstem_sequence",
    "link",
    "to",
    "Length",
    "order",
    "n",
    "So",
    "BtmWdth",
    "gages",
]


available_columns = [
    column
    for column in display_columns
    if column in mainstem_df.columns
]


print(
    mainstem_df[
        available_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 25. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("UPPER SAINT JOHN NETWORK SUMMARY")
print("=" * 80)

print(
    "Upstream boundary :",
    DICKEY,
)

print(
    "Outlet            :",
    FORT_KENT,
)

print(
    "Mainstem reaches  :",
    len(mainstem),
)

print(
    "Mainstem length   :",
    f"{mainstem_length_km:.2f} km",
)

print(
    "Total reaches     :",
    len(network_features),
)

print(
    "Tributary reaches :",
    len(network_features)
    - len(mainstem),
)

print(
    "Tributary mouths  :",
    len(tributary_mouths),
)

print(
    "Broken links      :",
    len(broken_connections),
)


# ============================================================
# 26. OUTPUT FILES
# ============================================================

print("\nSaved:")

print(mainstem_csv)
print(network_csv)
print(tributary_csv)
print(feature_id_file)
print(raw_subset_file)


# ============================================================
# 27. CLOSE DATASET
# ============================================================

ds.close()


# ============================================================
# 28. FINISHED
# ============================================================

print("\n" + "=" * 80)
print("NETWORK EXTRACTION COMPLETE")
print("=" * 80)
