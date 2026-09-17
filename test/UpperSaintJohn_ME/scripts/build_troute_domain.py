# ============================================================
# BUILD T-ROUTE-READY UPPER SAINT JOHN DOMAIN
#
# Creates:
#
#   1. RouteLink_FortKent_full_upstream.nc
#   2. LAKEPARM_FortKent_full_upstream.nc
#
# The Fort Kent reach is converted to the terminal outlet by
# setting its downstream "to" value to 0.
#
# All other RouteLink parameters remain unchanged.
# ============================================================

from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DOMAIN_DIR = BASE_DIR / "domain"


RAW_ROUTE_FILE = (
    DOMAIN_DIR
    / "RouteLink_FortKent_full_upstream_raw.nc"
)


FULL_LAKE_FILE = (
    DOMAIN_DIR
    / "LAKEPARM_CONUS_v3.0.nc"
)


OUTPUT_ROUTE_FILE = (
    DOMAIN_DIR
    / "RouteLink_FortKent_full_upstream.nc"
)


OUTPUT_LAKE_FILE = (
    DOMAIN_DIR
    / "LAKEPARM_FortKent_full_upstream.nc"
)


# ============================================================
# 2. OUTLET
# ============================================================

FORT_KENT = 4287759


# ============================================================
# 3. OPEN ROUTELINK
# ============================================================

print("=" * 80)
print("BUILDING T-ROUTE ROUTELINK")
print("=" * 80)


route = xr.open_dataset(
    RAW_ROUTE_FILE
)

route.load()


links = np.asarray(
    route["link"].values
)


outlet_index = np.where(
    links == FORT_KENT
)[0]


if len(outlet_index) != 1:

    raise RuntimeError(
        "Expected exactly one Fort Kent reach, "
        f"found {len(outlet_index)}."
    )


outlet_index = int(
    outlet_index[0]
)


original_downstream = int(
    route["to"].values[
        outlet_index
    ]
)


print(
    "Fort Kent feature ID:",
    FORT_KENT,
)

print(
    "Original downstream:",
    original_downstream,
)


# ============================================================
# 4. MAKE FORT KENT TERMINAL
# ============================================================

route["to"].values[
    outlet_index
] = 0


new_downstream = int(
    route["to"].values[
        outlet_index
    ]
)


print(
    "New downstream     :",
    new_downstream,
)


# ============================================================
# 5. SAVE FINAL ROUTELINK
# ============================================================

route.to_netcdf(
    OUTPUT_ROUTE_FILE
)


print(
    "\nSaved RouteLink:"
)

print(
    OUTPUT_ROUTE_FILE
)


# ============================================================
# 6. IDENTIFY DOMAIN WATERBODIES
# ============================================================

waterbody_ids = np.unique(
    np.asarray(
        route[
            "NHDWaterbodyComID"
        ].values
    )
)


# Remove NWM no-waterbody value
waterbody_ids = waterbody_ids[
    waterbody_ids != -9999
]


print(
    "\nUnique domain waterbodies:",
    len(waterbody_ids),
)


# ============================================================
# 7. OPEN FULL LAKEPARM
# ============================================================

lake = xr.open_dataset(
    FULL_LAKE_FILE
)


lake_ids = np.asarray(
    lake["lake_id"].values
)


# ============================================================
# 8. FIND DOMAIN WATERBODIES
# ============================================================

mask = np.isin(
    lake_ids,
    waterbody_ids,
)


lake_indices = np.where(
    mask
)[0]


print(
    "Matched LAKEPARM records:",
    len(lake_indices),
)


missing = np.setdiff1d(
    waterbody_ids,
    lake_ids,
)


print(
    "Missing LAKEPARM records:",
    len(missing),
)


if len(missing) > 0:

    raise RuntimeError(
        "Some domain waterbodies are missing "
        "from LAKEPARM."
    )


# ============================================================
# 9. CREATE LAKEPARM SUBSET
# ============================================================

lake_subset = lake.isel(
    feature_id=lake_indices
)


lake_subset.load()


lake_subset.to_netcdf(
    OUTPUT_LAKE_FILE
)


print(
    "\nSaved LAKEPARM:"
)

print(
    OUTPUT_LAKE_FILE
)


# ============================================================
# 10. VALIDATE FINAL ROUTELINK
# ============================================================

final_links = set(
    int(x)
    for x in route[
        "link"
    ].values
)


broken_connections = []


for feature_id, downstream_id in zip(
    route["link"].values,
    route["to"].values,
):

    feature_id = int(
        feature_id
    )

    downstream_id = int(
        downstream_id
    )

    if feature_id == FORT_KENT:

        if downstream_id != 0:

            broken_connections.append(
                (
                    feature_id,
                    downstream_id,
                )
            )

        continue

    if downstream_id not in final_links:

        broken_connections.append(
            (
                feature_id,
                downstream_id,
            )
        )


# ============================================================
# 11. VALIDATE WATERBODY SUBSET
# ============================================================

saved_lake_ids = set(
    int(x)
    for x in lake_subset[
        "lake_id"
    ].values
)


missing_after_subset = (
    set(
        int(x)
        for x in waterbody_ids
    )
    - saved_lake_ids
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("T-ROUTE DOMAIN SUMMARY")
print("=" * 80)

print(
    "RouteLink reaches     :",
    route.sizes[
        "feature_id"
    ],
)

print(
    "Fort Kent outlet      :",
    FORT_KENT,
)

print(
    "Fort Kent downstream  :",
    int(
        route["to"].values[
            outlet_index
        ]
    ),
)

print(
    "Unique waterbodies    :",
    len(waterbody_ids),
)

print(
    "LAKEPARM records      :",
    lake_subset.sizes[
        "feature_id"
    ],
)

print(
    "Broken connections    :",
    len(
        broken_connections
    ),
)

print(
    "Missing lake records  :",
    len(
        missing_after_subset
    ),
)


if broken_connections:

    print(
        "\nBroken connections:"
    )

    for feature_id, downstream_id in broken_connections[:20]:

        print(
            feature_id,
            "->",
            downstream_id,
        )


# ============================================================
# 13. CLOSE
# ============================================================

route.close()

lake.close()

lake_subset.close()


# ============================================================
# 14. FINISHED
# ============================================================

if (
    len(broken_connections) == 0
    and len(missing_after_subset) == 0
):

    print("\n" + "=" * 80)
    print("T-ROUTE DOMAIN BUILD SUCCESSFUL")
    print("=" * 80)

else:

    raise RuntimeError(
        "Final domain validation failed."
    )
