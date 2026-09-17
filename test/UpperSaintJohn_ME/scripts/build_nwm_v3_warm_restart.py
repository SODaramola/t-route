# ============================================================
# BUILD NWM V3 WARM-START FILES FOR T-ROUTE
# UPPER SAINT JOHN / FORT KENT DOMAIN
#
# Initial time:
#     2016-11-01 00:00:00
#
# CHANNEL INITIAL CONDITIONS
# --------------------------
# Only ordinary Muskingum-Cunge channel reaches are included
# in the channel restart.
#
# RouteLink reaches with:
#
#     NHDWaterbodyComID != -9999
#
# are excluded from the MC channel restart because the
# t-route configuration will use:
#
#     break_network_at_waterbodies: True
#
# and those reaches are represented by level-pool waterbodies.
#
# For ordinary channels:
#
#     qu0 = NWM v3 streamflow
#     qd0 = NWM v3 streamflow
#     h0  = reconstructed from NWM v3 velocity using the
#           same Manning/trapezoidal hydraulic-radius
#           formulation used by the t-route/NWM
#           Muskingum-Cunge kernel.
#
#
# WATERBODY INITIAL CONDITIONS
# ----------------------------
#
# For each unique NHD waterbody:
#
#     qd0 = NWM v3 LAKEOUT outflow
#     h0  = NWM v3 LAKEOUT water_sfc_elev
#
#
# EXPECTED DOMAIN STRUCTURE
# -------------------------
#
# Total RouteLink reaches                  : 4392
# Waterbody-associated RouteLink reaches  : 365
# Ordinary MC channel reaches             : 4027
# Unique waterbodies                      : 35
#
#
# OUTPUT
# ------
#
# restart/channel_restart_201611010000
# restart/waterbody_restart_201611010000
#
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import s3fs


# ============================================================
# 1. PATHS AND CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]


DOMAIN_FILE = (
    BASE_DIR
    / "domain"
    / "RouteLink_FortKent_full_upstream.nc"
)


RESTART_DIR = (
    BASE_DIR
    / "restart"
)

RESTART_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHANNEL_RESTART_FILE = (
    RESTART_DIR
    / "channel_restart_201611010000"
)


WATERBODY_RESTART_FILE = (
    RESTART_DIR
    / "waterbody_restart_201611010000"
)


BUCKET = (
    "noaa-nwm-retrospective-3-0-pds"
)


CHRTOUT_STORE = (
    f"{BUCKET}/CONUS/zarr/chrtout.zarr"
)


LAKEOUT_STORE = (
    f"{BUCKET}/CONUS/zarr/lakeout.zarr"
)


T0 = pd.Timestamp(
    "2016-11-01 00:00:00"
)


DICKEY = 4288603

FORT_KENT = 4287759


# ============================================================
# 2. OPEN T-ROUTE ROUTELINK DOMAIN
# ============================================================

print("=" * 80)
print("READING T-ROUTE DOMAIN")
print("=" * 80)


route = xr.open_dataset(
    DOMAIN_FILE
)


feature_ids = np.asarray(
    route["link"].values,
    dtype=np.int64,
)


print(
    "Domain reaches:",
    len(feature_ids),
)


# ------------------------------------------------------------
# RouteLink hydraulic parameters
# ------------------------------------------------------------

mann_n = np.asarray(
    route["n"].values,
    dtype=np.float64,
)


slope = np.asarray(
    route["So"].values,
    dtype=np.float64,
)


bottom_width = np.asarray(
    route["BtmWdth"].values,
    dtype=np.float64,
)


channel_slope = np.asarray(
    route["ChSlp"].values,
    dtype=np.float64,
)


waterbody_raw = np.asarray(
    route["NHDWaterbodyComID"].values,
    dtype=np.int64,
)


# ============================================================
# 3. SEPARATE ORDINARY CHANNELS FROM WATERBODY REACHES
# ============================================================

# RouteLink uses -9999 for reaches that are not associated
# with NHD waterbodies.

channel_mask = (
    waterbody_raw == -9999
)


waterbody_reach_mask = (
    waterbody_raw != -9999
)


channel_feature_ids = (
    feature_ids[
        channel_mask
    ]
)


waterbody_ids = np.unique(
    waterbody_raw[
        waterbody_reach_mask
    ]
)


print(
    "Ordinary MC channel reaches:",
    len(channel_feature_ids),
)


print(
    "Waterbody-associated RouteLink reaches:",
    int(
        waterbody_reach_mask.sum()
    ),
)


print(
    "Unique waterbodies:",
    len(waterbody_ids),
)


# ------------------------------------------------------------
# Hydraulic parameters for ordinary MC channels only
# ------------------------------------------------------------

mann_n_channel = (
    mann_n[
        channel_mask
    ]
)


slope_channel = (
    slope[
        channel_mask
    ]
)


bottom_width_channel = (
    bottom_width[
        channel_mask
    ]
)


channel_slope_channel = (
    channel_slope[
        channel_mask
    ]
)


# ============================================================
# 4. OPEN NWM V3 CHRTOUT
# ============================================================

print("\n" + "=" * 80)
print("OPENING NWM V3 CHRTOUT")
print("=" * 80)


fs = s3fs.S3FileSystem(
    anon=True
)


chrt_mapper = fs.get_mapper(
    CHRTOUT_STORE
)


chrt = xr.open_zarr(
    chrt_mapper,
    consolidated=True,
)


# ============================================================
# 5. FIND EXACT CHRTOUT TIME
# ============================================================

chrt_times = pd.DatetimeIndex(
    chrt["time"].values
)


time_matches = np.where(
    chrt_times == T0
)[0]


if len(time_matches) != 1:

    raise RuntimeError(
        f"Expected exactly one CHRTOUT record at {T0}, "
        f"found {len(time_matches)}."
    )


time_index = int(
    time_matches[0]
)


print(
    "CHRTOUT time index:",
    time_index,
)


# ============================================================
# 6. MATCH ALL DOMAIN FEATURE IDs TO NWM FEATURE IDs
# ============================================================

print(
    "Loading NWM feature_id coordinate..."
)


nwm_feature_ids = np.asarray(
    chrt["feature_id"].values,
    dtype=np.int64,
)


feature_lookup = pd.Index(
    nwm_feature_ids
)


positions = (
    feature_lookup.get_indexer(
        feature_ids
    )
)


missing_mask = (
    positions < 0
)


if np.any(
    missing_mask
):

    missing_ids = (
        feature_ids[
            missing_mask
        ]
    )

    print(
        "\nMissing NWM feature IDs:"
    )

    print(
        missing_ids[:50]
    )

    raise RuntimeError(
        f"{len(missing_ids)} RouteLink reaches "
        "are missing from NWM v3 CHRTOUT."
    )


print(
    "Matched RouteLink reaches:",
    len(positions),
)


# ============================================================
# 7. LOAD NWM STREAMFLOW AND VELOCITY
# ============================================================

print(
    "\nDownloading NWM channel state..."
)


state = (
    chrt[
        [
            "streamflow",
            "velocity",
        ]
    ]
    .isel(
        time=time_index,
        feature_id=positions,
    )
    .load()
)


streamflow_all = np.asarray(
    state["streamflow"].values,
    dtype=np.float64,
)


velocity_all = np.asarray(
    state["velocity"].values,
    dtype=np.float64,
)


# ============================================================
# 8. VERIFY EXPECTED WATERBODY NaN STRUCTURE
# ============================================================

nonfinite_streamflow_all = (
    ~np.isfinite(
        streamflow_all
    )
)


nonfinite_channels = (
    nonfinite_streamflow_all
    & channel_mask
)


nonfinite_waterbody_reaches = (
    nonfinite_streamflow_all
    & waterbody_reach_mask
)


finite_waterbody_reaches = (
    np.isfinite(
        streamflow_all
    )
    & waterbody_reach_mask
)


print("\n" + "=" * 80)
print("ROUTELINK / CHRTOUT STATE DIAGNOSTIC")
print("=" * 80)


print(
    "Total non-finite streamflow reaches:",
    int(
        nonfinite_streamflow_all.sum()
    ),
)


print(
    "Non-finite ordinary MC reaches:",
    int(
        nonfinite_channels.sum()
    ),
)


print(
    "Non-finite waterbody-associated reaches:",
    int(
        nonfinite_waterbody_reaches.sum()
    ),
)


print(
    "Finite waterbody-associated reaches:",
    int(
        finite_waterbody_reaches.sum()
    ),
)


# No ordinary MC channel is allowed to have missing streamflow.

if np.any(
    nonfinite_channels
):

    bad_ids = (
        feature_ids[
            nonfinite_channels
        ]
    )

    print(
        "\nOrdinary channels with non-finite streamflow:"
    )

    print(
        bad_ids[:50]
    )

    raise RuntimeError(
        f"{len(bad_ids)} ordinary MC reaches "
        "have non-finite NWM streamflow."
    )


# ============================================================
# 9. VERIFY ONE FINITE REPRESENTATIVE REACH PER WATERBODY
# ============================================================

waterbody_check = pd.DataFrame(
    {
        "feature_id": (
            feature_ids[
                waterbody_reach_mask
            ]
        ),
        "waterbody_id": (
            waterbody_raw[
                waterbody_reach_mask
            ]
        ),
        "finite_streamflow": (
            np.isfinite(
                streamflow_all[
                    waterbody_reach_mask
                ]
            )
        ),
    }
)


waterbody_summary = (
    waterbody_check
    .groupby(
        "waterbody_id"
    )
    .agg(
        total_reaches=(
            "feature_id",
            "size",
        ),
        finite_reaches=(
            "finite_streamflow",
            "sum",
        ),
    )
)


bad_waterbody_representation = (
    waterbody_summary[
        "finite_reaches"
    ]
    != 1
)


if np.any(
    bad_waterbody_representation
):

    print(
        "\nUnexpected waterbody representation:"
    )

    print(
        waterbody_summary[
            bad_waterbody_representation
        ]
    )

    raise RuntimeError(
        "Expected exactly one finite CHRTOUT "
        "representative reach for every waterbody."
    )


print(
    "Waterbodies with exactly one finite "
    "representative reach:",
    len(
        waterbody_summary
    ),
)


# ============================================================
# 10. SUBSET TO ORDINARY MC CHANNELS
# ============================================================

streamflow = (
    streamflow_all[
        channel_mask
    ]
)


velocity = (
    velocity_all[
        channel_mask
    ]
)


feature_ids_channel = (
    feature_ids[
        channel_mask
    ]
)


print("\n" + "=" * 80)
print("ORDINARY MC CHANNEL STATE")
print("=" * 80)


print(
    "Ordinary MC reaches:",
    len(
        feature_ids_channel
    ),
)


print(
    "Finite streamflow states:",
    int(
        np.isfinite(
            streamflow
        ).sum()
    ),
)


# ============================================================
# 11. VALIDATE STREAMFLOW
# ============================================================

if np.any(
    ~np.isfinite(
        streamflow
    )
):

    bad = (
        feature_ids_channel[
            ~np.isfinite(
                streamflow
            )
        ]
    )

    raise RuntimeError(
        f"Non-finite streamflow found for "
        f"{len(bad)} ordinary MC reaches."
    )


if np.any(
    streamflow < 0.0
):

    bad = (
        feature_ids_channel[
            streamflow < 0.0
        ]
    )

    print(
        "First negative-streamflow reaches:"
    )

    print(
        bad[:50]
    )

    raise RuntimeError(
        f"Negative streamflow found for "
        f"{len(bad)} ordinary MC reaches."
    )


# ============================================================
# 12. CONVERT ROUTELINK ChSlp TO FORTRAN z
#
# In MCsingleSegStime_f2py_NOLOOP.f90:
#
#     if(cs == 0):
#         z = 1
#     else:
#         z = 1 / cs
#
# Avoid np.where here because both branches are evaluated.
# ============================================================

z = np.full(
    len(
        feature_ids_channel
    ),
    np.nan,
    dtype=np.float64,
)


cs_zero = (
    channel_slope_channel
    == 0.0
)


cs_nonzero = (
    ~cs_zero
    & np.isfinite(
        channel_slope_channel
    )
)


z[
    cs_zero
] = 1.0


z[
    cs_nonzero
] = (
    1.0
    / channel_slope_channel[
        cs_nonzero
    ]
)


# ============================================================
# 13. IDENTIFY CHANNELS THAT CAN BE DEPTH-RECONSTRUCTED
# ============================================================

positive_flow = (
    streamflow > 0.0
)


zero_flow = (
    streamflow == 0.0
)


valid_velocity = (
    np.isfinite(
        velocity
    )
    & (
        velocity > 0.0
    )
)


valid_geometry = (
    np.isfinite(
        mann_n_channel
    )
    & np.isfinite(
        slope_channel
    )
    & np.isfinite(
        bottom_width_channel
    )
    & np.isfinite(
        z
    )
    & (
        mann_n_channel > 0.0
    )
    & (
        slope_channel > 0.0
    )
    & (
        bottom_width_channel > 0.0
    )
    & (
        z > 0.0
    )
)


positive_flow_bad_velocity = (
    positive_flow
    & ~valid_velocity
)


positive_flow_bad_geometry = (
    positive_flow
    & ~valid_geometry
)


problem_mask = (
    positive_flow_bad_velocity
    | positive_flow_bad_geometry
)


print("\n" + "=" * 80)
print("CHANNEL DEPTH-RECONSTRUCTION DIAGNOSTICS")
print("=" * 80)


print(
    "Positive-flow reaches:",
    int(
        positive_flow.sum()
    ),
)


print(
    "Zero-flow reaches:",
    int(
        zero_flow.sum()
    ),
)


print(
    "Positive-flow reaches with invalid/zero velocity:",
    int(
        positive_flow_bad_velocity.sum()
    ),
)


print(
    "Positive-flow reaches with invalid geometry:",
    int(
        positive_flow_bad_geometry.sum()
    ),
)


if np.any(
    problem_mask
):

    problem_df = pd.DataFrame(
        {
            "feature_id": (
                feature_ids_channel[
                    problem_mask
                ]
            ),
            "streamflow": (
                streamflow[
                    problem_mask
                ]
            ),
            "velocity": (
                velocity[
                    problem_mask
                ]
            ),
            "n": (
                mann_n_channel[
                    problem_mask
                ]
            ),
            "So": (
                slope_channel[
                    problem_mask
                ]
            ),
            "BtmWdth": (
                bottom_width_channel[
                    problem_mask
                ]
            ),
            "ChSlp": (
                channel_slope_channel[
                    problem_mask
                ]
            ),
            "z": (
                z[
                    problem_mask
                ]
            ),
        }
    )


    print(
        "\nFirst problematic ordinary MC reaches:"
    )


    print(
        problem_df
        .head(
            50
        )
        .to_string(
            index=False
        )
    )


    raise RuntimeError(
        "Cannot reconstruct h0 safely for all "
        "positive-flow ordinary MC reaches. "
        "No restart files were written."
    )


# ============================================================
# 14. RECONSTRUCT CHANNEL DEPTH
#
# t-route/NWM MC velocity calculation:
#
#     V = (1/n) * R^(2/3) * sqrt(So)
#
# Therefore:
#
#     R = (V*n/sqrt(So))^(3/2)
#
#
# In the velocity calculation:
#
#     twl = bw + 2*z*h
#
# and:
#
#       h * (bw + twl) / 2
# R = -------------------------
#       bw + 2*sqrt(
#            ((twl-bw)/2)^2 + h^2
#       )
#
#
# Because:
#
#     twl = bw + 2*z*h
#
# this simplifies to:
#
#          h*(bw + z*h)
# R = ---------------------------
#       bw + 2*h*sqrt(1+z^2)
#
#
# Rearrangement gives:
#
# z*h^2
# + [bw - 2*R*sqrt(1+z^2)]*h
# - R*bw
# = 0
#
# We use the positive quadratic root.
# ============================================================

h0 = np.zeros(
    len(
        feature_ids_channel
    ),
    dtype=np.float64,
)


wet = (
    positive_flow
    & valid_velocity
    & valid_geometry
)


target_R = np.zeros(
    len(
        feature_ids_channel
    ),
    dtype=np.float64,
)


target_R[
    wet
] = (
    (
        velocity[
            wet
        ]
        * mann_n_channel[
            wet
        ]
        / np.sqrt(
            slope_channel[
                wet
            ]
        )
    )
    ** 1.5
)


sqrt_one_plus_z2 = np.sqrt(
    1.0
    + z[
        wet
    ] ** 2
)


quadratic_b = (
    bottom_width_channel[
        wet
    ]
    - 2.0
    * target_R[
        wet
    ]
    * sqrt_one_plus_z2
)


discriminant = (
    quadratic_b ** 2
    + 4.0
    * z[
        wet
    ]
    * target_R[
        wet
    ]
    * bottom_width_channel[
        wet
    ]
)


if np.any(
    ~np.isfinite(
        discriminant
    )
):

    raise RuntimeError(
        "Non-finite depth quadratic "
        "discriminant encountered."
    )


if np.any(
    discriminant < 0.0
):

    raise RuntimeError(
        "Negative depth quadratic "
        "discriminant encountered."
    )


h0[
    wet
] = (
    (
        -quadratic_b
        + np.sqrt(
            discriminant
        )
    )
    / (
        2.0
        * z[
            wet
        ]
    )
)


# ============================================================
# 15. VALIDATE RECONSTRUCTED DEPTHS
# ============================================================

if np.any(
    ~np.isfinite(
        h0
    )
):

    bad = (
        feature_ids_channel[
            ~np.isfinite(
                h0
            )
        ]
    )

    raise RuntimeError(
        f"Non-finite reconstructed depth for "
        f"{len(bad)} ordinary MC reaches."
    )


if np.any(
    h0 < 0.0
):

    bad = (
        feature_ids_channel[
            h0 < 0.0
        ]
    )

    raise RuntimeError(
        f"Negative reconstructed depth for "
        f"{len(bad)} ordinary MC reaches."
    )


# ============================================================
# 16. RECOMPUTE VELOCITY USING EXACT MC VELOCITY GEOMETRY
#
# This checks that inversion of the Manning relationship
# reproduces the archived NWM velocity.
# ============================================================

twl = (
    bottom_width_channel
    + 2.0
    * z
    * h0
)


numerator = (
    h0
    * (
        bottom_width_channel
        + twl
    )
    / 2.0
)


denominator = (
    bottom_width_channel
    + 2.0
    * np.sqrt(
        (
            (
                twl
                - bottom_width_channel
            )
            / 2.0
        ) ** 2
        + h0 ** 2
    )
)


R_check = np.zeros(
    len(
        feature_ids_channel
    ),
    dtype=np.float64,
)


positive_depth = (
    h0 > 0.0
)


R_check[
    positive_depth
] = (
    numerator[
        positive_depth
    ]
    / denominator[
        positive_depth
    ]
)


velocity_check = np.zeros(
    len(
        feature_ids_channel
    ),
    dtype=np.float64,
)


velocity_check[
    positive_depth
] = (
    (
        1.0
        / mann_n_channel[
            positive_depth
        ]
    )
    * (
        R_check[
            positive_depth
        ]
        ** (
            2.0
            / 3.0
        )
    )
    * np.sqrt(
        slope_channel[
            positive_depth
        ]
    )
)


velocity_error = np.abs(
    velocity_check[
        wet
    ]
    - velocity[
        wet
    ]
)


relative_velocity_error = (
    velocity_error
    / np.maximum(
        np.abs(
            velocity[
                wet
            ]
        ),
        1.0e-12,
    )
)


print("\n" + "=" * 80)
print("RECONSTRUCTED CHANNEL DEPTH SUMMARY")
print("=" * 80)


print(
    "Depth minimum:",
    float(
        np.min(
            h0
        )
    ),
)


print(
    "Depth mean:",
    float(
        np.mean(
            h0
        )
    ),
)


print(
    "Depth maximum:",
    float(
        np.max(
            h0
        )
    ),
)


if np.any(
    wet
):

    print(
        "Maximum absolute velocity reconstruction error:",
        float(
            np.max(
                velocity_error
            )
        ),
    )


    print(
        "Maximum relative velocity reconstruction error:",
        float(
            np.max(
                relative_velocity_error
            )
        ),
    )


# ============================================================
# 17. BUILD CHANNEL LITE RESTART DATAFRAME
#
# The t-route lite restart format contains:
#
#     qu0
#     qd0
#     h0
#     time
#
# For this NWM-state reconstruction:
#
#     qu0 = streamflow
#     qd0 = streamflow
#
# at T0.
# ============================================================

channel_restart = pd.DataFrame(
    {
        "qu0": (
            streamflow.astype(
                "float32"
            )
        ),
        "qd0": (
            streamflow.astype(
                "float32"
            )
        ),
        "h0": (
            h0.astype(
                "float32"
            )
        ),
    },
    index=feature_ids_channel,
)


channel_restart.index.name = (
    "feature_id"
)


channel_restart[
    "time"
] = T0


channel_restart = (
    channel_restart
    .sort_index()
)


# ============================================================
# 18. PRINT DICKEY AND FORT KENT CHANNEL STATES
# ============================================================

print("\n" + "=" * 80)
print("DICKEY / FORT KENT INITIAL CHANNEL CONDITIONS")
print("=" * 80)


for feature_id, name in [
    (
        DICKEY,
        "Dickey",
    ),
    (
        FORT_KENT,
        "Fort Kent",
    ),
]:

    if (
        feature_id
        not in
        channel_restart.index
    ):

        print(
            f"{name}: "
            f"feature_id={feature_id} "
            "NOT FOUND IN ORDINARY CHANNEL RESTART"
        )

        continue


    row = (
        channel_restart.loc[
            feature_id
        ]
    )


    original_position = np.where(
        feature_ids
        == feature_id
    )[0]


    if len(
        original_position
    ) == 1:

        original_position = int(
            original_position[
                0
            ]
        )

        archived_velocity = (
            velocity_all[
                original_position
            ]
        )

    else:

        archived_velocity = (
            np.nan
        )


    print(
        f"{name:10s}",
        f"feature_id={feature_id}",
        f"Q={row['qu0']:.3f} m3/s",
        f"V={archived_velocity:.3f} m/s",
        f"h0={row['h0']:.3f} m",
    )


# ============================================================
# 19. OPEN NWM V3 LAKEOUT
# ============================================================

print("\n" + "=" * 80)
print("OPENING NWM V3 LAKEOUT")
print("=" * 80)


lake_mapper = fs.get_mapper(
    LAKEOUT_STORE
)


lake = xr.open_zarr(
    lake_mapper,
    consolidated=True,
)


# ============================================================
# 20. FIND EXACT LAKEOUT TIME
# ============================================================

lake_times = pd.DatetimeIndex(
    lake["time"].values
)


lake_time_matches = np.where(
    lake_times == T0
)[0]


if len(
    lake_time_matches
) != 1:

    raise RuntimeError(
        f"Expected exactly one LAKEOUT record at {T0}, "
        f"found {len(lake_time_matches)}."
    )


lake_time_index = int(
    lake_time_matches[
        0
    ]
)


print(
    "LAKEOUT time index:",
    lake_time_index,
)


# ============================================================
# 21. MATCH WATERBODY IDs TO LAKEOUT
# ============================================================

lake_feature_ids = np.asarray(
    lake[
        "feature_id"
    ].values,
    dtype=np.int64,
)


lake_lookup = pd.Index(
    lake_feature_ids
)


lake_positions = (
    lake_lookup.get_indexer(
        waterbody_ids
    )
)


missing_lakes = (
    lake_positions < 0
)


if np.any(
    missing_lakes
):

    print(
        "\nMissing lake IDs:"
    )

    print(
        waterbody_ids[
            missing_lakes
        ]
    )

    raise RuntimeError(
        "Some domain waterbodies are missing "
        "from NWM v3 LAKEOUT."
    )


print(
    "Matched LAKEOUT waterbodies:",
    len(
        lake_positions
    ),
)


# ============================================================
# 22. LOAD WATERBODY INITIAL STATES
# ============================================================

print(
    "Downloading NWM waterbody state..."
)


lake_state = (
    lake[
        [
            "outflow",
            "water_sfc_elev",
        ]
    ]
    .isel(
        time=lake_time_index,
        feature_id=lake_positions,
    )
    .load()
)


lake_outflow = np.asarray(
    lake_state[
        "outflow"
    ].values,
    dtype=np.float64,
)


lake_elevation = np.asarray(
    lake_state[
        "water_sfc_elev"
    ].values,
    dtype=np.float64,
)


# ============================================================
# 23. VALIDATE WATERBODY INITIAL STATES
# ============================================================

if np.any(
    ~np.isfinite(
        lake_outflow
    )
):

    bad = (
        waterbody_ids[
            ~np.isfinite(
                lake_outflow
            )
        ]
    )

    print(
        "Waterbodies with non-finite outflow:"
    )

    print(
        bad
    )

    raise RuntimeError(
        "Non-finite LAKEOUT outflow encountered."
    )


if np.any(
    lake_outflow < 0.0
):

    bad = (
        waterbody_ids[
            lake_outflow < 0.0
        ]
    )

    print(
        "Waterbodies with negative outflow:"
    )

    print(
        bad
    )

    raise RuntimeError(
        "Negative LAKEOUT outflow encountered."
    )


if np.any(
    ~np.isfinite(
        lake_elevation
    )
):

    bad = (
        waterbody_ids[
            ~np.isfinite(
                lake_elevation
            )
        ]
    )

    print(
        "Waterbodies with non-finite "
        "water surface elevation:"
    )

    print(
        bad
    )

    raise RuntimeError(
        "Non-finite LAKEOUT water surface "
        "elevation encountered."
    )


# ============================================================
# 24. BUILD WATERBODY LITE RESTART DATAFRAME
# ============================================================

waterbody_restart = pd.DataFrame(
    {
        "qd0": (
            lake_outflow.astype(
                "float32"
            )
        ),
        "h0": (
            lake_elevation.astype(
                "float32"
            )
        ),
    },
    index=waterbody_ids,
)


waterbody_restart.index.name = (
    "lake_id"
)


waterbody_restart[
    "time"
] = T0


waterbody_restart = (
    waterbody_restart
    .sort_index()
)


# ============================================================
# 25. WATERBODY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("WATERBODY INITIAL CONDITIONS")
print("=" * 80)


print(
    "Waterbodies:",
    len(
        waterbody_restart
    ),
)


print(
    "Outflow min/mean/max:",
    float(
        waterbody_restart[
            "qd0"
        ].min()
    ),
    float(
        waterbody_restart[
            "qd0"
        ].mean()
    ),
    float(
        waterbody_restart[
            "qd0"
        ].max()
    ),
)


print(
    "Elevation min/mean/max:",
    float(
        waterbody_restart[
            "h0"
        ].min()
    ),
    float(
        waterbody_restart[
            "h0"
        ].mean()
    ),
    float(
        waterbody_restart[
            "h0"
        ].max()
    ),
)


# ============================================================
# 26. OPTIONAL CONSISTENCY CHECK:
#     CHRTOUT REPRESENTATIVE REACH VS LAKEOUT OUTFLOW
#
# Each of the 35 waterbodies was shown to have exactly one
# finite representative RouteLink streamflow value.
#
# This comparison is diagnostic only. We do not require exact
# equality because the two products may reflect different
# locations/state conventions within the reservoir routing.
# ============================================================

representative_rows = []


for lake_id in waterbody_ids:

    lake_reach_mask = (
        waterbody_raw
        == lake_id
    )


    finite_rep_mask = (
        lake_reach_mask
        & np.isfinite(
            streamflow_all
        )
    )


    rep_ids = (
        feature_ids[
            finite_rep_mask
        ]
    )


    rep_q = (
        streamflow_all[
            finite_rep_mask
        ]
    )


    if (
        len(
            rep_ids
        )
        == 1
    ):

        lake_idx = np.where(
            waterbody_ids
            == lake_id
        )[0]


        if len(
            lake_idx
        ) == 1:

            lake_idx = int(
                lake_idx[
                    0
                ]
            )


            representative_rows.append(
                {
                    "lake_id": int(
                        lake_id
                    ),
                    "representative_feature_id": int(
                        rep_ids[
                            0
                        ]
                    ),
                    "chrtout_streamflow": float(
                        rep_q[
                            0
                        ]
                    ),
                    "lakeout_outflow": float(
                        lake_outflow[
                            lake_idx
                        ]
                    ),
                }
            )


representative_df = pd.DataFrame(
    representative_rows
)


if not representative_df.empty:

    representative_df[
        "absolute_difference"
    ] = np.abs(
        representative_df[
            "chrtout_streamflow"
        ]
        - representative_df[
            "lakeout_outflow"
        ]
    )


    print("\n" + "=" * 80)
    print("CHRTOUT / LAKEOUT OUTFLOW DIAGNOSTIC")
    print("=" * 80)


    print(
        "Representative waterbody reaches checked:",
        len(
            representative_df
        ),
    )


    print(
        "Maximum absolute difference:",
        float(
            representative_df[
                "absolute_difference"
            ].max()
        ),
    )


    print(
        "\nFirst 10 waterbody comparisons:"
    )


    print(
        representative_df
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# 27. FINAL PRE-WRITE VALIDATION
# ============================================================

expected_channel_count = int(
    channel_mask.sum()
)


expected_waterbody_count = len(
    waterbody_ids
)


if len(
    channel_restart
) != expected_channel_count:

    raise RuntimeError(
        "Channel restart row count does not "
        "match ordinary MC channel count."
    )


if len(
    waterbody_restart
) != expected_waterbody_count:

    raise RuntimeError(
        "Waterbody restart row count does not "
        "match unique waterbody count."
    )


if not (
    channel_restart.index.is_unique
):

    raise RuntimeError(
        "Channel restart index contains duplicates."
    )


if not (
    waterbody_restart.index.is_unique
):

    raise RuntimeError(
        "Waterbody restart index contains duplicates."
    )


if (
    channel_restart[
        [
            "qu0",
            "qd0",
            "h0",
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "NaN detected in channel restart."
    )


if (
    waterbody_restart[
        [
            "qd0",
            "h0",
        ]
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "NaN detected in waterbody restart."
    )


# ============================================================
# 28. WRITE PICKLE RESTART FILES
# ============================================================

channel_restart.to_pickle(
    CHANNEL_RESTART_FILE
)


waterbody_restart.to_pickle(
    WATERBODY_RESTART_FILE
)


print("\n" + "=" * 80)
print("SAVED RESTART FILES")
print("=" * 80)


print(
    CHANNEL_RESTART_FILE
)


print(
    WATERBODY_RESTART_FILE
)


# ============================================================
# 29. READ-BACK VALIDATION
# ============================================================

channel_check = pd.read_pickle(
    CHANNEL_RESTART_FILE
)


waterbody_check = pd.read_pickle(
    WATERBODY_RESTART_FILE
)


assert (
    len(
        channel_check
    )
    == expected_channel_count
)


assert (
    len(
        waterbody_check
    )
    == expected_waterbody_count
)


assert list(
    channel_check.columns
) == [
    "qu0",
    "qd0",
    "h0",
    "time",
]


assert list(
    waterbody_check.columns
) == [
    "qd0",
    "h0",
    "time",
]


assert (
    channel_check[
        "time"
    ]
    .nunique()
    == 1
)


assert (
    waterbody_check[
        "time"
    ]
    .nunique()
    == 1
)


assert (
    pd.Timestamp(
        channel_check[
            "time"
        ].iloc[
            0
        ]
    )
    == T0
)


assert (
    pd.Timestamp(
        waterbody_check[
            "time"
        ].iloc[
            0
        ]
    )
    == T0
)


print("\n" + "=" * 80)
print("READ-BACK VALIDATION")
print("=" * 80)


print(
    "Channel restart rows:",
    len(
        channel_check
    ),
)


print(
    "Waterbody restart rows:",
    len(
        waterbody_check
    ),
)


print(
    "Restart time:",
    channel_check[
        "time"
    ].iloc[
        0
    ],
)


print(
    "Channel restart columns:",
    list(
        channel_check.columns
    ),
)


print(
    "Waterbody restart columns:",
    list(
        waterbody_check.columns
    ),
)


# ============================================================
# 30. CLOSE DATASETS
# ============================================================

state.close()

lake_state.close()

chrt.close()

lake.close()

route.close()


# ============================================================
# 31. SUCCESS
# ============================================================

print("\n" + "=" * 80)
print("NWM V3 WARM RESTART BUILD SUCCESSFUL")
print("=" * 80)