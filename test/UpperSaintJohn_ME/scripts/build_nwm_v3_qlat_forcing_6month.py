# ============================================================
# BUILD NWM V3 Q_LATERAL CHRTOUT FORCING FOR T-ROUTE
# UPPER SAINT JOHN / FORT KENT DOMAIN
#
# Simulation start:
#     2016-11-01 00:00:00
#
# Forcing period:
#     2016-11-01 00:00:00
#     through
#     2017-04-30 23:00:00
#
# Number of forcing files:
#     4344 hourly CHRTOUT files
#
# Domain:
#     4392 RouteLink reaches
#
# IMPORTANT:
#     Only q_lateral is written.
#
#     qBucket and qSfcLatRunoff are intentionally NOT written
#     because t-route gives those variables priority over
#     q_lateral if both are present.
#
# Output folder:
#     forcing/chrtout_20161101_20170501/
#
# File naming:
#     YYYYMMDDHHMM.CHRTOUT_DOMAIN1
#
# Example:
#     201611010000.CHRTOUT_DOMAIN1
#     ...
#     201704302300.CHRTOUT_DOMAIN1
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


OUTPUT_DIR = (
    BASE_DIR
    / "forcing"
    / "chrtout_20161101_20170501"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


BUCKET = (
    "noaa-nwm-retrospective-3-0-pds"
)


CHRTOUT_STORE = (
    f"{BUCKET}/CONUS/zarr/chrtout.zarr"
)


MODEL_START = pd.Timestamp(
    "2016-11-01 00:00:00"
)


FORCING_START = pd.Timestamp(
    "2016-11-01 00:00:00"
)


FORCING_END = pd.Timestamp(
    "2017-04-30 23:00:00"
)


EXPECTED_HOURS = 4344


# ============================================================
# 2. READ T-ROUTE DOMAIN
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


waterbody_ids = np.asarray(
    route["NHDWaterbodyComID"].values,
    dtype=np.int64,
)


channel_mask = (
    waterbody_ids == -9999
)


waterbody_mask = (
    waterbody_ids != -9999
)


print(
    "Domain reaches:",
    len(feature_ids),
)


print(
    "Ordinary MC reaches:",
    int(
        channel_mask.sum()
    ),
)


print(
    "Waterbody-associated RouteLink reaches:",
    int(
        waterbody_mask.sum()
    ),
)


# ============================================================
# 3. BUILD EXACT HOURLY FORCING TIME SEQUENCE
# ============================================================

forcing_times = pd.date_range(
    FORCING_START,
    FORCING_END,
    freq="1h",
)


if len(
    forcing_times
) != EXPECTED_HOURS:

    raise RuntimeError(
        f"Expected {EXPECTED_HOURS} forcing hours, "
        f"found {len(forcing_times)}."
    )


# First forcing must begin exactly at model start.

if (
    forcing_times[0]
    != MODEL_START
):

    raise RuntimeError(
        "First forcing time does not equal model start."
    )


print()
print(
    "Model start:",
    MODEL_START,
)


print(
    "First forcing:",
    forcing_times[0],
)


print(
    "Last forcing:",
    forcing_times[-1],
)


print(
    "Number of hourly forcing files:",
    len(
        forcing_times
    ),
)


# ============================================================
# 4. OPEN NWM V3 RETROSPECTIVE CHRTOUT ZARR
# ============================================================

print("\n" + "=" * 80)
print("OPENING NWM V3 RETROSPECTIVE")
print("=" * 80)


fs = s3fs.S3FileSystem(
    anon=True
)


mapper = fs.get_mapper(
    CHRTOUT_STORE
)


nwm = xr.open_zarr(
    mapper,
    consolidated=True,
)


# ============================================================
# 5. MATCH FEATURE IDs EXACTLY
#
# Never use nearest-neighbor matching for categorical feature
# IDs.
# ============================================================

print(
    "Loading NWM feature_id coordinate..."
)


nwm_feature_ids = np.asarray(
    nwm["feature_id"].values,
    dtype=np.int64,
)


feature_positions = (
    pd.Index(
        nwm_feature_ids
    )
    .get_indexer(
        feature_ids
    )
)


missing_feature_mask = (
    feature_positions < 0
)


if np.any(
    missing_feature_mask
):

    missing_ids = (
        feature_ids[
            missing_feature_mask
        ]
    )

    print(
        "\nMissing NWM feature IDs:"
    )

    print(
        missing_ids[:50]
    )

    raise RuntimeError(
        f"{len(missing_ids)} RouteLink feature IDs "
        "are absent from NWM v3."
    )


print(
    "Matched feature IDs:",
    len(
        feature_positions
    ),
)


# ============================================================
# 6. MATCH FORCING TIMES EXACTLY
# ============================================================

print(
    "Loading NWM time coordinate..."
)


nwm_times = pd.DatetimeIndex(
    nwm["time"].values
)


time_positions = (
    pd.Index(
        nwm_times
    )
    .get_indexer(
        forcing_times
    )
)


missing_time_mask = (
    time_positions < 0
)


if np.any(
    missing_time_mask
):

    missing_times = (
        forcing_times[
            missing_time_mask
        ]
    )

    print(
        "\nMissing NWM forcing times:"
    )

    print(
        missing_times
    )

    raise RuntimeError(
        f"{len(missing_times)} requested forcing "
        "times are absent from NWM v3."
    )


print(
    "Matched forcing times:",
    len(
        time_positions
    ),
)


# ============================================================
# 7. LOAD Q_LATERAL
# ============================================================

print("\n" + "=" * 80)
print("DOWNLOADING NWM V3 Q_LATERAL")
print("=" * 80)


qlat_da = (
    nwm["q_lateral"]
    .isel(
        time=time_positions,
        feature_id=feature_positions,
    )
    .load()
)


qlat = np.asarray(
    qlat_da.values,
    dtype=np.float64,
)


expected_shape = (
    len(
        forcing_times
    ),
    len(
        feature_ids
    ),
)


if qlat.shape != expected_shape:

    raise RuntimeError(
        f"Unexpected q_lateral shape: {qlat.shape}; "
        f"expected {expected_shape}."
    )


print(
    "q_lateral shape:",
    qlat.shape,
)


# ============================================================
# 8. Q_LATERAL VALIDATION
# ============================================================

nonfinite_all = (
    ~np.isfinite(
        qlat
    )
)


negative_all = (
    qlat < 0.0
)


print("\n" + "=" * 80)
print("Q_LATERAL VALIDATION")
print("=" * 80)


print(
    "All-domain non-finite values:",
    int(
        nonfinite_all.sum()
    ),
)


print(
    "Ordinary-channel non-finite values:",
    int(
        (
            ~np.isfinite(
                qlat[
                    :,
                    channel_mask
                ]
            )
        ).sum()
    ),
)


print(
    "Waterbody-associated non-finite values:",
    int(
        (
            ~np.isfinite(
                qlat[
                    :,
                    waterbody_mask
                ]
            )
        ).sum()
    ),
)


print(
    "Negative values:",
    int(
        negative_all.sum()
    ),
)


if np.any(
    nonfinite_all
):

    raise RuntimeError(
        "Non-finite q_lateral values encountered. "
        "No forcing files were written."
    )


if np.any(
    negative_all
):

    raise RuntimeError(
        "Negative q_lateral values encountered. "
        "No forcing files were written."
    )


print()
print(
    "All-domain min:",
    float(
        np.min(
            qlat
        )
    ),
)


print(
    "All-domain mean:",
    float(
        np.mean(
            qlat
        )
    ),
)


print(
    "All-domain max:",
    float(
        np.max(
            qlat
        )
    ),
)


# ============================================================
# 9. CHECK OUTPUT DIRECTORY
#
# Only CHRTOUT files with the exact requested timestamps are
# allowed in this dedicated forcing folder.
# ============================================================

expected_filenames = [
    (
        timestamp.strftime(
            "%Y%m%d%H%M"
        )
        + ".CHRTOUT_DOMAIN1"
    )
    for timestamp
    in forcing_times
]


existing_chrtout = sorted(
    OUTPUT_DIR.glob(
        "*.CHRTOUT_DOMAIN1"
    )
)


unexpected_existing = [
    path
    for path
    in existing_chrtout
    if path.name
    not in expected_filenames
]


if unexpected_existing:

    print(
        "\nUnexpected CHRTOUT files already exist "
        "in the forcing directory:"
    )

    for path in unexpected_existing:

        print(
            path
        )

    raise RuntimeError(
        "Remove or move unexpected forcing files "
        "before continuing."
    )


# ============================================================
# 10. WRITE 4344 CHRTOUT FILES
#
# Known-good NWM CHRTOUT structure has:
#
#     feature_id dimension
#     time dimension of length 1
#
# with q_lateral stored over feature_id.
#
# We intentionally omit:
#
#     qBucket
#     qSfcLatRunoff
#
# so that t-route reads q_lateral directly.
# ============================================================

print("\n" + "=" * 80)
print("WRITING CHRTOUT FORCING FILES")
print("=" * 80)


written_files = []


for i, timestamp in enumerate(
    forcing_times
):

    filename = (
        timestamp.strftime(
            "%Y%m%d%H%M"
        )
        + ".CHRTOUT_DOMAIN1"
    )


    output_file = (
        OUTPUT_DIR
        / filename
    )


    ds_out = xr.Dataset(
        data_vars={
            "q_lateral": (
                (
                    "feature_id",
                ),
                qlat[
                    i,
                    :
                ].astype(
                    np.float32
                ),
            ),
        },
        coords={
            "feature_id": (
                (
                    "feature_id",
                ),
                feature_ids.astype(
                    np.int64
                ),
            ),
            "time": (
                (
                    "time",
                ),
                np.array(
                    [
                        timestamp.to_datetime64()
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
        },
        attrs={
            "TITLE": (
                "NWM v3 retrospective q_lateral "
                "forcing subset for t-route"
            ),
            "featureType": (
                "timeSeries"
            ),
            "station_dimension": (
                "feature_id"
            ),
            "model_initialization_time": (
                MODEL_START.strftime(
                    "%Y-%m-%d_%H:%M:%S"
                )
            ),
            "model_output_valid_time": (
                timestamp.strftime(
                    "%Y-%m-%d_%H:%M:%S"
                )
            ),
            "model_output_type": (
                "channel_rt"
            ),
            "model_configuration": (
                "retrospective"
            ),
            "Conventions": (
                "CF-1.6"
            ),
            "source": (
                "NOAA NWM Retrospective v3.0 "
                "CONUS/chrtout.zarr"
            ),
        },
    )


    ds_out["q_lateral"].attrs = {
        "long_name": (
            "lateral inflow"
        ),
        "units": (
            "m3 s-1"
        ),
    }


    ds_out["feature_id"].attrs = {
        "long_name": (
            "NWM feature identifier"
        ),
    }


    encoding = {
        "q_lateral": {
            "dtype": "float32",
            "_FillValue": np.float32(
                -9999.0
            ),
        },
        "feature_id": {
            "dtype": "int64",
        },
    }


    ds_out.to_netcdf(
        output_file,
        mode="w",
        engine="netcdf4",
        format="NETCDF4",
        encoding=encoding,
    )


    ds_out.close()


    written_files.append(
        output_file
    )


    if (
        i == 0
        or i == len(
            forcing_times
        ) - 1
        or (
            i + 1
        ) % 12 == 0
    ):

        print(
            f"Wrote {i + 1:2d}/"
            f"{len(forcing_times)}: "
            f"{filename}"
        )


# ============================================================
# 11. VERIFY FILE COUNT AND FILENAMES
# ============================================================

actual_files = sorted(
    OUTPUT_DIR.glob(
        "*.CHRTOUT_DOMAIN1"
    )
)


if len(
    actual_files
) != EXPECTED_HOURS:

    raise RuntimeError(
        f"Expected {EXPECTED_HOURS} CHRTOUT files, "
        f"found {len(actual_files)}."
    )


actual_names = [
    path.name
    for path
    in actual_files
]


if actual_names != expected_filenames:

    raise RuntimeError(
        "CHRTOUT filenames do not exactly match "
        "the expected forcing sequence."
    )


# ============================================================
# 12. READ-BACK VALIDATION FOR EVERY FILE
# ============================================================

print("\n" + "=" * 80)
print("READ-BACK VALIDATION")
print("=" * 80)


max_abs_difference = 0.0


for i, (
    timestamp,
    path,
) in enumerate(
    zip(
        forcing_times,
        actual_files,
    )
):

    with xr.open_dataset(
        path
    ) as check:

        # Required variable
        if (
            "q_lateral"
            not in check.variables
        ):

            raise RuntimeError(
                f"q_lateral missing from {path.name}"
            )


        # Variables intentionally excluded
        if (
            "qBucket"
            in check.variables
            or
            "qSfcLatRunoff"
            in check.variables
        ):

            raise RuntimeError(
                f"Unexpected qBucket/qSfcLatRunoff "
                f"found in {path.name}."
            )


        check_ids = np.asarray(
            check[
                "feature_id"
            ].values,
            dtype=np.int64,
        )


        if not np.array_equal(
            check_ids,
            feature_ids,
        ):

            raise RuntimeError(
                f"feature_id mismatch in {path.name}."
            )


        valid_time = (
            check.attrs.get(
                "model_output_valid_time"
            )
        )


        expected_valid_time = (
            timestamp.strftime(
                "%Y-%m-%d_%H:%M:%S"
            )
        )


        if (
            valid_time
            != expected_valid_time
        ):

            raise RuntimeError(
                f"model_output_valid_time mismatch "
                f"in {path.name}: "
                f"{valid_time}"
            )


        q_check = np.asarray(
            check[
                "q_lateral"
            ].values,
            dtype=np.float64,
        )


        if q_check.shape != (
            len(
                feature_ids
            ),
        ):

            raise RuntimeError(
                f"Unexpected q_lateral shape "
                f"in {path.name}: "
                f"{q_check.shape}"
            )


        difference = np.abs(
            q_check
            - qlat[
                i,
                :
            ]
        )


        local_max = float(
            np.max(
                difference
            )
        )


        max_abs_difference = max(
            max_abs_difference,
            local_max,
        )


# ============================================================
# 13. PRINT FIRST AND LAST FILE DETAILS
# ============================================================

first_file = (
    actual_files[
        0
    ]
)


last_file = (
    actual_files[
        -1
    ]
)


print(
    "Files validated:",
    len(
        actual_files
    ),
)


print(
    "First file:",
    first_file.name,
)


print(
    "Last file:",
    last_file.name,
)


print(
    "Maximum write/read q_lateral difference:",
    max_abs_difference,
)


# ============================================================
# 14. CHECK HOURLY VALID-TIME SPACING FROM FILE ATTRIBUTES
# ============================================================

valid_times = []


for path in actual_files:

    with xr.open_dataset(
        path
    ) as check:

        valid_times.append(
            pd.Timestamp(
                check.attrs[
                    "model_output_valid_time"
                ].replace(
                    "_",
                    " "
                )
            )
        )


valid_times = pd.DatetimeIndex(
    valid_times
)


time_differences = (
    valid_times[
        1:
    ]
    - valid_times[
        :-1
    ]
)


if not np.all(
    time_differences
    == pd.Timedelta(
        hours=1
    )
):

    raise RuntimeError(
        "Forcing files are not uniformly "
        "spaced at one-hour intervals."
    )


print(
    "Forcing interval:",
    time_differences[
        0
    ],
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FORCING SUMMARY")
print("=" * 80)


print(
    "Output directory:",
    OUTPUT_DIR
)


print(
    "Number of files:",
    len(
        actual_files
    ),
)


print(
    "Features per file:",
    len(
        feature_ids
    ),
)


print(
    "First valid time:",
    valid_times[
        0
    ],
)


print(
    "Last valid time:",
    valid_times[
        -1
    ],
)


print(
    "q_lateral minimum:",
    float(
        np.min(
            qlat
        )
    ),
)


print(
    "q_lateral mean:",
    float(
        np.mean(
            qlat
        )
    ),
)


print(
    "q_lateral maximum:",
    float(
        np.max(
            qlat
        )
    ),
)


# ============================================================
# 16. CLOSE SOURCE DATASETS
# ============================================================

qlat_da.close()

nwm.close()

route.close()


# ============================================================
# 17. SUCCESS
# ============================================================

print("\n" + "=" * 80)
print("NWM V3 Q_LATERAL FORCING BUILD SUCCESSFUL")
print("=" * 80)
