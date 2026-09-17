# ============================================================
# INSPECT NOAA NWM V3.0 RETROSPECTIVE S3 BUCKET
#
# Purpose:
# Search the top-level structure for static routing/domain/
# parameter files before using any RouteLink from another NWM
# version.
# ============================================================

import s3fs


# ============================================================
# 1. BUCKET
# ============================================================

BUCKET = "noaa-nwm-retrospective-3-0-pds"

fs = s3fs.S3FileSystem(
    anon=True
)


# ============================================================
# 2. HELPER
# ============================================================

def show_directory(path):

    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    try:

        items = fs.ls(
            path,
            detail=False,
        )

    except Exception as exc:

        print("Could not list directory:")
        print(exc)

        return []

    print(
        f"Number of entries: {len(items)}"
    )

    for item in items:

        print(item)

    return items


# ============================================================
# 3. ROOT
# ============================================================

root_items = show_directory(
    BUCKET
)


# ============================================================
# 4. CONUS
# ============================================================

conus_path = (
    f"{BUCKET}/CONUS"
)

conus_items = show_directory(
    conus_path
)


# ============================================================
# 5. INSPECT IMPORTANT CONUS SUBDIRECTORIES
# ============================================================

possible_paths = [
    f"{conus_path}/netcdf",
    f"{conus_path}/zarr",
    f"{conus_path}/domain",
    f"{conus_path}/parm",
    f"{conus_path}/parameter",
    f"{conus_path}/parameters",
    f"{conus_path}/static",
    f"{conus_path}/hydrofabric",
]


for path in possible_paths:

    try:

        if fs.exists(path):

            show_directory(
                path
            )

    except Exception as exc:

        print(
            f"\nCould not inspect {path}:"
        )

        print(exc)


# ============================================================
# 6. SEARCH TOP-LEVEL NAMES FOR USEFUL KEYWORDS
# ============================================================

keywords = [
    "route",
    "link",
    "domain",
    "parm",
    "param",
    "hydro",
    "static",
]


print("\n" + "=" * 80)
print("POSSIBLE ROUTING / DOMAIN ITEMS")
print("=" * 80)


all_items = (
    root_items
    + conus_items
)


matches = []

for item in all_items:

    lower = item.lower()

    if any(
        keyword in lower
        for keyword in keywords
    ):

        matches.append(
            item
        )


if matches:

    for item in sorted(
        set(matches)
    ):

        print(item)

else:

    print(
        "No obvious routing/domain/static "
        "items found at these levels."
    )


# ============================================================
# 7. FINISHED
# ============================================================

print("\n" + "=" * 80)
print("BUCKET INSPECTION COMPLETE")
print("=" * 80)
