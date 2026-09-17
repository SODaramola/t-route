import xarray as xr
import s3fs


BUCKET = "noaa-nwm-retrospective-3-0-pds"

fs = s3fs.S3FileSystem(
    anon=True
)


stores = {
    "RTOUT": f"{BUCKET}/CONUS/zarr/rtout.zarr",
    "LAKEOUT": f"{BUCKET}/CONUS/zarr/lakeout.zarr",
}


for name, path in stores.items():

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    mapper = fs.get_mapper(
        path
    )

    ds = xr.open_zarr(
        mapper,
        consolidated=True,
    )

    print(ds)

    print("\nDimensions:")
    print(ds.sizes)

    print("\nCoordinates:")
    for coord in ds.coords:
        print(coord)

    print("\nVariables:")
    for var in ds.data_vars:
        print(
            f"{var:30s}",
            ds[var].dims,
            ds[var].dtype,
        )

    ds.close()
