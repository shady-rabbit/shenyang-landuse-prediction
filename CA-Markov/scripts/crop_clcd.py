"""Crop CLCD annual land-cover rasters to a city boundary.

The script keeps the original CLCD class codes unchanged:
1 Cropland, 2 Forest, 3 Shrub, 4 Grassland, 5 Water,
6 Snow/Ice, 7 Barren, 8 Impervious, 9 Wetland.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


DEFAULT_YEARS = [2000, 2005, 2010, 2015, 2020, 2025]

CITY_ALIASES = {
    "shenyang": {"eng": "Shenyang", "code": "210100"},
    "dalian": {"eng": "Dalian", "code": "210200"},
}

CLCD_CLASS_NAMES = {
    1: "Cropland",
    2: "Forest",
    3: "Shrub",
    4: "Grassland",
    5: "Water",
    6: "Snow/Ice",
    7: "Barren",
    8: "Impervious",
    9: "Wetland",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop CLCD *_albert.tif files to Shenyang or Dalian."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("E:/CA-Markov"),
        help="Project root directory.",
    )
    parser.add_argument(
        "--city",
        choices=sorted(CITY_ALIASES),
        default="shenyang",
        help="City boundary to crop to.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="CLCD years to crop.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing CLCD_v01_YYYY_albert.tif files.",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=None,
        help="City-level boundary shapefile.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for cropped rasters.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional output CSV with raw CLCD class pixel counts and areas.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Include pixels touched by the boundary, not only pixel centers.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    root = args.project_root
    if args.input_dir is None:
        args.input_dir = root / "data" / "raw" / "landuse_clcd" / "clcd_zenodo"
    if args.boundary is None:
        args.boundary = root / "data" / "raw" / "boundary" / "2023年初地级矢量.shp"
    if args.output_dir is None:
        args.output_dir = root / "data" / "processed" / "landuse" / args.city
    if args.summary_csv is None:
        args.summary_csv = (
            root / "tables" / f"{args.city}_clcd_original_class_summary.csv"
        )
    return args


def load_city_boundary(boundary_path: Path, city: str) -> gpd.GeoDataFrame:
    aliases = CITY_ALIASES[city]
    gdf = gpd.read_file(boundary_path)

    match = np.zeros(len(gdf), dtype=bool)
    if "ENG_NAME" in gdf.columns:
        match |= gdf["ENG_NAME"].astype(str).str.lower().eq(aliases["eng"].lower())
    if "NAME_2" in gdf.columns:
        match |= gdf["NAME_2"].astype(str).str.lower().eq(aliases["eng"].lower())
    if "code" in gdf.columns:
        match |= gdf["code"].astype(str).str.strip().eq(aliases["code"])

    city_gdf = gdf.loc[match].copy()
    if city_gdf.empty:
        fields = ", ".join(str(c) for c in gdf.columns)
        raise ValueError(
            f"Could not find {city} in {boundary_path}. Available fields: {fields}"
        )

    city_gdf["__dissolve__"] = 1
    return city_gdf.dissolve(by="__dissolve__", as_index=False)[["geometry"]]


def summarize_array(year: int, array: np.ndarray, nodata: float | int | None) -> list[dict]:
    data = array.reshape(-1)
    if nodata is not None:
        data = data[data != nodata]
    data = data[data != 0]

    values, counts = np.unique(data, return_counts=True)
    rows = []
    pixel_area_km2 = 30 * 30 / 1_000_000
    for value, count in zip(values, counts, strict=True):
        code = int(value)
        rows.append(
            {
                "year": year,
                "class_code": code,
                "class_name": CLCD_CLASS_NAMES.get(code, "Unknown"),
                "pixel_count": int(count),
                "area_km2": round(float(count) * pixel_area_km2, 6),
            }
        )
    return rows


def crop_one_year(
    input_path: Path,
    output_path: Path,
    city_boundary: gpd.GeoDataFrame,
    all_touched: bool,
) -> tuple[np.ndarray, dict, float | int | None]:
    with rasterio.open(input_path) as src:
        boundary_in_raster_crs = city_boundary.to_crs(src.crs)
        shapes = [geom for geom in boundary_in_raster_crs.geometry if not geom.is_empty]
        if not shapes:
            raise ValueError("City boundary is empty after CRS transformation.")

        out_image, out_transform = mask(
            src,
            shapes,
            crop=True,
            nodata=src.nodata if src.nodata is not None else 0,
            filled=True,
            all_touched=all_touched,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "nodata": src.nodata if src.nodata is not None else 0,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_image)

    return out_image[0], profile, profile.get("nodata")


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["year", "class_code", "class_name", "pixel_count", "area_km2"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = resolve_paths(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    city_boundary = load_city_boundary(args.boundary, args.city)
    summary_rows: list[dict] = []

    print(f"City: {args.city}")
    print(f"Boundary: {args.boundary}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("")

    for year in args.years:
        input_path = args.input_dir / f"CLCD_v01_{year}_albert.tif"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing CLCD raster: {input_path}")

        output_path = args.output_dir / f"{args.city}_clcd_v01_{year}_original.tif"
        array, profile, nodata = crop_one_year(
            input_path=input_path,
            output_path=output_path,
            city_boundary=city_boundary,
            all_touched=args.all_touched,
        )
        summary_rows.extend(summarize_array(year, array, nodata))
        print(
            f"{year}: wrote {output_path.name} "
            f"({profile['width']} x {profile['height']}, nodata={nodata})"
        )

    write_summary(args.summary_csv, summary_rows)
    print("")
    print(f"Summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
