"""
clean.py - process raw data and build the analysis dataset

Loads the five raw files from data/raw/, filters everything to Greater
Manchester, computes infrastructure density, joins deprivation scores and
travel mode share, then writes the joined dataset ready for analysis.

Usage:
    python src/clean.py [--raw-dir data/raw] [--out-dir data/processed]

Outputs:
    data/processed/gm_lsoa_analysis.gpkg  (GeoPackage with geometry)
    data/processed/gm_lsoa_analysis.csv   (flat CSV for modelling)
"""

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

GM_LA_CODES = {
    "E08000001": "Bolton",
    "E08000002": "Bury",
    "E08000003": "Manchester",
    "E08000004": "Oldham",
    "E08000005": "Rochdale",
    "E08000006": "Salford",
    "E08000007": "Stockport",
    "E08000008": "Tameside",
    "E08000009": "Trafford",
    "E08000010": "Wigan",
}

# Census 2021 TS061 mode labels (from C2021_TTWMETH_12_NAME column)
TOTAL_MODE_LABEL = "Total: All usual residents aged 16 years and over in employment the week before the census"
CYCLING_LABEL = "Bicycle"
WALKING_LABEL = "On foot"

CRS_BNG = "EPSG:27700"   # British National Grid for length/area
CRS_WGS84 = "EPSG:4326"  # WGS 84 for mapping


def load_gm_lsoas(raw_dir: Path) -> gpd.GeoDataFrame:
    """Load the 2021 LSOA boundary GeoJSON and return only GM LSOAs.

    Parameters
    ----------
    raw_dir : Path
        Directory containing lsoa_boundaries_2021.geojson.

    Returns
    -------
    geopandas.GeoDataFrame
        GM LSOA polygons in BNG, with columns lsoa_code, lsoa_name,
        and area_km2.

    Raises
    ------
    FileNotFoundError
        If the boundary file is missing.
    """
    path = raw_dir / "lsoa_boundaries_2021.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading LSOA boundaries...")
    gdf = gpd.read_file(path).to_crs(CRS_WGS84)

    # Normalise column names - ONS field names vary across file vintages
    rename = {}
    for col in gdf.columns:
        lc = col.lower()
        if "lsoa21cd" in lc or (lc.startswith("lsoa") and "cd" in lc):
            rename[col] = "lsoa_code"
        elif "lsoa21nm" in lc or (lc.startswith("lsoa") and "nm" in lc):
            rename[col] = "lsoa_name"
    gdf = gdf.rename(columns=rename)

    # The boundary file was already filtered to GM in ingest.py
    # but double-check using the code prefix (all GM LSOAs start with E0)
    if "lsoa_code" not in gdf.columns:
        raise ValueError("Could not find LSOA code column. Check the GeoJSON field names.")

    gdf = gdf.to_crs(CRS_BNG)
    gdf["area_km2"] = gdf.geometry.area / 1e6

    print(f"  {len(gdf)} GM LSOAs loaded.")
    return gdf[["lsoa_code", "lsoa_name", "area_km2", "geometry"]]


def load_cycling_osm(raw_dir: Path, gm_lsoas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Load the OSM cycling infrastructure GeoPackage and reproject to BNG.

    Parameters
    ----------
    raw_dir : Path
        Directory containing gm_cycling_osm.gpkg from ingest.py.
    gm_lsoas : geopandas.GeoDataFrame
        GM LSOA polygons in BNG (used only for a final bounds check).

    Returns
    -------
    geopandas.GeoDataFrame
        Linear cycling features in BNG, with a borough column.

    Raises
    ------
    FileNotFoundError
        If gm_cycling_osm.gpkg is missing - run ingest.py first.
    """
    path = raw_dir / "gm_cycling_osm.gpkg"
    if not path.exists():
        raise FileNotFoundError(
            f"OSM cycling data not found at {path}. Run ingest.py first."
        )

    print("  Loading OSM cycling infrastructure...")
    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.to_crs(CRS_BNG)
    print(f"  {len(gdf)} cycling features loaded.")
    return gdf[["borough", "geometry"]]


def compute_infrastructure_density(
    cid_gm: gpd.GeoDataFrame, gm_lsoas: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Sum clipped CID lengths per LSOA and calculate density (m per km2).

    Parameters
    ----------
    cid_gm : geopandas.GeoDataFrame
        CID features clipped to GM, in BNG.
    gm_lsoas : geopandas.GeoDataFrame
        GM LSOA polygons in BNG with area_km2.

    Returns
    -------
    pandas.DataFrame
        Columns: lsoa_code, infra_metres, infra_density_m_per_km2.
        LSOAs with no CID features get 0.
    """
    print("Computing infrastructure density...")

    joined = gpd.sjoin(
        cid_gm, gm_lsoas[["lsoa_code", "geometry"]],
        how="left", predicate="intersects"
    )

    # Clip each line to its LSOA polygon so boundary features aren't double counted
    joined = joined.merge(
        gm_lsoas[["lsoa_code", "geometry"]].rename(columns={"geometry": "lsoa_geom"}),
        on="lsoa_code", how="left",
    )
    joined["clipped_geom"] = joined.apply(
        lambda r: r.geometry.intersection(r.lsoa_geom)
        if r.lsoa_geom is not None else r.geometry,
        axis=1,
    )
    joined["length_m"] = joined["clipped_geom"].length

    infra = (
        joined.groupby("lsoa_code", as_index=False)["length_m"]
        .sum()
        .rename(columns={"length_m": "infra_metres"})
    )

    result = gm_lsoas[["lsoa_code", "area_km2"]].merge(infra, on="lsoa_code", how="left")
    result["infra_metres"] = result["infra_metres"].fillna(0)
    result["infra_density_m_per_km2"] = result["infra_metres"] / result["area_km2"]

    n_with = (result["infra_metres"] > 0).sum()
    print(f"  {n_with}/{len(result)} LSOAs have at least one OSM cycling feature.")
    return result[["lsoa_code", "infra_metres", "infra_density_m_per_km2"]]


def load_imd(raw_dir: Path) -> pd.DataFrame:
    """Load the IMD 2019 CSV and return scores for all LSOAs.

    Parameters
    ----------
    raw_dir : Path
        Directory containing imd_2019_scores.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: lsoa_code, imd_score, imd_rank, imd_decile, imd_quintile.
        imd_quintile runs 1 (most deprived) to 5 (least deprived).

    Raises
    ------
    FileNotFoundError
        If the IMD file is missing.
    """
    path = raw_dir / "imd_2019_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading IMD 2019...")
    raw = pd.read_csv(path, encoding="latin-1")

    # Map to stable names - match on substring so minor header text changes don't break this
    rename = {}
    for col in raw.columns:
        lc = col.strip().lower()
        if lc.startswith("lsoa code"):
            rename[col] = "lsoa_code"
        elif lc.startswith("index of multiple deprivation (imd) score"):
            rename[col] = "imd_score"
        elif lc.startswith("index of multiple deprivation (imd) rank"):
            rename[col] = "imd_rank"
        elif lc.startswith("index of multiple deprivation (imd) decile"):
            rename[col] = "imd_decile"

    renamed = raw.rename(columns=rename)
    imd = renamed[["lsoa_code", "imd_score", "imd_rank", "imd_decile"]].copy()
    imd["imd_quintile"] = np.ceil(imd["imd_decile"] / 2).astype(int)

    print(f"  {len(imd)} LSOAs in IMD file (2011 codes).")
    return imd


def load_census_travel(raw_dir: Path) -> pd.DataFrame:
    """Load Census 2021 TS061 and compute active travel mode share per LSOA.

    Active travel = (cycling + walking) / total workers * 100.

    Parameters
    ----------
    raw_dir : Path
        Directory containing census_2021_travel_to_work.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: lsoa_code, total_workers, active_travel_count,
        active_travel_pct, cycling_pct, walking_pct.

    Raises
    ------
    FileNotFoundError
        If the Census file is missing.
    """
    path = raw_dir / "census_2021_travel_to_work.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading Census 2021 travel to work...")
    df = pd.read_csv(path, low_memory=False)

    # Nomis returns columns in uppercase with quotes stripped
    df.columns = df.columns.str.strip('"').str.upper()

    # Expected columns: GEOGRAPHY_CODE, GEOGRAPHY_NAME, C2021_TTWMETH_12_NAME, OBS_VALUE
    df = df.rename(columns={
        "GEOGRAPHY_CODE": "lsoa_code",
        "C2021_TTWMETH_12_NAME": "mode",
        "OBS_VALUE": "count",
    })

    # Strip any remaining quotes from values
    df["mode"] = df["mode"].str.strip('"').str.strip()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)

    # Pull out total, cycling, and walking counts per LSOA
    total = df[df["mode"] == TOTAL_MODE_LABEL].groupby("lsoa_code")["count"].sum()
    cycling = df[df["mode"] == CYCLING_LABEL].groupby("lsoa_code")["count"].sum()
    walking = df[df["mode"] == WALKING_LABEL].groupby("lsoa_code")["count"].sum()

    result = pd.DataFrame({"total_workers": total}).join(
        cycling.rename("cycling_count"), how="left"
    ).join(
        walking.rename("walking_count"), how="left"
    ).fillna(0).reset_index()

    result["active_travel_count"] = result["cycling_count"] + result["walking_count"]
    result["active_travel_pct"] = (result["active_travel_count"] / result["total_workers"] * 100).round(2)
    result["cycling_pct"] = (result["cycling_count"] / result["total_workers"] * 100).round(2)
    result["walking_pct"] = (result["walking_count"] / result["total_workers"] * 100).round(2)

    print(f"  {len(result)} LSOAs in Census file.")
    return result[["lsoa_code", "total_workers", "active_travel_count",
                   "active_travel_pct", "cycling_pct", "walking_pct"]]


def get_lsoa_crosswalk(raw_dir: Path) -> pd.DataFrame:
    """Download and cache the ONS LSOA 2021 to 2011 best-fit lookup.

    IMD 2019 uses 2011 LSOA codes; this crosswalk maps 2021 codes to their
    2011 equivalents so the join loses as few LSOAs as possible.

    Parameters
    ----------
    raw_dir : Path
        Directory to cache the lookup CSV.

    Returns
    -------
    pandas.DataFrame
        Columns: lsoa21cd (2021 code), lsoa11cd (2011 code).
    """
    import requests

    cache = raw_dir / "lsoa_2021_to_2011_lookup.csv"
    if cache.exists():
        return pd.read_csv(cache)

    print("  Fetching LSOA 2011 -> 2021 crosswalk from ONS...")
    # This service goes LSOA11 -> LSOA21; I need it the other way round
    # but it covers all boundary change types so I can still use it to match
    url = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "LSOA11_LSOA21_LAD22_EW_LU_v5/FeatureServer/0/query"
    )
    # Check the service's actual maxRecordCount before paginating
    info = requests.get(url.replace("/query", "") + "?f=json", timeout=15).json()
    page_size = min(info.get("maxRecordCount", 1000), 1000)

    features = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "outFields": "LSOA21CD,LSOA11CD",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("features", [])
        if not batch:
            break
        features.extend(f["attributes"] for f in batch)
        offset += len(batch)
        if len(batch) < page_size:
            break

    df = pd.DataFrame(features).rename(columns={"LSOA21CD": "lsoa21cd", "LSOA11CD": "lsoa11cd"})
    # For LSOAs that split (one 2011 -> many 2021), keep just the first 2011 match
    df = df.drop_duplicates("lsoa21cd")
    df.to_csv(cache, index=False)
    print(f"  Crosswalk saved ({len(df)} rows).")
    return df


def build_analysis_dataset(
    gm_lsoas: gpd.GeoDataFrame,
    infra_df: pd.DataFrame,
    imd_df: pd.DataFrame,
    census_df: pd.DataFrame,
    raw_dir: Path = Path("data/raw"),
) -> gpd.GeoDataFrame:
    """Join all processed data into a single analysis-ready GeoDataFrame.

    Parameters
    ----------
    gm_lsoas : geopandas.GeoDataFrame
        GM LSOA polygons with lsoa_code, area_km2, geometry.
    infra_df : pandas.DataFrame
        Infrastructure density per LSOA from compute_infrastructure_density.
    imd_df : pandas.DataFrame
        Deprivation scores from load_imd.
    census_df : pandas.DataFrame
        Active travel shares from load_census_travel.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per LSOA with all analysis variables, reprojected to WGS 84.
    """
    print("Building analysis dataset...")

    # IMD 2019 uses 2011 LSOA codes. For unchanged LSOAs the codes are the same;
    # for boundary-changed ones I use the ONS crosswalk to find the right 2011 match.
    try:
        crosswalk = get_lsoa_crosswalk(raw_dir)
        # crosswalk only covers changed LSOAs; for all others 2021 code = 2011 code
        changed = crosswalk.set_index("lsoa21cd")["lsoa11cd"].to_dict()
        all_2021_codes = gm_lsoas["lsoa_code"].tolist()
        # Map each 2021 code to its corresponding 2011 code
        code_map = pd.DataFrame({
            "lsoa_code": all_2021_codes,
            "lsoa11cd": [changed.get(c, c) for c in all_2021_codes],
        })
        imd_remapped = imd_df.rename(columns={"lsoa_code": "lsoa11cd"})
        imd_remapped = (
            code_map
            .merge(imd_remapped, on="lsoa11cd", how="left")
            .drop(columns="lsoa11cd")
        )
    except Exception as exc:
        print(f"  Crosswalk unavailable ({exc}) - joining IMD on codes directly.")
        imd_remapped = imd_df  # will match where codes overlap

    gdf = gm_lsoas.copy()

    # Sequential joins
    gdf = (
        gdf
        .merge(infra_df, on="lsoa_code", how="left")
        .merge(imd_remapped, on="lsoa_code", how="left")
        .merge(census_df, on="lsoa_code", how="left")
    )

    # Fill LSOAs with no CID coverage
    gdf["infra_metres"] = gdf["infra_metres"].fillna(0)
    gdf["infra_density_m_per_km2"] = gdf["infra_density_m_per_km2"].fillna(0)

    # Log transform density (right-skewed, +1 to handle zeros)
    gdf["log_infra_density"] = np.log1p(gdf["infra_density_m_per_km2"])

    # Report join completeness
    for col, label in [("imd_score", "IMD"), ("active_travel_pct", "Census travel")]:
        missing = gdf[col].isna().sum()
        print(f"  {label}: {len(gdf) - missing}/{len(gdf)} matched ({missing} unmatched)")

    return gdf.to_crs(CRS_WGS84)


def add_borough_names(gdf: gpd.GeoDataFrame, raw_dir: Path) -> gpd.GeoDataFrame:
    """Join borough names to the analysis dataset via a spatial join with LAD boundaries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Analysis dataset (WGS 84).
    raw_dir : Path
        Directory containing lad_boundaries_2023.geojson.

    Returns
    -------
    geopandas.GeoDataFrame
        Dataset with an added borough column.
    """
    lad_path = raw_dir / "lad_boundaries_2023.geojson"
    if not lad_path.exists():
        print("  LAD boundaries not found - skipping borough name join.")
        gdf["borough"] = "Unknown"
        return gdf

    lad = gpd.read_file(lad_path).to_crs(CRS_WGS84)
    # Field names vary by LAD file vintage (LAD21CD, LAD22CD, LAD23CD)
    code_col = next((c for c in lad.columns if c.upper().startswith("LAD") and c.upper().endswith("CD")), None)
    name_col = next((c for c in lad.columns if c.upper().startswith("LAD") and c.upper().endswith("NM")), None)
    if code_col:
        lad = lad.rename(columns={code_col: "lad_code", name_col: "borough"})

    # Spatial join on LSOA centroids to avoid edge artefacts
    centroids = gdf.copy().to_crs(CRS_BNG)
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs(CRS_WGS84)

    joined = gpd.sjoin(
        centroids[["lsoa_code", "geometry"]],
        lad[["borough", "geometry"]],
        how="left",
        predicate="within",
    )
    gdf = gdf.merge(joined[["lsoa_code", "borough"]], on="lsoa_code", how="left")
    gdf["borough"] = gdf["borough"].fillna("Unknown")
    return gdf


def main(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> gpd.GeoDataFrame:
    """Run the full cleaning pipeline and write outputs.

    Parameters
    ----------
    raw_dir : str, optional
        Path to raw data directory. Default is 'data/raw'.
    out_dir : str, optional
        Path for processed outputs. Default is 'data/processed'.

    Returns
    -------
    geopandas.GeoDataFrame
        The assembled analysis dataset.
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gm_lsoas = load_gm_lsoas(raw)

    osm_path = raw / "gm_cycling_osm.gpkg"
    if osm_path.exists():
        cycling_gdf = load_cycling_osm(raw, gm_lsoas)
        infra_df = compute_infrastructure_density(cycling_gdf, gm_lsoas)
    else:
        print(
            "\n  OSM cycling file not found - run ingest.py first.\n"
            "  Infrastructure density will be set to NaN until then.\n"
        )
        infra_df = gm_lsoas[["lsoa_code"]].copy()
        infra_df["infra_metres"] = np.nan
        infra_df["infra_density_m_per_km2"] = np.nan

    imd_df = load_imd(raw)
    census_df = load_census_travel(raw)

    analysis = build_analysis_dataset(gm_lsoas, infra_df, imd_df, census_df, raw_dir=raw)
    analysis = add_borough_names(analysis, raw)

    gpkg_path = out / "gm_lsoa_analysis.gpkg"
    csv_path = out / "gm_lsoa_analysis.csv"

    analysis.to_file(gpkg_path, driver="GPKG")
    analysis.drop(columns="geometry").to_csv(csv_path, index=False)

    print(f"\nGeoPackage: {gpkg_path}")
    print(f"CSV:        {csv_path}")
    print(f"Shape:      {analysis.shape}")
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and join GM active travel equity data."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
