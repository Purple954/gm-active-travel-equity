"""
ingest.py - download raw data for the GM active travel equity project

All sources are open government data with no registration needed.
Run this first to populate data/raw/ before anything else.

Usage:
    python src/ingest.py [--data-dir data/raw]
"""

import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm


GM_LA_CODES = [
    "E08000001",  # Bolton
    "E08000002",  # Bury
    "E08000003",  # Manchester
    "E08000004",  # Oldham
    "E08000005",  # Rochdale
    "E08000006",  # Salford
    "E08000007",  # Stockport
    "E08000008",  # Tameside
    "E08000009",  # Trafford
    "E08000010",  # Wigan
]

GM_BOROUGH_NAMES = [
    "Bolton", "Bury", "Manchester", "Oldham", "Rochdale",
    "Salford", "Stockport", "Tameside", "Trafford", "Wigan",
]

# ONS lookup service - gives LSOA 2021 codes filtered to a local authority
_LSOA_LOOKUP_SVC = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA_2021_to_Ward_to_Lower_Tier_Local_Authority_May_2022_Lookup_for_England_2022/"
    "FeatureServer/0/query"
)

# ONS LSOA 2021 boundary service (Super Clipped - fine for analysis)
_LSOA_BOUNDARY_SVC = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "LSOA_2021_EW_BSC_V4_RUC/FeatureServer/0/query"
)

# ONS LAD boundary service (Dec 2021 BFC - has actual polygon geometry)
_LAD_BOUNDARY_SVC = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2021_UK_BFC_2022/FeatureServer/0/query"
)


def _get_gm_lsoa_codes(raw_dir: Path) -> list:
    """Get the list of 2021 LSOA codes for the ten GM boroughs.

    Queries the ONS LSOA-to-LAD lookup service and caches the result to
    a JSON file so subsequent calls don't need a network request.

    Parameters
    ----------
    raw_dir : Path
        Raw data directory where the code list is cached.

    Returns
    -------
    list of str
        All LSOA21CD codes within Greater Manchester.
    """
    cache = raw_dir / "gm_lsoa_codes.json"
    if cache.exists():
        return json.loads(cache.read_text())

    la_filter = ",".join(f"'{c}'" for c in GM_LA_CODES)
    params = {
        "where": f"LTLA22CD IN ({la_filter})",
        "outFields": "LSOA21CD,LTLA22CD,LTLA22NM",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": 0,
        "resultRecordCount": 2000,
    }

    codes = []
    print("  Fetching GM LSOA code list from ONS lookup...")
    while True:
        r = requests.get(_LSOA_LOOKUP_SVC, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = data.get("features", [])
        codes.extend(f["attributes"]["LSOA21CD"] for f in batch)
        if len(batch) < 2000:
            break
        params["resultOffset"] += 2000

    cache.write_text(json.dumps(codes))
    print(f"  Found {len(codes)} GM LSOAs.")
    return codes


def download_lsoa_boundaries(raw_dir: Path) -> Path:
    """Download 2021 LSOA boundaries for Greater Manchester.

    Fetches polygons from the ONS boundary service in batches of 200 codes
    rather than pulling the whole England/Wales file (~35k features).

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved GeoJSON file.
    """
    dest = raw_dir / "lsoa_boundaries_2021.geojson"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    gm_codes = _get_gm_lsoa_codes(raw_dir)
    features = []
    batch_size = 200

    print(f"  Downloading boundaries for {len(gm_codes)} GM LSOAs...")
    batches = [gm_codes[i:i+batch_size] for i in range(0, len(gm_codes), batch_size)]

    for batch in tqdm(batches, desc="  LSOA boundary pages"):
        code_list = ",".join(f"'{c}'" for c in batch)
        # POST avoids URL length limits with large code lists
        data = {
            "where": f"LSOA21CD IN ({code_list})",
            "outFields": "LSOA21CD,LSOA21NM",
            "outSR": "4326",
            "f": "geojson",
        }
        r = requests.post(_LSOA_BOUNDARY_SVC, data=data, timeout=60)
        r.raise_for_status()
        features.extend(r.json().get("features", []))

    geojson = {"type": "FeatureCollection", "features": features}
    dest.write_text(json.dumps(geojson), encoding="utf-8")
    print(f"  Saved {len(features)} LSOA boundaries.")
    return dest


def download_lad_boundaries(raw_dir: Path) -> Path:
    """Download Local Authority District boundaries for the ten GM boroughs.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved GeoJSON file.
    """
    dest = raw_dir / "lad_boundaries_2023.geojson"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    la_list = ",".join(f"'{c}'" for c in GM_LA_CODES)
    params = {
        "where": f"LAD21CD IN ({la_list})",
        "outFields": "LAD21CD,LAD21NM",
        "outSR": "4326",
        "f": "geojson",
    }
    r = requests.get(_LAD_BOUNDARY_SVC, params=params, timeout=30)
    r.raise_for_status()
    dest.write_text(r.text, encoding="utf-8")
    print(f"  Saved GM LAD boundaries.")
    return dest


def download_imd(raw_dir: Path) -> Path:
    """Download the Index of Multiple Deprivation 2019 (File 7) CSV.

    Covers all LSOAs in England with scores, ranks, and deciles.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    dest = raw_dir / "imd_2019_scores.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    url = (
        "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/"
        "attachment_data/file/845345/"
        "File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
    )
    print("  Downloading IMD 2019...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))
    return dest


def download_census_travel(raw_dir: Path) -> Path:
    """Download Census 2021 TS061 (method of travel to work) from Nomis.

    Queries the Nomis API for all GM LSOAs only, rather than pulling
    the full England/Wales dataset. Mode categories include Bicycle,
    On foot, and all other travel modes plus the total.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved CSV file.

    Notes
    -----
    Dataset ID: NM_2078_1. Mode dimension: C2021_TTWMETH_12.
    See https://www.nomisweb.co.uk/datasets/c2021ts061 for the full table.
    """
    dest = raw_dir / "census_2021_travel_to_work.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    gm_codes = _get_gm_lsoa_codes(raw_dir)
    batch_size = 100
    batches = [gm_codes[i:i+batch_size] for i in range(0, len(gm_codes), batch_size)]

    all_rows = []
    header = None
    print(f"  Downloading Census TS061 for {len(gm_codes)} GM LSOAs from Nomis...")

    for batch in tqdm(batches, desc="  Census batches"):
        geo_param = ",".join(batch)
        params = {
            "geography": geo_param,
            "measures": "20100",
            "select": "GEOGRAPHY_CODE,GEOGRAPHY_NAME,C2021_TTWMETH_12_NAME,OBS_VALUE",
        }
        r = requests.get(
            "https://www.nomisweb.co.uk/api/v01/dataset/NM_2078_1.data.csv",
            params=params,
            timeout=60,
        )
        r.raise_for_status()
        lines = r.text.strip().splitlines()
        if not lines:
            continue
        if header is None:
            header = lines[0]
            all_rows.extend(lines[1:])
        else:
            all_rows.extend(lines[1:])

    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        fh.write("\n".join(all_rows))

    print(f"  Saved Census travel data ({len(all_rows)} rows).")
    return dest


def download_cycling_osm(raw_dir: Path) -> Path:
    """Download cycling infrastructure for Greater Manchester from OpenStreetMap.

    Uses osmnx to query the Overpass API for all way features tagged as
    cycleways, cycle lanes, or cycle tracks across each GM borough. The
    result is saved as a GeoPackage and used in place of the CID.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved GeoPackage (gm_cycling_osm.gpkg).

    Notes
    -----
    OSM data is available under the Open Database Licence (ODbL).
    No account or API key is needed.
    Coverage depends on OSM contributor activity; urban areas in GM are
    generally well mapped for cycling infrastructure.
    """
    import osmnx as ox
    import geopandas as gpd
    import pandas as pd

    dest = raw_dir / "gm_cycling_osm.gpkg"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    # Tags covering the main cycling infrastructure types in OSM
    tags = {
        "highway": "cycleway",
        "cycleway": ["lane", "track", "shared_lane", "opposite_lane", "opposite_track"],
    }

    boroughs = GM_BOROUGH_NAMES
    frames = []

    print(f"  Downloading cycling infrastructure from OSM for {len(boroughs)} GM boroughs...")
    for borough in tqdm(boroughs, desc="  Boroughs"):
        try:
            place = f"{borough}, Greater Manchester, UK"
            gdf = ox.features_from_place(place, tags=tags)
            # Keep only linear features
            gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
            gdf["borough"] = borough
            frames.append(gdf[["borough", "geometry"]])
        except Exception as exc:
            print(f"  Warning: {borough} failed ({exc})")

    if not frames:
        raise RuntimeError("No cycling features returned from OSM for any GM borough.")

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    combined.to_file(dest, driver="GPKG")
    print(f"  Saved {len(combined)} OSM cycling features to {dest.name}")
    return dest


def main(data_dir: str = "data/raw") -> None:
    """Download all raw data files to data_dir.

    Parameters
    ----------
    data_dir : str, optional
        Path to raw data directory. Default is 'data/raw'.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== LSOA boundaries ===")
    download_lsoa_boundaries(raw_dir)

    print("\n=== LAD boundaries ===")
    download_lad_boundaries(raw_dir)

    print("\n=== IMD 2019 ===")
    download_imd(raw_dir)

    print("\n=== Census 2021 travel to work ===")
    download_census_travel(raw_dir)

    print("\n=== Cycling infrastructure (OpenStreetMap) ===")
    download_cycling_osm(raw_dir)

    print("\nDone. Raw files are in:", raw_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download raw data for the GM active travel equity analysis."
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Where to save raw files (default: data/raw)",
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir)
