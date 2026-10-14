from pathlib import Path

import pandas as pd
import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = ROOT / "outputs" / "geotiff_metadata.csv"
BOUNDARY_FILE = (
    ROOT
    / "data"
    / "natural_earth"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
OUTPUT_FILE = ROOT / "outputs" / "geographic_manifest_rebuilt.csv"

# Samples within this distance of a country boundary or coastline
# are flagged for review rather than automatically assigned.
BOUNDARY_BUFFER_KM = 1.0

COUNTRY_REGIONS = {
    "Western": ["FRA", "BEL", "NLD", "LUX", "GBR", "IRL", "MCO"],
    "Central": [
        "DEU", "AUT", "CHE", "LIE", "POL",
        "CZE", "SVK", "HUN", "SVN",
    ],
    "Southern": [
        "ESP", "PRT", "ITA", "SMR", "VAT", "MLT", "GRC",
        "ALB", "MKD", "HRV", "BIH", "MNE", "SRB", "KOS", "CYP",
    ],
    "Eastern": ["ROU", "BGR", "MDA", "UKR", "BLR", "RUS"],
}

REGION_PARTITION = {
    "Western": "development",
    "Central": "development",
    "Southern": "ood_test",
    "Eastern": "ood_test",
}


def main():
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_FILE}")

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(f"Missing shapefile: {BOUNDARY_FILE}")

    metadata = pd.read_csv(METADATA_FILE)

    required = {"class", "sample_id", "longitude", "latitude"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

    if metadata["sample_id"].duplicated().any():
        raise ValueError("Metadata contains duplicate sample IDs.")

    print("Loading Natural Earth country boundaries...")

    reader = shapefile.Reader(str(BOUNDARY_FILE))
    field_names = [field[0] for field in reader.fields[1:]]
    code_index = field_names.index("ADM0_A3")
    name_index = field_names.index("ADMIN")

    geometries = []
    country_codes = []
    country_names = []

    for item in reader.iterShapeRecords():
        geom = shape(item.shape.__geo_interface__)

        if geom.is_empty:
            continue

        geometries.append(geom)
        country_codes.append(str(item.record[code_index]).strip())
        country_names.append(str(item.record[name_index]).strip())

    tree = STRtree(geometries)

    # EPSG:3035 provides metric distances suitable for Europe.
    project = Transformer.from_crs(
        "EPSG:4326", "EPSG:3035", always_xy=True
    )

    def project_geometry(geometry):
        return transform(project.transform, geometry)
    
    projected_boundaries = [
    project_geometry(geometry.boundary)
    for geometry in geometries
]

    print(f"Loaded {len(geometries)} country geometries.")
    print(f"Processing {len(metadata)} samples...")

    rows = []

    for row in metadata.to_dict("records"):
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        point = Point(lon, lat)

        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid coordinates for {row['sample_id']}")

        candidate_indices = tree.query(point)

        matches = [
            int(i)
            for i in candidate_indices
            if geometries[int(i)].covers(point)
        ]

        country = None
        country_code = None
        region = None
        distance_km = None

        if len(matches) == 0:
            mapping_status = "no_country_match"
            partition = "unresolved"
            point_within_country = False

        elif len(matches) > 1:
            # Overlapping country polygons require manual review.
            mapping_status = "border_or_coast_review"
            partition = "unresolved"
            point_within_country = True

            country = ";".join(sorted(country_names[i] for i in matches))
            country_code = ";".join(
                sorted(country_codes[i] for i in matches)
            )
            distance_km = 0.0

        else:
            i = matches[0]
            country = country_names[i]
            country_code = country_codes[i]
            point_within_country = True

            region = next(
                (
                    name
                    for name, codes in COUNTRY_REGIONS.items()
                    if country_code in codes
                ),
                None,
            )

            projected_point = project_geometry(point)
            projected_boundary = projected_boundaries[i]

            distance_km = (
                projected_point.distance(projected_boundary) / 1000.0
            )

            if distance_km <= BOUNDARY_BUFFER_KM:
                mapping_status = "border_or_coast_review"
                partition = "unresolved"
            else:
                mapping_status = "covered"

                if region is None:
                    partition = "excluded_region"
                else:
                    partition = REGION_PARTITION[region]

        rows.append({
            "class": row["class"],
            "sample_id": row["sample_id"],
            "longitude": lon,
            "latitude": lat,
            "country": country,
            "country_code": country_code,
            "region": region,
            "point_within_country": point_within_country,
            "distance_to_country_boundary_km": distance_km,
            "mapping_status": mapping_status,
            "candidate_partition": partition,
        })

    manifest = pd.DataFrame(rows)

    if len(manifest) != len(metadata):
        raise ValueError("Output sample count does not match input metadata.")

    if manifest["sample_id"].duplicated().any():
        raise ValueError("Output manifest contains duplicate sample IDs.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_FILE, index=False)

    print("\nCandidate partition counts:")
    print(manifest["candidate_partition"].value_counts(dropna=False))

    print("\nMapping status counts:")
    print(manifest["mapping_status"].value_counts(dropna=False))

    print("\nRegion counts:")
    print(manifest["region"].value_counts(dropna=False))

    print(f"\nSaved new manifest to:\n{OUTPUT_FILE}")

    # Compare with the existing preliminary manifest, if available.
    old_file = ROOT / "outputs" / "preliminary_region_manifest.csv"

    if old_file.exists():
        old = pd.read_csv(old_file)
        comparison = old[["sample_id", "candidate_partition"]].merge(
            manifest[["sample_id", "candidate_partition"]],
            on="sample_id",
            how="outer",
            suffixes=("_old", "_new"),
            indicator=True,
        )

        print("\nComparison with the existing manifest:")
        print("Rows matched by sample ID:", (comparison["_merge"] == "both").sum())
        print("Missing from either manifest:", (comparison["_merge"] != "both").sum())

        changed = comparison[
            (comparison["_merge"] == "both")
            & (
                comparison["candidate_partition_old"]
                != comparison["candidate_partition_new"]
            )
        ]

        print("Changed partition assignments:", len(changed))
        if not changed.empty:
            print(changed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()