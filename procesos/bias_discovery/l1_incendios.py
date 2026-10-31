"""L1: en tractos quemados, ¿hay brecha de mapeo de edificios o es destrucción real?

Contrasta bldg_count (Overture) contra usfs_BuildingCount_sum (WRC building count,
pre-incendio; solo diagnóstico) en tractos quemados contra controles de la misma
region x ruralidad, y separa destrucción de faltante con CAL FIRE DINS
(damage inspection por estructura, CC-BY, data.cnra.ca.gov).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "datos" / "externos" / "cal_fire_dins"   # consultado 2026-10-30
OUT_CSV = ROOT / "datos" / "bias_discovery" / "l1_incendios.csv"

DINS_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query"
)
FIRES = [
    {"name": "Camp", "county": "Butte", "region": "northern-ca"},
    {"name": "Valley", "county": "Lake", "region": "northern-ca"},
    {"name": "Carr", "county": "Shasta", "region": "northern-ca"},
]
DINS_FIELDS = "DAMAGE,LATITUDE,LONGITUDE,STRUCTURECATEGORY,STRUCTURETYPE,YEARBUILT"
DET_M = 15.0          # radio de deteccion: hay moviles a ~10 m entre si
NEAR_M = 40.0         # radio de busqueda del edificio mas cercano
DESTROYED = "Destroyed (>50%)"
STANDING = {
    "No Damage",
    "Affected (>0-10%)",
    "Minor (10-25%)",
    "Major (25-50%)",
}
BURN_MIN_PCT = 10.0   # % del tracto quemado para contarlo como quemado
BURN_MIN_YEAR = 2013  # incendios desde 2013: ventana en que el rebuild ya se veria
PROJ = 3310           # CA Albers, metros


def fetch_dins(fire: dict) -> pd.DataFrame:
    EXT.mkdir(parents=True, exist_ok=True)
    cache = EXT / f"dins_{fire['county'].lower()}_{fire['name'].lower()}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    where = f"COUNTY='{fire['county']}' AND INCIDENTNAME='{fire['name']}'"
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode(
            {
                "where": where,
                "outFields": DINS_FIELDS,
                "returnGeometry": "false",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "f": "json",
            }
        )
        with urllib.request.urlopen(f"{DINS_URL}?{q}", timeout=120) as r:
            payload = json.load(r)
        feats = payload.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < 2000:
            break
        offset += 2000
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


def load_tracts(region: str) -> gpd.GeoDataFrame:
    p = ROOT / f"CONCURSO/data/strata/{region}/{region}-census-tracts.parquet"
    df = duckdb.connect().execute(
        f"SELECT GEOID, ST_AsWKB(geometry) AS wkb FROM read_parquet('{p}')"
    ).df()
    geom = shapely.from_wkb([bytes(w) for w in df["wkb"]])
    return gpd.GeoDataFrame(
        {"GEOID": df["GEOID"].astype(str)}, geometry=geom, crs="EPSG:4326"
    )


def load_buildings(region: str, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    p = ROOT / f"CONCURSO/data/reference/{region}/{region}-overture-buildings.parquet"
    x0, y0, x1, y1 = bbox
    df = duckdb.connect().execute(
        f"""SELECT ST_AsWKB(geometry) AS wkb FROM read_parquet('{p}')
            WHERE bbox.xmin >= {x0} AND bbox.ymin >= {y0}
              AND bbox.xmin <= {x1} AND bbox.ymin <= {y1}"""
    ).df()
    return gpd.GeoDataFrame(
        geometry=shapely.from_wkb([bytes(w) for w in df["wkb"]]), crs="EPSG:4326"
    ).to_crs(PROJ)


def boot_median_diff(a: np.ndarray, b: np.ndarray, reps: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    d = np.empty(reps)
    for i in range(reps):
        d[i] = np.median(rng.choice(a, a.size)) - np.median(rng.choice(b, b.size))
    return np.percentile(d, [2.5, 97.5])


def main() -> None:
    b = pd.read_parquet(ROOT / "datos/features_base/features_buildings.parquet")
    b["GEOID"] = b["GEOID"].astype(str)
    s = pd.read_parquet(
        ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
        columns=[
            "GEOID", "COUNTYFP", "mtbs_wildfire_ever", "mtbs_wildfire_last_year",
            "mtbs_wildfire_burned_pct_land", "usfs_BuildingCount_sum",
            "usfs_HUCount_sum", "carbonplan_buildings",
        ],
    )
    s["GEOID"] = s["GEOID"].astype(str)
    d = b.merge(s, on="GEOID", how="left", validate="one_to_one")
    d["rural"] = d["pct_urban"].fillna(0) < 0.5
    d["ratio"] = d["bldg_count"] / d["usfs_BuildingCount_sum"]
    d["burned"] = (
        d["mtbs_wildfire_last_year"].fillna(0) >= BURN_MIN_YEAR
    ) & (d["mtbs_wildfire_burned_pct_land"].fillna(0) >= BURN_MIN_PCT)
    d["deficit"] = d["usfs_BuildingCount_sum"] - d["bldg_count"]
    d["dins_incident"] = ""
    d["dins_inspected"] = np.nan
    d["dins_destroyed"] = np.nan
    d["dins_standing"] = np.nan
    d["det_standing"] = np.nan
    d["det_destroyed"] = np.nan
    d["deficit_per_destroyed"] = np.nan

    # ---- 1. quemados vs controles, misma region x ruralidad -------------------
    print("=" * 78)
    print("1. RATIO Overture/WRC en quemados vs no quemados (misma region x ruralidad)")
    ok = d["ratio"].notna() & (d["usfs_BuildingCount_sum"] > 0)
    for (reg, rur), g in d[ok].groupby(["region", "rural"]):
        a = g.loc[g["burned"], "ratio"].to_numpy()
        c = g.loc[~g["burned"], "ratio"].to_numpy()
        if a.size < 5 or c.size < 5:
            continue
        lo, hi = boot_median_diff(a, c)
        print(
            f"{reg:<16} rural={str(rur):<5} n_q={a.size:<4} n_c={c.size:<4} "
            f"med_q={np.median(a):.3f} med_c={np.median(c):.3f} "
            f"dif={np.median(a)-np.median(c):+.3f} IC95=[{lo:+.3f},{hi:+.3f}]"
        )

    print("\n   mismo county de California (control local, quemado vs no quemado):")
    ca = d[ok & (d["region"] == "northern-ca")]
    for cty, g in ca.groupby("COUNTYFP"):
        a = g.loc[g["burned"], "ratio"].to_numpy()
        c = g.loc[~g["burned"], "ratio"].to_numpy()
        if a.size >= 3 and c.size >= 5:
            lo, hi = boot_median_diff(a, c)
            print(
                f"   county={cty} n_q={a.size:<3} n_c={c.size:<3} "
                f"med_q={np.median(a):.3f} med_c={np.median(c):.3f} "
                f"dif={np.median(a)-np.median(c):+.3f} IC95=[{lo:+.3f},{hi:+.3f}]"
            )

    # ---- 2. DINS: destruccion real y deteccion por estructura ----------------
    print("\n" + "=" * 78)
    print("2. CAL FIRE DINS: estructuras inspeccionadas, destruidas y deteccion")
    for fire in FIRES:
        raw = fetch_dins(fire)
        raw = raw.dropna(subset=["LATITUDE", "LONGITUDE"])
        pts = gpd.GeoDataFrame(
            raw,
            geometry=gpd.points_from_xy(raw["LONGITUDE"], raw["LATITUDE"]),
            crs="EPSG:4326",
        ).to_crs(PROJ)
        tracts = load_tracts(fire["region"]).to_crs(PROJ)
        pts = gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left", predicate="within")
        pts = pts[~pts.index.duplicated(keep="first")]
        x0, y0, x1, y1 = (
            gpd.GeoSeries([pts.geometry.union_all()], crs=PROJ).to_crs(4326).total_bounds
        )
        bld = load_buildings(fire["region"], (x0, y0, x1, y1))
        near = gpd.sjoin_nearest(
            pts[["geometry"]], bld, how="left", max_distance=NEAR_M,
            distance_col="dist_m",
        )
        dist = near.groupby(level=0)["dist_m"].min()
        pts["dist_m"] = dist.reindex(pts.index)
        pts["damage"] = pts["DAMAGE"]
        pts["destroyed"] = pts["damage"] == DESTROYED
        pts["standing"] = pts["damage"].isin(STANDING)
        pts["det"] = pts["dist_m"] <= DET_M
        pts["det25"] = pts["dist_m"] <= 25.0
        print(
            f"   {fire['name']} ({fire['county']}): n={len(pts)} "
            f"destruidas={(pts['destroyed']).sum()} "
            f"en pie={(pts['standing']).sum()} "
            f"tractos={pts['GEOID'].nunique()} edificios_area={len(bld)}"
        )
        print(
            f"      deteccion Overture <= {DET_M:.0f} m: en pie="
            f"{pts.loc[pts['standing'], 'det'].mean():.3f}  "
            f"destruidas={pts.loc[pts['destroyed'], 'det'].mean():.3f}  "
            f"dist mediana en pie={pts.loc[pts['standing'], 'dist_m'].median():.1f} m"
        )
        print(
            f"      deteccion <= 25 m: en pie={pts.loc[pts['standing'], 'det25'].mean():.3f}  "
            f"destruidas={pts.loc[pts['destroyed'], 'det25'].mean():.3f}"
        )
        rows = []
        for catt, g in pts.assign(cat=pts["STRUCTURECATEGORY"].fillna("?")).groupby("cat"):
            s_ = g[g["standing"]]
            if len(s_) >= 20:
                rows.append(f"{catt}={s_['det'].mean():.2f} (n={len(s_)})")
        print("      deteccion en pie por categoria: " + "  ".join(rows))
        print(
            "      categorias destruidas: "
            + str(pts.loc[pts["destroyed"], "STRUCTURECATEGORY"].value_counts().head(4).to_dict())
        )
        agg = pd.DataFrame(
            {
                "GEOID": pts["GEOID"],
                "destroyed": pts["destroyed"].astype(float),
                "standing": pts["standing"].astype(float),
                "det_s": (pts["det"] & pts["standing"]).astype(float),
                "det_d": (pts["det"] & pts["destroyed"]).astype(float),
            }
        ).groupby("GEOID").agg(
            inspected=("destroyed", "size"), destroyed=("destroyed", "sum"),
            standing=("standing", "sum"), det_s=("det_s", "sum"), det_d=("det_d", "sum"),
        )
        agg["det_standing"] = agg["det_s"] / agg["standing"].replace(0, np.nan)
        agg["det_destroyed"] = agg["det_d"] / agg["destroyed"].replace(0, np.nan)
        mask = d["GEOID"].isin(agg.index)
        for col in ["inspected", "destroyed", "standing"]:
            d.loc[mask, f"dins_{col}"] = d.loc[mask, "GEOID"].map(agg[col]).to_numpy()
        for col in ["det_standing", "det_destroyed"]:
            d.loc[mask, col] = d.loc[mask, "GEOID"].map(agg[col]).to_numpy()
        d.loc[mask, "dins_incident"] = fire["name"]

    d["deficit_per_destroyed"] = d["deficit"] / d["dins_destroyed"].replace(0, np.nan)

    # ---- 3. conciliacion: el deficit se explica por lo destruido? -----------
    print("\n" + "=" * 78)
    print("3. CONCILIACION por tracto: deficit (WRC - Overture) vs destruido DINS")
    f = d[d["dins_incident"] != ""].copy()
    cols = [
        "GEOID", "dins_incident", "bldg_count", "usfs_BuildingCount_sum", "deficit",
        "dins_inspected", "dins_destroyed", "dins_standing", "deficit_per_destroyed",
        "det_standing", "det_destroyed", "ratio",
    ]
    print(f[cols].sort_values("deficit_per_destroyed").round(3).to_string(index=False))
    tot = f.groupby("dins_incident")[
        ["bldg_count", "usfs_BuildingCount_sum", "deficit", "dins_destroyed"]
    ].sum()
    tot["deficit_vs_destroyed"] = tot["deficit"] / tot["dins_destroyed"]
    print("\n   total por incendio:")
    print(tot.round(1).to_string())
    low = f[f["ratio"] < 0.8]
    det = (low["det_standing"] * low["dins_standing"]).sum() / low["dins_standing"].sum()
    print(
        f"\n   tractos con ratio<0.8 (n={len(low)}): deficit={low['deficit'].sum():.0f} "
        f"destruidas DINS={low['dins_destroyed'].sum():.0f} "
        f"deficit/destruidas={low['deficit'].sum()/low['dins_destroyed'].sum():.2f}"
    )
    print(
        f"   en pie detectadas en esos tractos: {det:.3f} "
        f"(n={low['dins_standing'].sum():.0f} estructuras en pie)"
    )

    # ---- 4. tabla por tracto -------------------------------------------------
    keep = [
        "GEOID", "region", "COUNTYFP", "rural", "pct_urban", "pop_total",
        "housing_units", "usfs_HUCount_sum", "carbonplan_buildings",
        "mtbs_wildfire_ever", "mtbs_wildfire_last_year",
        "mtbs_wildfire_burned_pct_land", "burned", "bldg_count",
        "usfs_BuildingCount_sum", "ratio", "deficit", "dins_incident",
        "dins_inspected", "dins_destroyed", "dins_standing",
        "deficit_per_destroyed", "det_standing", "det_destroyed",
    ]
    out = d[keep].sort_values(["region", "GEOID"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nescrito {OUT_CSV}  ({len(out)} filas)")


if __name__ == "__main__":
    main()
