"""Sonda s08: esfuerzo de mapeo humano (vias sin nombre, edificios no-OSM, POIs sin contacto).

Salida: datos/features_sonda/s08_mapping_effort.parquet
Uso:    ./.venv/Scripts/python.exe procesos/probes_s01_s10/s08_mapping_effort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "datos/features_sonda"
OUT = OUT_DIR / "s08_mapping_effort.parquet"

REGIONS = ["eastern-wa", "eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]

# Un POI cuenta como "con contacto" si trae telefono, web, email o una direccion
# con calle (freeform). locality/postcode sueltos no son direccion postal util.
CONTACT_SQL = """
       coalesce(len(phones), 0) > 0
    OR coalesce(len(websites), 0) > 0
    OR coalesce(len(emails), 0) > 0
    OR coalesce(len(list_filter(addresses,
           a -> a IS NOT NULL AND a.freeform IS NOT NULL AND a.freeform <> '')), 0) > 0
"""

# Variante laxa (cualquier addresses no vacio), solo diagnostico.
CONTACT_LOOSE_SQL = (
    "coalesce(len(phones),0)>0 OR coalesce(len(websites),0)>0 "
    "OR coalesce(len(emails),0)>0 OR coalesce(len(addresses),0)>0"
)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("PRAGMA threads=2;")
    con.execute("SET memory_limit='4GB';")
    con.execute("SET lambda_syntax='ENABLE_SINGLE_ARROW';")
    return con


def paths(region: str) -> dict[str, Path]:
    return {
        "pois": BASE / "CONCURSO/data/reference" / region / f"{region}-overture-pois.parquet",
        "tracts": BASE / "CONCURSO/data/strata" / region / f"{region}-census-tracts.parquet",
        "sample": BASE / "CONCURSO/sample_submissions" / f"{region}-sample-submission.csv",
    }


def transform_sql(col: str) -> str:
    return f"ST_Transform({col}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"


def build_region(con: duckdb.DuckDBPyConnection, region: str) -> tuple[pd.DataFrame, dict]:
    p = paths(region)

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE base AS
            SELECT DISTINCT GEOID
            FROM read_csv('{p["sample"].as_posix()}', types={{'GEOID': 'VARCHAR'}})"""
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE tract AS
            SELECT t.GEOID, {transform_sql("t.geometry")} AS g
            FROM read_parquet('{p["tracts"].as_posix()}') t
            SEMI JOIN base b ON t.GEOID = b.GEOID"""
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE poi AS
            SELECT id,
                   CASE WHEN {CONTACT_SQL} THEN 1 ELSE 0 END AS has_contact,
                   CASE WHEN {CONTACT_LOOSE_SQL} THEN 1 ELSE 0 END AS has_contact_loose,
                   {transform_sql("geometry")} AS g
            FROM read_parquet('{p["pois"].as_posix()}')"""
    )
    con.execute(
        """CREATE OR REPLACE TEMP TABLE poi_hit AS
           SELECT p.id, t.GEOID, p.has_contact, p.has_contact_loose
           FROM poi p JOIN tract t ON ST_Intersects(t.g, p.g)"""
    )
    con.execute(
        """CREATE OR REPLACE TEMP TABLE poi_uniq AS
           SELECT * FROM poi_hit QUALIFY row_number() OVER (PARTITION BY id ORDER BY GEOID) = 1"""
    )
    con.execute(
        """CREATE OR REPLACE TEMP TABLE poi_agg AS
           SELECT GEOID, count(*) AS poi_total,
                  sum(has_contact) AS poi_contact,
                  sum(has_contact_loose) AS poi_contact_loose
           FROM poi_uniq GROUP BY 1"""
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE feat AS
            SELECT b.GEOID,
                   coalesce(a.poi_total, 0) AS poi_total,
                   coalesce(a.poi_contact, 0) AS poi_contact,
                   coalesce(a.poi_contact_loose, 0) AS poi_contact_loose
            FROM base b LEFT JOIN poi_agg a ON b.GEOID = a.GEOID"""
    )

    df = con.execute("SELECT * FROM feat").df()
    df["region"] = region
    stats = con.execute(
        """SELECT (SELECT count(*) FROM base), (SELECT count(*) FROM tract),
                  (SELECT count(*) FROM poi), (SELECT count(DISTINCT id) FROM poi_uniq)"""
    ).fetchone()
    return df, {
        "region": region,
        "sample": stats[0],
        "tracts": stats[1],
        "pois_file": stats[2],
        "pois_assigned": stats[3],
    }


def main() -> int:
    roads = pd.read_parquet(
        BASE / "datos/features_base/features_roads.parquet",
        columns=["GEOID", "km_named", "km_total_unfiltered", "km_highway_named", "km_highway",
                 "pct_urban"],
    )
    bldg = pd.read_parquet(
        BASE / "datos/features_base/features_buildings.parquet",
        columns=["GEOID", "src_osm_share", "pct_with_confidence", "bldg_count", "pct_urban"],
    )

    con = connect()
    frames, stats = [], []
    for region in REGIONS:
        df, s = build_region(con, region)
        frames.append(df)
        stats.append(s)
        print(f"[{region}] sample={s['sample']} tracts={s['tracts']} "
              f"pois={s['pois_file']} asignados={s['pois_assigned']}", flush=True)
    pois = pd.concat(frames, ignore_index=True)

    df = pois.merge(roads.drop(columns=["pct_urban"]), on="GEOID", how="left", validate="1:1")
    df = df.merge(bldg, on="GEOID", how="left", validate="1:1")
    assert df["GEOID"].is_unique and len(df) == 9794, (len(df), df["GEOID"].is_unique)

    # km_total_unfiltered es el denominador de km_named en features_roads.
    km_tot = df["km_total_unfiltered"]
    df["e_unnamed_road_share"] = ((1 - df["km_named"] / km_tot)
                                  .where(km_tot > 0, 0.0).astype(float))
    df["e_nonosm_bldg_share"] = (1 - df["src_osm_share"]).fillna(0.0).astype(float)
    poi_tot = df["poi_total"]
    df["e_poi_no_contact_share"] = ((1 - df["poi_contact"] / poi_tot)
                                    .where(poi_tot > 0, 0.0).astype(float))
    df["e_low_confidence_share"] = (1 - df["pct_with_confidence"]).fillna(0.0).astype(float)

    cols = ["e_unnamed_road_share", "e_nonosm_bldg_share",
            "e_poi_no_contact_share", "e_low_confidence_share"]
    df[cols] = df[cols].clip(0.0, 1.0)
    assert not df[cols].isna().any().any(), df[cols].isna().sum().to_dict()
    assert all(df[c].std() > 0 for c in cols), "columna constante"

    # Diagnostico
    rural = 1 - df["pct_urban"]
    n_poi = int(df["poi_total"].sum())
    n_c = int(df["poi_contact"].sum())
    n_cl = int(df["poi_contact_loose"].sum())
    print(f"\nPOIs asignados={n_poi}; con contacto (estricto)={n_c} "
          f"({100*n_c/n_poi:.1f}%); laxo={n_cl} ({100*n_cl/n_poi:.1f}%)")
    print(f"tractos sin POIs={int((df['poi_total']==0).sum())} (con e_poi_no_contact=0)")
    print("\ncorrelacion con rural (1-pct_urban):")
    for c in cols:
        print(f"  {c}: {df[c].corr(rural):+.3f}  mean={df[c].mean():.3f} "
              f"sd={df[c].std():.3f} min={df[c].min():.3f} max={df[c].max():.3f}")
    print("\ncorrelacion entre features:")
    print(df[cols].corr().round(2).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df[["GEOID"] + cols].copy()
    out["GEOID"] = out["GEOID"].astype(str)
    out.to_parquet(OUT, compression="zstd", index=False)
    print(f"\n{OUT.as_posix()}: {len(out)} filas, {out['GEOID'].str.len().unique()} largo GEOID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
