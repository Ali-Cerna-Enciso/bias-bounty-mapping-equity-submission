"""s01 — features de transporte por el campo `routes` de Overture roads.

Uso:
    python s01_routes.py                 # las 5 regiones
    python s01_routes.py eastern-wa ...  # solo las indicadas

Salida: datos/features_sonda/s01_routes.parquet (ZSTD)
        GEOID (str 11) + 4 columnas float en [0,1]. Denominador = km_total del tracto.

Mecanismo (transport_gap): el componente compara Overture motorway/trunk/primary/secondary
contra TIGER S1100/S1200, que son las vias con designacion numerada. Si una via designada
queda clasificada por Overture como terciaria/residencial/sin clasificar, Overture subcuenta
km de carretera y el gap sube. Conteo de rutas por segmento, no por red: un segmento con
len(routes)>0 es "designado" sin importar cuantas redes lo compartan.

Geometria: igual que step_03_roads (EPSG:5070, always_xy := true, ST_Intersection, ST_Length).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONCURSO = ROOT / "CONCURSO"
OUT = ROOT / "datos/features_sonda/s01_routes.parquet"

REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]

HIGHWAY = "('motorway','trunk','primary')"
LOWCLASS = "('tertiary','residential','service','unclassified','other')"
# Red estatal/condal: excluye interestatal y US, y las variantes de senalizacion.
STATE_RX = r"^US:[A-Z]{2}(:.*)?$"
SIGN_RX = r":(Historic|Business|Scenic|Alternate|Future|Truck)$"


def paths(region: str) -> dict[str, Path]:
    return {
        "tracts": CONCURSO / "data/strata" / region / f"{region}-census-tracts.parquet",
        "roads": CONCURSO / "data/reference" / region / f"{region}-overture-roads.parquet",
        "sample": CONCURSO / "sample_submissions" / f"{region}-sample-submission.csv",
    }


def build_region(con: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    p = paths(region)
    for path in p.values():
        if not path.exists():
            raise FileNotFoundError(path)

    for t in ("tracts", "roads", "pr", "agg", "out"):
        con.execute(f"DROP TABLE IF EXISTS {t}")

    con.execute(f"""
        CREATE TABLE tracts AS
        SELECT GEOID, ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS g
        FROM read_parquet('{p["tracts"].as_posix()}')
    """)
    con.execute(f"""
        CREATE TABLE roads AS
        WITH b AS (
            SELECT class, geometry,
                   coalesce(len(routes) > 0, false) AS has_route,
                   coalesce(list_contains(list_transform(routes, lambda x: x.network), 'US:I'), false)
                       OR coalesce(list_contains(list_transform(routes, lambda x: x.network), 'US:US'), false)
                       AS is_national,
                   list_filter(routes, lambda x: x.network IS NOT NULL) AS rts
            FROM read_parquet('{p["roads"].as_posix()}')
        )
        SELECT class, has_route, is_national,
               len(list_filter(rts, lambda x: regexp_matches(x.network, '{STATE_RX}')
                                          AND NOT regexp_matches(x.network, '{SIGN_RX}'))) > 0
                   AS is_state,
               ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS g
        FROM b
    """)
    con.execute("""
        CREATE TABLE pr AS
        SELECT t.GEOID, r.class, r.has_route, r.is_national, r.is_state,
               ST_Length(ST_Intersection(r.g, t.g)) AS m_in
        FROM roads r JOIN tracts t ON ST_Intersects(r.g, t.g)
    """)
    con.execute(f"""
        CREATE TABLE agg AS
        SELECT GEOID,
               coalesce(sum(m_in), 0) / 1000.0 AS km_total,
               coalesce(sum(CASE WHEN has_route THEN m_in END), 0) / 1000.0 AS km_route,
               coalesce(sum(CASE WHEN has_route AND class IN {LOWCLASS} THEN m_in END), 0) / 1000.0 AS km_route_low,
               coalesce(sum(CASE WHEN has_route AND class NOT IN {HIGHWAY} THEN m_in END), 0) / 1000.0 AS km_route_nopri,
               coalesce(sum(CASE WHEN is_state AND NOT is_national AND class IN {LOWCLASS} THEN m_in END), 0)
                   / 1000.0 AS km_state_low
        FROM pr WHERE m_in > 0
        GROUP BY GEOID
    """)

    spine = pd.read_csv(p["sample"], dtype={"GEOID": str}, keep_default_na=False)
    con.register("spine_df", spine[["GEOID"]])
    con.execute("""
        CREATE TABLE out AS
        SELECT sp.GEOID,
               coalesce(a.km_total, 0)      AS km_total,
               coalesce(a.km_route, 0)      AS km_route,
               coalesce(a.km_route_low, 0)  AS km_route_low,
               coalesce(a.km_route_nopri, 0) AS km_route_nopri,
               coalesce(a.km_state_low, 0)  AS km_state_low
        FROM spine_df sp LEFT JOIN agg a USING (GEOID)
    """)
    con.unregister("spine_df")
    df = con.execute("SELECT * FROM out").df()

    df["GEOID"] = df["GEOID"].astype(str)
    zero = df["km_total"] <= 0
    den = df["km_total"].where(~zero)
    for src, dst in (("km_route", "t_route_km_share"),
                     ("km_route_low", "t_route_lowclass_share"),
                     ("km_route_nopri", "t_route_no_primary_share"),
                     ("km_state_low", "t_route_state_share")):
        df[dst] = (df[src] / den).fillna(0.0).clip(0.0, 1.0).astype("float64").round(6)
    df.attrs["zero_den"] = int(zero.sum())
    df.attrs["km_total"] = float(df["km_total"].sum())
    df.attrs["km_route"] = float(df["km_route"].sum())
    df.attrs["km_state_low"] = float(df["km_state_low"].sum())
    return df[["GEOID", "t_route_km_share", "t_route_lowclass_share",
               "t_route_no_primary_share", "t_route_state_share"]]


def main(argv: list[str]) -> int:
    regions = argv[1:] or REGIONS
    bad = [r for r in regions if r not in REGIONS]
    if bad:
        print(f"region desconocida: {bad}", file=sys.stderr)
        return 2

    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("PRAGMA threads=2")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=false")

    frames = []
    for region in regions:
        t0 = time.time()
        df = build_region(con, region)
        frames.append(df)
        print(f"[{region}] {time.time() - t0:.1f}s tractos={len(df)} sin_vias={df.attrs['zero_den']} "
              f"km_total={df.attrs['km_total']:.0f} km_route={df.attrs['km_route']:.0f} "
              f"km_state_low={df.attrs['km_state_low']:.0f}", flush=True)

    feats = pd.concat(frames, ignore_index=True)
    if regions == REGIONS:
        assert len(feats) == 9794, len(feats)
    assert feats["GEOID"].is_unique and feats["GEOID"].str.len().eq(11).all()
    num = [c for c in feats.columns if c != "GEOID"]
    feats[num] = feats[num].astype("float64")
    assert not feats[num].isna().any().any()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, compression="zstd", index=False)
    print(f"\n{OUT}  filas={len(feats)}  cols={len(feats.columns)}")
    print(feats[num].describe().T[["mean", "50%", "min", "max"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
