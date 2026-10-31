"""Features de vias por tracto (Bias Bounty).

Uso:
    python script.py                       # las 5 regiones
    python script.py eastern-wa            # una o varias regiones
    python script.py --out otro.parquet ...  # ruta de salida alternativa

Salida: features_roads.parquet en esta carpeta (ZSTD) + resumen por stdout.

Metodo geometrico
-----------------
Tractos y vias se transforman a EPSG:5070 (metros, conus albers) usando
always_xy := true. Para cada par (via, tracto) se calcula:
    m_in  = ST_Length(via INTERSECT tracto)
    m_buf = ST_Length(via INTERSECT buffer(tracto, 25 m))
El candidato del join es el buffer (ST_Intersects), asi que m_buf >= m_in y
la franja exterior de 25 m queda definida como m_buf - m_in (>= 0 por
construccion, sin ST_Difference). Una sola pasada por region da km por clase,
km dentro del tracto y km en la franja.

La capa `roads-unfiltered` contiene todas las filas de `roads` con geometria
identica (ids y ST_Equals verificados en las 5 regiones), asi que todo se
calcula sobre unfiltered.

Regex de designacion de ruta:
    (?i)^\\s*(I|US|SR|SH|FM|CR|SPUR|LOOP)\\s*-?\\s*\\d
Se aplica a names.primary. Ademas se reporta km_tertiary_refroute, que usa
routes[1].ref IS NOT NULL (senal mas densa en OK y TX).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # raiz del repo
CONCURSO = ROOT / "CONCURSO"
AQUI = Path(__file__).resolve().parent
OUT_PARQUET = ROOT / "datos" / "features_base" / "features_roads.parquet"

REGIONS = ["eastern-wa", "eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]

HIGHWAY = ("motorway", "trunk", "primary", "secondary")
HW_COLS = [f"km_{c}" for c in HIGHWAY]
CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary",
           "residential", "unclassified", "service")

ROUTE_RX = r"(?i)^\s*(I|US|SR|SH|FM|CR|SPUR|LOOP)\s*-?\s*\d"


def paths(region: str) -> dict[str, Path]:
    return {
        "tracts": CONCURSO / "data/strata" / region / f"{region}-census-tracts.parquet",
        "strata": CONCURSO / "data/strata" / region / f"{region}-strata-tract-table.parquet",
        "roads": CONCURSO / "data/reference" / region / f"{region}-overture-roads-unfiltered.parquet",
        "rail": CONCURSO / "data/reference" / region / f"{region}-overture-rail.parquet",
        "sample": CONCURSO / "sample_submissions" / f"{region}-sample-submission.csv",
    }


def build_region(con: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    p = paths(region)
    for path in p.values():
        if not path.exists():
            raise FileNotFoundError(path)

    con.execute("DROP TABLE IF EXISTS tracts")
    con.execute("DROP TABLE IF EXISTS tb")
    con.execute("DROP TABLE IF EXISTS roads")
    con.execute("DROP TABLE IF EXISTS pr")
    con.execute("DROP TABLE IF EXISTS seg")
    con.execute("DROP TABLE IF EXISTS agg")
    con.execute("DROP TABLE IF EXISTS rail_km")
    con.execute("DROP TABLE IF EXISTS out")
    con.execute("DROP TABLE IF EXISTS out2")

    con.execute(f"""
        CREATE TABLE tracts AS
        SELECT GEOID,
               ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS g
        FROM read_parquet('{p["tracts"].as_posix()}')
    """)
    con.execute("""
        CREATE TABLE tb AS
        SELECT GEOID, g, ST_Buffer(g, 25.0) AS buf FROM tracts
    """)
    con.execute(f"""
        CREATE TABLE roads AS
        SELECT id, class, names.primary AS nm,
               (routes IS NOT NULL AND len(routes) > 0 AND routes[1].ref IS NOT NULL) AS has_ref,
               ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS g
        FROM read_parquet('{p["roads"].as_posix()}')
    """)

    con.execute("""
        CREATE TABLE pr AS
        SELECT t.GEOID, r.id, r.class, r.nm, r.has_ref,
               ST_Length(ST_Intersection(r.g, t.g))   AS m_in,
               ST_Length(ST_Intersection(r.g, t.buf)) AS m_buf
        FROM roads r JOIN tb t ON ST_Intersects(r.g, t.buf)
    """)

    # Asignacion unica de cada via al tracto donde tiene mas metros: el conteo de
    # segmentos no se duplica en tractos vecinos.
    con.execute("""
        CREATE TABLE seg AS
        SELECT GEOID, class FROM (
            SELECT GEOID, class,
                   row_number() OVER (PARTITION BY id ORDER BY m_in DESC, GEOID) AS rn
            FROM pr WHERE m_in > 0
        ) WHERE rn = 1
    """)

    cls_sums = ",\n".join(
        f"       coalesce(sum(CASE WHEN class = '{c}' THEN m_in END), 0) / 1000.0 AS km_{c}"
        for c in CLASSES
    )
    named = "nm IS NOT NULL AND trim(nm) <> ''"
    hw = "class IN ('motorway','trunk','primary','secondary')"
    tern = "class IN ('tertiary','unclassified')"

    con.execute(f"""
        CREATE TABLE agg AS
        WITH a AS (
            SELECT GEOID,
{cls_sums},
                   coalesce(sum(CASE WHEN class NOT IN ('motorway','trunk','primary','secondary',
                       'tertiary','residential','unclassified','service') THEN m_in END), 0) / 1000.0 AS km_other,
                   coalesce(sum(m_in), 0) / 1000.0 AS km_total_unfiltered,
                   coalesce(sum(CASE WHEN class = 'service' THEN m_in END), 0) / 1000.0 AS km_service_unfiltered,
                   coalesce(sum(CASE WHEN {named} THEN m_in END), 0) / 1000.0 AS km_named,
                   coalesce(sum(CASE WHEN {named} AND {hw} THEN m_in END), 0) / 1000.0 AS km_highway_named,
                   coalesce(sum(CASE WHEN {named} AND {tern} THEN m_in END), 0) / 1000.0 AS km_tertiary_named,
                   coalesce(sum(CASE WHEN {hw} THEN m_buf - m_in END), 0) / 1000.0 AS km_highway_ring25,
                   coalesce(sum(CASE WHEN {named} AND {tern} THEN m_buf - m_in END), 0) / 1000.0 AS km_tertiary_named_ring25,
                   coalesce(sum(CASE WHEN {tern} AND regexp_matches(nm, '{ROUTE_RX}') THEN m_in END), 0) / 1000.0 AS km_tertiary_routeprefix,
                   coalesce(sum(CASE WHEN {tern} AND regexp_matches(nm, '{ROUTE_RX}') THEN m_buf - m_in END), 0) / 1000.0 AS km_tertiary_routeprefix_ring25,
                   coalesce(sum(CASE WHEN {tern} AND has_ref THEN m_in END), 0) / 1000.0 AS km_tertiary_refroute
            FROM pr
            GROUP BY GEOID
        ), s AS (
            SELECT GEOID,
                   count(*) AS n_road_segments,
                   count(*) FILTER (WHERE {hw}) AS n_highway_segments
            FROM seg GROUP BY GEOID
        )
        SELECT a.*, s.n_road_segments, s.n_highway_segments
        FROM a LEFT JOIN s USING (GEOID)
    """)

    con.execute(f"""
        CREATE TABLE rail_km AS
        SELECT t.GEOID, coalesce(sum(ST_Length(ST_Intersection(r.g, t.g))), 0) / 1000.0 AS km_rail
        FROM (
            SELECT ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS g
            FROM read_parquet('{p["rail"].as_posix()}')
        ) r
        JOIN tracts t ON ST_Intersects(r.g, t.g)
        GROUP BY t.GEOID
    """)

    spine = pd.read_csv(p["sample"], dtype={"GEOID": str}, keep_default_na=False)
    con.register("spine_df", spine[["GEOID"]])
    con.execute(f"""
        CREATE TABLE out AS
        SELECT sp.GEOID,
               '{region}' AS region,
               a.* EXCLUDE (GEOID),
               coalesce(rl.km_rail, 0) AS km_rail,
               st.ALAND, st.pop_total, st.pct_urban, st.ruca_primary
        FROM spine_df sp
        LEFT JOIN agg a USING (GEOID)
        LEFT JOIN rail_km rl USING (GEOID)
        LEFT JOIN read_parquet('{p["strata"].as_posix()}') st USING (GEOID)
    """)
    con.unregister("spine_df")

    nm = [c[0] for c in con.execute("DESCRIBE out").fetchall()]
    num = [n for n in nm if n not in ("GEOID", "region", "ruca_primary")]
    fills = ", ".join(f"coalesce({n}, 0) AS {n}" for n in num)
    con.execute(f"CREATE TABLE out2 AS SELECT GEOID, region, {fills}, ruca_primary FROM out")
    df = con.execute("SELECT * FROM out2").df()

    cols = [c for c in df.columns if c.startswith("km_")]
    df[cols] = df[cols].astype("float64").round(6)
    df["km_highway"] = df[HW_COLS].sum(axis=1).round(6)
    for c in ("n_road_segments", "n_highway_segments"):
        df[c] = df[c].astype("int64")
    return df


def main(argv: list[str]) -> int:
    out = OUT_PARQUET
    args = list(argv[1:])
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1]).resolve()
        del args[i:i + 2]
    regions = args or REGIONS
    bad = [r for r in regions if r not in REGIONS]
    if bad:
        print(f"region desconocida: {bad}", file=sys.stderr)
        return 2

    con = duckdb.connect()
    con.execute("INSTALL spatial")
    con.execute("LOAD spatial")
    con.execute("PRAGMA threads=2")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET preserve_insertion_order=false")

    frames, summary = [], []
    for region in regions:
        t0 = time.time()
        df = build_region(con, region)
        dt = time.time() - t0
        frames.append(df)
        zero = float((df["km_highway"] == 0).mean())
        bad_order = int((df["km_highway"] > df["km_total_unfiltered"] + 1e-6).sum())
        summary.append(dict(region=region, tractos=len(df), km_total=round(df["km_total_unfiltered"].sum(), 1),
                            km_highway=round(df["km_highway"].sum(), 1), km_rail=round(df["km_rail"].sum(), 1),
                            share_km_highway_0=round(zero, 4), km_hw_gt_total=bad_order, segundos=round(dt, 1)))
        print(f"[{region}] {dt:.1f}s tractos={len(df)} km_total={summary[-1]['km_total']} "
              f"km_highway={summary[-1]['km_highway']} share_hw0={zero:.3f} hw>total={bad_order}")

    feats = pd.concat(frames, ignore_index=True)
    con.register("feats", feats)
    con.execute(f"COPY (SELECT * FROM feats ORDER BY region, GEOID) TO '{out.as_posix()}' "
                f"(FORMAT parquet, COMPRESSION zstd)")
    con.unregister("feats")

    print("\n" + pd.DataFrame(summary).to_string(index=False))
    print(f"\nparquet: {out}  filas={len(feats)}  cols={len(feats.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
