"""Features de edificios Overture por tracto.

Uso:
    python script.py eastern-wa            # corre una region y rearma el parquet final
    python script.py --all                 # corre las 5 regiones
    python script.py --merge               # solo rearma features_buildings.parquet

Salidas:
    tmp/features_<region>.parquet
    features_buildings.parquet

Regla: usfs_BuildingCount_sum y carbonplan_buildings solo entran como columnas
`diag_*` descriptivas; ningun cociente de esta tabla las usa.
"""

import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2] / "CONCURSO"  # paquete oficial
HERE = Path(__file__).resolve().parent
TMP = ROOT.parent / "datos" / "intermedios" / "edificios"
TMP.mkdir(parents=True, exist_ok=True)
OUT = ROOT.parent / "datos" / "features_base" / "features_buildings.parquet"

REGIONS = ["eastern-wa", "maricopa-az", "northern-ca", "eastern-ok", "south-central-tx"]


def connect():
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("PRAGMA threads=2;")
    con.execute("SET memory_limit='4GB';")
    con.execute(f"SET temp_directory='{TMP.as_posix()}';")
    return con


def run_region(con, region):
    bld_p = ROOT / "data/reference" / region / f"{region}-overture-buildings.parquet"
    tr_p = ROOT / "data/strata" / region / f"{region}-census-tracts.parquet"
    acs_p = ROOT / "data/reference" / region / f"{region}-census-acs-housing.parquet"
    st_p = ROOT / "data/strata" / region / f"{region}-strata-tract-table.parquet"
    ss_p = ROOT / "sample_submissions" / f"{region}-sample-submission.csv"

    target = TMP / f"features_{region}.parquet"
    t0 = time.time()
    print(f"[{region}] start", flush=True)

    con.execute("DROP TABLE IF EXISTS geo; DROP TABLE IF EXISTS tr; DROP TABLE IF EXISTS bld; DROP TABLE IF EXISTS agg;")
    con.execute(f"CREATE TABLE geo AS SELECT GEOID::VARCHAR AS GEOID FROM read_csv('{ss_p.as_posix()}', header=true, all_varchar=true)")

    # tractos al AOI en EPSG:5070 (metros); solo las filas del sample submission
    con.execute(f"""
        CREATE TABLE tr AS
        SELECT g.GEOID, ST_Transform(c.geometry, 'EPSG:5070', always_xy := true) AS geom
        FROM geo g JOIN read_parquet('{tr_p.as_posix()}') c USING (GEOID)
    """)
    n_tr = con.execute("SELECT count(*) FROM tr").fetchone()[0]
    print(f"[{region}] tracts {n_tr} {time.time()-t0:.1f}s", flush=True)

    # edificios: area de la huella completa + fuente primaria + fuentes presentes
    con.execute(f"""
        CREATE TABLE bld AS
        SELECT
            ST_Transform(geometry, 'EPSG:5070', always_xy := true) AS geom,
            CASE sources[1].dataset
                WHEN 'Microsoft ML Buildings' THEN 'microsoft'
                WHEN 'OpenStreetMap' THEN 'osm'
                WHEN 'Google Open Buildings' THEN 'google'
                WHEN 'Esri Community Maps' THEN 'esri'
                WHEN 'USGS Lidar' THEN 'usgs_lidar'
                ELSE 'missing' END AS src1,
            len(sources) AS n_src,
            coalesce(sources[1].confidence, sources[2].confidence) AS conf,
            (sources[1].dataset = 'Microsoft ML Buildings' OR sources[2].dataset = 'Microsoft ML Buildings') AS has_microsoft,
            (sources[1].dataset = 'OpenStreetMap' OR sources[2].dataset = 'OpenStreetMap') AS has_osm,
            (sources[1].dataset = 'Google Open Buildings' OR sources[2].dataset = 'Google Open Buildings') AS has_google,
            (sources[1].dataset = 'Esri Community Maps' OR sources[2].dataset = 'Esri Community Maps') AS has_esri,
            (sources[1].dataset = 'USGS Lidar' OR sources[2].dataset = 'USGS Lidar') AS has_usgs
        FROM read_parquet('{bld_p.as_posix()}')
    """)
    n_b = con.execute("SELECT count(*) FROM bld").fetchone()[0]
    print(f"[{region}] buildings {n_b} {time.time()-t0:.1f}s", flush=True)

    # interseccion tracto-huella; area recortada al tracto
    t1 = time.time()
    con.execute("""
        CREATE TABLE agg AS
        WITH j AS (
            SELECT t.GEOID, ST_Area(b.geom) AS a_full, b.src1, b.n_src, b.conf,
                   b.has_microsoft, b.has_osm, b.has_google, b.has_esri, b.has_usgs,
                   CASE WHEN ST_Contains(t.geom, b.geom) THEN ST_Area(b.geom)
                        ELSE ST_Area(ST_Intersection(t.geom, b.geom)) END AS a_in
            FROM tr t JOIN bld b ON ST_Intersects(t.geom, b.geom)
        )
        SELECT
            GEOID,
            count(*)                                          AS bldg_count,
            sum(a_in)                                         AS bldg_area_total_m2,
            median(a_in)                                      AS bldg_area_median_m2,
            quantile_cont(a_in, 0.9)                          AS bldg_area_p90_m2,
            max(a_in)                                         AS bldg_area_max_m2,
            count(*) FILTER (WHERE a_full > 400)              AS bldg_large_count,
            count(*) FILTER (WHERE src1 = 'microsoft') / count(*)::DOUBLE AS src_microsoft_share,
            count(*) FILTER (WHERE src1 = 'osm')       / count(*)::DOUBLE AS src_osm_share,
            count(*) FILTER (WHERE src1 = 'google')    / count(*)::DOUBLE AS src_google_share,
            count(*) FILTER (WHERE src1 IN ('esri', 'usgs_lidar')) / count(*)::DOUBLE AS src_other_share,
            count(*) FILTER (WHERE src1 = 'missing')   / count(*)::DOUBLE AS src_missing_share,
            avg(conf)                                         AS confidence_mean,
            count(conf)                                       AS confidence_count,
            count(conf) / count(*)::DOUBLE                    AS pct_with_confidence,
            count(*) FILTER (WHERE n_src > 1) / count(*)::DOUBLE AS diag_multi_source_share,
            count(*) FILTER (WHERE has_microsoft) / count(*)::DOUBLE AS diag_src_microsoft_any,
            count(*) FILTER (WHERE has_osm)       / count(*)::DOUBLE AS diag_src_osm_any,
            count(*) FILTER (WHERE has_google)    / count(*)::DOUBLE AS diag_src_google_any,
            count(*) FILTER (WHERE has_esri)      / count(*)::DOUBLE AS diag_src_esri_any,
            count(*) FILTER (WHERE has_usgs)      / count(*)::DOUBLE AS diag_src_usgs_lidar_any
        FROM j GROUP BY GEOID
    """)
    n_hit = con.execute("SELECT count(*) FROM agg").fetchone()[0]
    n_pairs = con.execute("SELECT sum(bldg_count) FROM agg").fetchone()[0]
    print(f"[{region}] pairs {n_pairs} tracts_hit {n_hit}/{n_tr} {time.time()-t1:.1f}s", flush=True)

    con.execute(f"""
        COPY (
            SELECT
                g.GEOID,
                '{region}'::VARCHAR                       AS region,
                coalesce(a.bldg_count, 0)::BIGINT         AS bldg_count,
                coalesce(a.bldg_area_total_m2, 0)         AS bldg_area_total_m2,
                coalesce(a.bldg_area_median_m2, 0)        AS bldg_area_median_m2,
                coalesce(a.bldg_area_p90_m2, 0)           AS bldg_area_p90_m2,
                coalesce(a.bldg_area_max_m2, 0)           AS bldg_area_max_m2,
                coalesce(a.bldg_large_count, 0)::BIGINT   AS bldg_large_count,
                coalesce(a.src_microsoft_share, 0)        AS src_microsoft_share,
                coalesce(a.src_osm_share, 0)              AS src_osm_share,
                coalesce(a.src_google_share, 0)           AS src_google_share,
                coalesce(a.src_other_share, 0)            AS src_other_share,
                coalesce(a.src_missing_share, 0)          AS src_missing_share,
                a.confidence_mean,
                coalesce(a.confidence_count, 0)::BIGINT   AS confidence_count,
                coalesce(a.pct_with_confidence, 0)        AS pct_with_confidence,
                acs.housing_units,
                st.pop_total,
                st.ALAND,
                st.pct_urban,
                st.ur_class,
                st.ruca_primary,
                st.rucc_2023,
                st.svi_housing_transport,
                CASE WHEN a.bldg_count IS NULL THEN 1 ELSE 0 END AS bldg_missing,
                coalesce(a.diag_multi_source_share, 0)    AS diag_multi_source_share,
                coalesce(a.diag_src_microsoft_any, 0)     AS diag_src_microsoft_any,
                coalesce(a.diag_src_osm_any, 0)           AS diag_src_osm_any,
                coalesce(a.diag_src_google_any, 0)        AS diag_src_google_any,
                coalesce(a.diag_src_esri_any, 0)          AS diag_src_esri_any,
                coalesce(a.diag_src_usgs_lidar_any, 0)    AS diag_src_usgs_lidar_any,
                st.usfs_BuildingCount_sum                 AS diag_usfs_BuildingCount_sum,
                st.carbonplan_buildings                   AS diag_carbonplan_buildings
            FROM geo g
            LEFT JOIN agg a USING (GEOID)
            LEFT JOIN read_parquet('{acs_p.as_posix()}') acs USING (GEOID)
            LEFT JOIN read_parquet('{st_p.as_posix()}') st USING (GEOID)
            ORDER BY g.GEOID
        ) TO '{target.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(f"[{region}] wrote {target.name} {time.time()-t0:.1f}s", flush=True)
    return target


def merge():
    parts = [TMP / f"features_{r}.parquet" for r in REGIONS if (TMP / f"features_{r}.parquet").exists()]
    if not parts:
        raise SystemExit("no hay parquets por region")
    con = connect()
    lst = ", ".join(f"'{p.as_posix()}'" for p in parts)
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet([{lst}], union_by_name := true)
            ORDER BY region, GEOID
        ) TO '{OUT.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT.as_posix()}')").fetchone()[0]
    print(f"merged {len(parts)} regions -> {OUT.name} rows={n}", flush=True)


def main():
    args = sys.argv[1:]
    if not args or args == ["--all"]:
        todo = REGIONS
    elif args == ["--merge"]:
        return merge()
    else:
        todo = args
    con = connect()
    for r in todo:
        run_region(con, r)
    merge()


if __name__ == "__main__":
    main()
