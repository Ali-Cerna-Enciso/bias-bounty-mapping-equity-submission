"""Features de POIs e infraestructura de transporte por tracto (5 regiones).

Uso:
    python script.py                 # las 5 regiones
    python script.py --regions eastern-wa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

BASE = Path(__file__).resolve().parents[2] / "CONCURSO"
OUT = Path(__file__).resolve().parents[2] / "datos/features_base/features_pois.parquet"

REGIONS = ["eastern-wa", "eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]

SCHOOL_CATEGORIES = (
    "elementary_school",
    "middle_school",
    "high_school",
    "school",
    "private_school",
    "public_school",
)

# Grupos por taxonomía Overture: hierarchy[1] es el dominio, hierarchy[2] la clase.
GROUPS = {
    "poi_education": "h1 = 'education'",
    "poi_health": "h1 = 'health_care'",
    "poi_retail": "h1 = 'shopping'",
    "poi_food": "h1 = 'food_and_drink'",
    "poi_government": (
        "h2 IN ('government_office','public_safety_service','public_utility',"
        "'public_facility','military_site')"
    ),
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("PRAGMA threads=2;")
    con.execute("SET memory_limit='4GB';")
    return con


def paths(region: str) -> dict[str, Path]:
    return {
        "pois": BASE / "data/reference" / region / f"{region}-overture-pois.parquet",
        "infra": BASE / "data/reference" / region / f"{region}-overture-infrastructure.parquet",
        "tracts": BASE / "data/strata" / region / f"{region}-census-tracts.parquet",
        "strata": BASE / "data/strata" / region / f"{region}-strata-tract-table.parquet",
        "acs": BASE / "data/reference" / region / f"{region}-census-acs-housing.parquet",
        "sample": BASE / "sample_submissions" / f"{region}-sample-submission.csv",
    }


def transform_sql(col: str) -> str:
    return f"ST_Transform({col}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"


def build_region(con: duckdb.DuckDBPyConnection, region: str) -> dict[str, int]:
    p = paths(region)

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE base AS
            SELECT DISTINCT GEOID
            FROM read_csv('{p["sample"].as_posix()}', types={{'GEOID': 'VARCHAR'}})"""
    )

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE tract AS
            SELECT t.GEOID, t.pop_total, t.ALAND, t.pct_urban, t.ur_class,
                   {transform_sql("t.geometry")} AS g
            FROM read_parquet('{p["tracts"].as_posix()}') t
            SEMI JOIN base b ON t.GEOID = b.GEOID"""
    )

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE poi AS
            SELECT id,
                   categories.primary            AS cat,
                   taxonomy.hierarchy[1]         AS h1,
                   taxonomy.hierarchy[2]         AS h2,
                   confidence,
                   {transform_sql("geometry")}   AS g
            FROM read_parquet('{p["pois"].as_posix()}')"""
    )

    con.execute(
        """CREATE OR REPLACE TEMP TABLE poi_hit AS
           SELECT p.id, t.GEOID, p.cat, p.h1, p.h2, p.confidence
           FROM poi p JOIN tract t ON ST_Intersects(t.g, p.g)"""
    )

    # Un POI sobre el límite puede caer en dos tractos: se asigna al GEOID menor.
    con.execute(
        """CREATE OR REPLACE TEMP TABLE poi_uniq AS
           SELECT * FROM poi_hit QUALIFY row_number() OVER (PARTITION BY id ORDER BY GEOID) = 1"""
    )

    school_list = ", ".join(f"'{c}'" for c in SCHOOL_CATEGORIES)
    group_cols = "".join(
        f",\n                   count(*) FILTER (WHERE {cond}) AS {name}"
        for name, cond in GROUPS.items()
    )

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE poi_agg AS
            SELECT GEOID,
                   count(*)                                                    AS poi_total,
                   count(*) FILTER (WHERE cat = 'fire_department')             AS poi_fire,
                   count(*) FILTER (WHERE cat = 'ambulance_and_ems_services')  AS poi_ems,
                   count(*) FILTER (WHERE cat = 'hospital')                    AS poi_hospital,
                   count(*) FILTER (WHERE cat = 'elementary_school')        AS poi_elementary,
                   count(*) FILTER (WHERE cat = 'middle_school')            AS poi_middle,
                   count(*) FILTER (WHERE cat = 'high_school')              AS poi_high,
                   count(*) FILTER (WHERE cat = 'school')                   AS poi_school,
                   count(*) FILTER (WHERE cat = 'private_school')           AS poi_private_school,
                   count(*) FILTER (WHERE cat = 'public_school')            AS poi_public_school,
                   count(*) FILTER (WHERE cat IN ({school_list}))              AS poi_schools,
                   count(*) FILTER (WHERE cat = 'fire_department'
                                       OR cat = 'ambulance_and_ems_services')  AS poi_emergency,
                   count(*) FILTER (WHERE cat = 'college_university')          AS poi_college_university,
                   count(*) FILTER (WHERE cat = 'police_department')           AS poi_police{group_cols},
                   avg(confidence)                                             AS confidence_mean,
                   count(confidence)                                           AS confidence_count
            FROM poi_uniq
            GROUP BY 1"""
    )

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE infra AS
            SELECT id, subtype, class,
                   {transform_sql("ST_Centroid(geometry)")} AS g
            FROM read_parquet('{p["infra"].as_posix()}')"""
    )

    con.execute(
        """CREATE OR REPLACE TEMP TABLE infra_hit AS
           SELECT i.id, t.GEOID, i.subtype, i.class
           FROM infra i JOIN tract t ON ST_Intersects(t.g, i.g)"""
    )

    con.execute(
        """CREATE OR REPLACE TEMP TABLE infra_uniq AS
           SELECT * FROM infra_hit QUALIFY row_number() OVER (PARTITION BY id ORDER BY GEOID) = 1"""
    )

    con.execute(
        """CREATE OR REPLACE TEMP TABLE infra_agg AS
           SELECT GEOID,
                  count(*)                                          AS infra_total,
                  count(*) FILTER (WHERE subtype = 'transit')       AS infra_transit,
                  count(*) FILTER (WHERE subtype = 'airport')       AS infra_airport,
                  count(*) FILTER (WHERE subtype = 'transit'
                                      OR subtype = 'airport')       AS infra_known,
                  count(*) FILTER (WHERE class = 'bus_stop')        AS infra_bus_stop,
                  count(*) FILTER (WHERE class = 'bus_station')     AS infra_bus_station,
                  count(*) FILTER (WHERE class = 'railway_station') AS infra_railway_station,
                  count(*) FILTER (WHERE class = 'platform')        AS infra_platform,
                  count(*) FILTER (WHERE class = 'airport')         AS infra_airport_facility,
                  count(*) FILTER (WHERE class = 'helipad')         AS infra_helipad
           FROM infra_uniq
           GROUP BY 1"""
    )

    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE feat AS
            SELECT b.GEOID,
                   '{region}'                                          AS region,
                   coalesce(a.poi_total, 0)                            AS poi_total,
                   coalesce(a.poi_fire, 0)                             AS poi_fire,
                   coalesce(a.poi_ems, 0)                              AS poi_ems,
                   coalesce(a.poi_schools, 0)                          AS poi_schools,
                   coalesce(a.poi_hospital, 0)                         AS poi_hospital,
                   coalesce(a.poi_emergency, 0)                        AS poi_emergency,
                   coalesce(a.poi_education, 0)                        AS poi_education,
                   coalesce(a.poi_health, 0)                           AS poi_health,
                   coalesce(a.poi_retail, 0)                           AS poi_retail,
                   coalesce(a.poi_food, 0)                             AS poi_food,
                   coalesce(a.poi_government, 0)                       AS poi_government,
                   coalesce(a.poi_elementary, 0)                       AS poi_elementary,
                   coalesce(a.poi_middle, 0)                           AS poi_middle,
                   coalesce(a.poi_high, 0)                             AS poi_high,
                   coalesce(a.poi_school, 0)                           AS poi_school,
                   coalesce(a.poi_private_school, 0)                   AS poi_private_school,
                   coalesce(a.poi_public_school, 0)                    AS poi_public_school,
                   coalesce(a.poi_college_university, 0)               AS poi_college_university,
                   coalesce(a.poi_police, 0)                           AS poi_police,
                   coalesce(i.infra_total, 0)                          AS infra_total,
                   coalesce(i.infra_transit, 0)                        AS infra_transit,
                   coalesce(i.infra_airport, 0)                        AS infra_airport,
                   coalesce(i.infra_total - i.infra_known, 0)          AS infra_other,
                   coalesce(i.infra_bus_stop, 0)                       AS infra_bus_stop,
                   coalesce(i.infra_bus_station, 0)                    AS infra_bus_station,
                   coalesce(i.infra_railway_station, 0)                AS infra_railway_station,
                   coalesce(i.infra_platform, 0)                       AS infra_platform,
                   coalesce(i.infra_airport_facility, 0)               AS infra_airport_facility,
                   coalesce(i.infra_helipad, 0)                        AS infra_helipad,
                   a.confidence_mean,
                   coalesce(a.confidence_count, 0)                     AS confidence_count,
                   CASE WHEN coalesce(a.poi_total, 0) = 0 THEN NULL
                        ELSE 100.0 * a.confidence_count / a.poi_total END AS pct_with_confidence,
                   t.pop_total, t.ALAND, t.pct_urban, t.ur_class,
                   s.housing_units, st.ruca_primary, st.svi_socioeconomic
            FROM base b
            LEFT JOIN tract t            ON b.GEOID = t.GEOID
            LEFT JOIN poi_agg a          ON b.GEOID = a.GEOID
            LEFT JOIN infra_agg i        ON b.GEOID = i.GEOID
            LEFT JOIN read_parquet('{p["acs"].as_posix()}') s       ON b.GEOID = s.GEOID
            LEFT JOIN read_parquet('{p["strata"].as_posix()}') st   ON b.GEOID = st.GEOID"""
    )

    return check_region(con, region)


def check_region(con: duckdb.DuckDBPyConnection, region: str) -> dict:
    n_sample, n_feat, n_geoid, n_tract = con.execute(
        """SELECT (SELECT count(*) FROM base),
                  (SELECT count(*) FROM feat),
                  (SELECT count(DISTINCT GEOID) FROM feat),
                  (SELECT count(*) FROM tract)"""
    ).fetchone()

    agg = con.execute(
        """SELECT sum(poi_total),
                  sum(poi_fire + poi_ems + poi_schools),
                  sum(infra_total)
           FROM feat"""
    ).fetchone()

    poi_rows, poi_matched = con.execute(
        "SELECT (SELECT count(*) FROM poi), (SELECT count(DISTINCT id) FROM poi_uniq)"
    ).fetchone()
    hit_rows = con.execute("SELECT count(*) FROM poi_hit").fetchone()[0]
    infra_rows, infra_matched = con.execute(
        "SELECT (SELECT count(*) FROM infra), (SELECT count(DISTINCT id) FROM infra_uniq)"
    ).fetchone()

    bad = con.execute(
        """SELECT count(*) FROM feat
           WHERE poi_total < 0 OR infra_total < 0
              OR poi_total < poi_fire + poi_ems + poi_schools
              OR poi_total < poi_education OR poi_total < poi_retail
              OR poi_total < poi_food OR poi_total < poi_health
              OR poi_total < poi_government"""
    ).fetchone()[0]

    problems = []
    if n_feat != n_sample:
        problems.append(f"filas {n_feat} != sample {n_sample}")
    if n_geoid != n_feat:
        problems.append("GEOID duplicados")
    if n_tract != n_sample:
        problems.append(f"tractos con geometria {n_tract} != sample {n_sample}")
    if bad:
        problems.append(f"{bad} filas con conteos inconsistentes")
    if poi_matched == 0 or infra_matched == 0:
        problems.append("ningun feature espacial asignado")

    stats = {
        "region": region,
        "rows": n_feat,
        "pois_file": poi_rows,
        "pois_assigned": poi_matched,
        "poi_hits_before_dedupe": hit_rows,
        "infra_file": infra_rows,
        "infra_assigned": infra_matched,
        "poi_total_sum": agg[0],
        "critical_sum": agg[1],
        "infra_total_sum": agg[2],
        "problems": problems,
    }
    return stats


def top_categories(con: duckdb.DuckDBPyConnection, region: str, limit: int = 15) -> list[tuple]:
    p = paths(region)["pois"]
    return con.execute(
        f"""SELECT coalesce(categories.primary, '<null>') AS cat, count(*) AS n
            FROM read_parquet('{p.as_posix()}')
            GROUP BY 1 ORDER BY n DESC LIMIT {limit}"""
    ).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS, choices=REGIONS)
    args = ap.parse_args()

    con = connect()
    stats: list[dict] = []

    for i, region in enumerate(args.regions):
        s = build_region(con, region)
        stats.append(s)
        if i == 0:
            con.execute("CREATE OR REPLACE TABLE out AS SELECT * FROM feat LIMIT 0")
        con.execute("INSERT INTO out SELECT * FROM feat")
        print(
            f"[{region}] filas={s['rows']} pois={s['pois_file']} "
            f"asignados={s['pois_assigned']} (hits {s['poi_hits_before_dedupe']}) "
            f"infra={s['infra_file']}/{s['infra_assigned']} "
            f"poi_total={s['poi_total_sum']} criticos={s['critical_sum']} "
            f"infra_total={s['infra_total_sum']} "
            f"{'OK' if not s['problems'] else 'FALLA: ' + '; '.join(s['problems'])}",
            flush=True,
        )
        if s["problems"]:
            return 1

    con.execute(
        f"""COPY (SELECT * FROM out ORDER BY region, GEOID) TO '{OUT.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    print("\nTop categorias por region (categories.primary):")
    for s in stats:
        tops = top_categories(con, s["region"], 10)
        print(f"  {s['region']}: " + ", ".join(f"{c}={n}" for c, n in tops))

    total = con.execute("SELECT count(*), count(DISTINCT region) FROM out").fetchone()
    print(f"\n{OUT.as_posix()}: {total[0]} filas, {total[1]} regiones, "
          f"{OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
