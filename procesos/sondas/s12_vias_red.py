"""s12 — km por tracto, por clase de vía y por red numerada (solo Overture roads).

Mecanismo: el `transport_gap` del organizador compara motorway..secondary de Overture contra las vías
primarias y secundarias de una capa de referencia que este script NO lee. La red numerada que Overture
guarda en `routes` (US:I, US:US, US:<ST>) permite estimar, dentro de Overture, qué parte de esa red quedó
en una clase baja. Todo sale de <region>-overture-roads.parquet y de <region>-census-tracts.parquet.

Salida: datos/intermedios/s12_vias_red_largo.parquet  (GEOID, grupo_clase, grupo_red, km)
Uso:    python procesos/sondas/s12_vias_red.py [regiones]
"""
import sys, time
from pathlib import Path
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
OUT = ROOT / "datos/intermedios/s12_vias_red_largo.parquet"  # formato largo: no va en features_sonda/, donde sondas_v2 hace join por GEOID
SIGN = r":(Historic|Business|Scenic|Alternate|Future|Truck|Bypass|Spur|Loop)$"


def region(con, r):
    t = (ROOT / f"CONCURSO/data/strata/{r}/{r}-census-tracts.parquet").as_posix()
    v = (ROOT / f"CONCURSO/data/reference/{r}/{r}-overture-roads.parquet").as_posix()
    return con.execute(f"""
    WITH tr AS (SELECT GEOID, ST_Transform(geometry,'EPSG:4326','EPSG:5070',always_xy:=true) g FROM read_parquet('{t}')),
    ro AS (
      SELECT class,
        list_transform(list_filter(routes, x -> x.network IS NOT NULL), x -> x.network) AS nets,
        ST_Transform(geometry,'EPSG:4326','EPSG:5070',always_xy:=true) g
      FROM read_parquet('{v}')),
    cl AS (
      SELECT CASE WHEN class IN ('motorway','trunk','primary','secondary') THEN 'alta'
                  WHEN class = 'tertiary' THEN 'tertiary' ELSE 'otra' END AS grupo_clase,
        CASE WHEN len(list_filter(nets, x -> x = 'US:I')) > 0 THEN 'interstate'
             WHEN len(list_filter(nets, x -> x = 'US:US')) > 0 THEN 'us'
             WHEN len(list_filter(nets, x -> regexp_matches(x,'^US:[A-Z]{{2}}$'))) > 0 THEN 'estatal'
             WHEN len(list_filter(nets, x -> regexp_matches(x,'^US:[A-Z]{{2}}:(FM|RM|PR|RE|SP|LP|BU|Loop|Spur)$'))) > 0 THEN 'estatal_sec'
             WHEN len(list_filter(nets, x -> regexp_matches(x,'^US:[A-Z]{{2}}:') AND NOT regexp_matches(x,'{SIGN}'))) > 0 THEN 'condal'
             WHEN len(nets) > 0 THEN 'otra_red' ELSE 'sin_red' END AS grupo_red, g
      FROM ro)
    SELECT t.GEOID, c.grupo_clase, c.grupo_red, sum(ST_Length(ST_Intersection(c.g,t.g)))/1000.0 AS km
    FROM cl c JOIN tr t ON ST_Intersects(c.g,t.g)
    GROUP BY 1,2,3""").df()


def main():
    regs = sys.argv[1:] or REG
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; PRAGMA threads=10; SET memory_limit='14GB'; SET preserve_insertion_order=false;")
    fr = []
    for r in regs:
        t0 = time.time()
        d = region(con, r)
        d["region"] = r
        fr.append(d)
        print(r, len(d), f"{time.time()-t0:.0f}s", flush=True)
    pd.concat(fr, ignore_index=True).to_parquet(OUT, compression="zstd", index=False)
    print("escrito", OUT)


if __name__ == "__main__":
    main()
