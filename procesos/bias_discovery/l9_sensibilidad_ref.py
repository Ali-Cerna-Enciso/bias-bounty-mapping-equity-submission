"""Sensibilidad de la cola de distancia a la referencia de salas autónomas.

Referencia base: roster vigente (datos/externos/fsed). Esta prueba agrega las licencias de la lista de Texas de 2020
que no aparecen en el roster vigente a menos de 300 m: pudieron cerrar o pasar a otra licencia. Tratarlas como abiertas
es la lectura más desfavorable para el hallazgo. No cambia la clasificación de ningún POI, solo la distancia al ER real.

Salida: datos/bias_discovery/l9_sensibilidad_ref.csv. Uso: python procesos/bias_discovery/l9_sensibilidad_ref.py
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from statsmodels.stats.contingency_tables import StratifiedTable

ROOT = Path(__file__).resolve().parents[2]
REG = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
ER_RX = r"EMERGENCY (?:ROOM|CENTER|CENTRE|DEPARTMENT|DEPT|CARE|MEDICAL)|\bE\.?R\.?\b|24.{0,3}(?:HOUR|HR).{0,12}EMERGENCY|\bEMERGENCY\b.*\b(?:HOSPITAL|24)"
NO_ER_RX = r"VET|ANIMAL|PET\b|DENTAL|DENTIST|ORTHODONT|EYE\b|OPTOM|PLUMB|ROOF|TOW|LOCKSMITH|RESTORATION|HVAC|ELECTRIC|PHYSICIAN GROUP|STAFFING|BILLING|TRAINING|AMBULANCE"


def xy(lon, lat):
    g = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(5070)
    return np.c_[g.x, g.y]


def mh(d, col):
    tabs = []
    for _, g in d.groupby("condado"):
        a = int((g.rural & g[col]).sum()); b = int((g.rural & ~g[col]).sum())
        c = int((~g.rural & g[col]).sum()); e = int((~g.rural & ~g[col]).sum())
        if (a + b) and (c + e):
            tabs.append(np.array([[a, b], [c, e]]))
    t = StratifiedTable(tabs)
    lo, hi = t.oddsratio_pooled_confint()
    return t.oddsratio_pooled, lo, hi


def main():
    t = pd.read_csv(ROOT / "datos/bias_discovery/l9_tractos.csv", dtype={"GEOID": str, "condado": str}, encoding="utf-8-sig")
    t["rural"] = t.rural.astype(bool)
    c = pd.concat([gpd.read_parquet(ROOT / f"CONCURSO/data/strata/{r}/{r}-census-tracts.parquet", columns=["GEOID", "geometry"]) for r in REG])
    c = c.to_crs(5070); c["x"] = c.geometry.centroid.x; c["y"] = c.geometry.centroid.y
    t = t.merge(c[["GEOID", "x", "y"]], on="GEOID")

    nuevo = pd.read_csv(ROOT / "datos/externos/fsed/fsed_referencia.csv")
    viejo = [f["geometry"]["coordinates"] for f in json.load(open(ROOT / "datos/externos/fser/tx_fser_2020.geojson", encoding="utf-8"))["features"] if f.get("geometry")]
    viejo = np.array([p for p in viejo if p[0] is not None and p[1] is not None], float)
    pv, pn = xy(viejo[:, 0], viejo[:, 1]), xy(nuevo.lon, nuevo.lat)
    solo = pv[cKDTree(pn).query(pv)[0] > 300]
    print(f"lista 2020: {len(pv)} licencias; {len(solo)} sin licencia vigente a <300 m")

    d = cKDTree(solo).query(t[["x", "y"]].values)[0] / 1000
    t["real2"] = np.minimum(t.real_km, d)
    t["extra2"] = (t.real2 - t.poi_km).clip(lower=0)
    filas = []
    for T in (10, 20):
        for nom, km in (("base", t.km_extra), ("con_lista_2020", t.extra2)):
            t["e"] = t.error.astype(bool) & (km >= T)
            o, lo, hi = mh(t, "e")
            filas.append(dict(referencia=nom, umbral_km=T, rural=t[t.rural].e.mean() * 100, urbano=t[~t.rural].e.mean() * 100, or_mh=o, lo=lo, hi=hi))
    # Prueba más dura: ningún registro cubre los departamentos de urgencias de hospital fuera de campus. Se cuenta como
    # ER real cualquier POI de Overture, de cualquier categoría de salud, cuyo nombre o categoría lo presente como tal.
    # Sobreestima las salas (entra lo mal nombrado), así que el resultado es una cota desfavorable para el hallazgo.
    con = duckdb.connect()
    p = pd.concat([con.execute(f"""SELECT names."primary" AS n, categories."primary" AS c, bbox.xmin AS lon, bbox.ymin AS lat
        FROM read_parquet('{(ROOT / f"CONCURSO/data/reference/{r}/{r}-overture-pois.parquet").as_posix()}')
        WHERE names."primary" IS NOT NULL""").df() for r in REG], ignore_index=True)
    nom = p.n.str.upper()
    salud = p.c.fillna("").str.contains("hospital|emergency|medical|health|urgent|doctor|clinic", regex=True)
    es = (p.c.eq("emergency_room") | (salud & nom.str.contains(ER_RX, regex=True))) & ~nom.str.contains(NO_ER_RX, regex=True)
    q = xy(p.lon[es].values, p.lat[es].values)
    print(f"POIs de Overture que se presentan como sala de urgencias: {len(q)}")
    d3 = cKDTree(q).query(t[["x", "y"]].values)[0] / 1000
    t["real3"] = np.minimum(t.real2, d3)
    t["extra3"] = (t.real3 - t.poi_km).clip(lower=0)
    t["err3"] = t.error.astype(bool) & ~t.poi_nombre.fillna("").str.upper().str.contains(ER_RX, regex=True)
    print(f"tractos con error: base {t.error.astype(bool).mean() * 100:.1f}%, con esa cota {t.err3.mean() * 100:.1f}%")
    for T in (10, 20):
        t["e"] = t.err3 & (t.extra3 >= T)
        o, lo, hi = mh(t, "e")
        filas.append(dict(referencia="cota_poi_urgencias", umbral_km=T, rural=t[t.rural].e.mean() * 100, urbano=t[~t.rural].e.mean() * 100, or_mh=o, lo=lo, hi=hi))
    print(t.assign(e=t.err3 & (t.extra3 >= 10)).groupby("rural").agg(n=("e", "size"), tractos=("e", "sum"), mediana_real=("real3", "median")).round(2))
    r = pd.DataFrame(filas).round(3)
    r.to_csv(ROOT / "datos/bias_discovery/l9_sensibilidad_ref.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
