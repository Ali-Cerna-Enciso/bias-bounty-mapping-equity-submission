"""L4: tractos con colonias vs sin colonias de south-central-tx, mismo condado x ruralidad.

Fuente de colonias: HUD "Colonias Communities" (Sec. 916 Cranston-Gonzalez), capa
publica de ArcGIS, Date of Coverage 2019, 1,808 comunidades en Texas.
https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/COLONIAS_COMMUNITIES/FeatureServer/2

Corre desde cualquier carpeta:  .venv/Scripts/python.exe procesos/bias_discovery/l4_colonias.py
"""
from __future__ import annotations

import json
import pathlib
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyarrow import parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "datos" / "bias_discovery"
EXT = ROOT / "datos" / "externos" / "hud_colonias"   # consultado 2026-10-30
EXT.mkdir(parents=True, exist_ok=True)
CACHE = EXT / "colonias_hud_tx_2019.geojson"

SERVICE = ("https://services.arcgis.com/VTyQ9soqVukalItT/arcgis/rest/services/"
           "COLONIAS_COMMUNITIES/FeatureServer/2/query")
SEED = 42
BOOT = 2000

TEXAS_COUNTY = {
    "001": "Anderson", "013": "Atascosa", "021": "Bastrop", "027": "Bell",
    "029": "Bexar", "039": "Brazoria", "047": "Brooks", "055": "Caldwell",
    "061": "Cameron", "085": "Collin", "091": "Comal", "099": "Coryell",
    "113": "Dallas", "121": "Denton", "127": "Dimmit", "131": "Duval",
    "141": "El Paso", "157": "Fort Bend", "167": "Galveston", "187": "Guadalupe",
    "201": "Harris", "209": "Hays", "213": "Henderson", "215": "Hidalgo",
    "229": "Hudspeth", "231": "Hunt", "247": "Jim Hogg", "249": "Jim Wells",
    "251": "Johnson", "255": "Karnes", "257": "Kaufman", "273": "Kleberg",
    "283": "La Salle", "303": "Lubbock", "323": "Maverick", "339": "Montgomery",
    "355": "Nueces", "367": "Parker", "377": "Presidio", "397": "Rockwall",
    "409": "San Patricio", "427": "Starr", "439": "Tarrant", "451": "Tom Green",
    "453": "Travis", "465": "Val Verde", "469": "Victoria", "479": "Webb",
    "489": "Willacy", "491": "Williamson", "505": "Zapata",
}


def fetch_colonias() -> gpd.GeoDataFrame:
    if CACHE.exists():
        return gpd.read_file(CACHE)
    feats: list[dict] = []
    offset = 0
    while True:
        q = urllib.parse.urlencode({
            "where": "STUSAB='TX'",
            "outFields": "NAME,ID",
            "outSR": "4326",
            "returnGeometry": "true",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": 1000,
        })
        with urllib.request.urlopen(f"{SERVICE}?{q}", timeout=120) as fh:
            page = json.load(fh)
        got = page.get("features", [])
        feats.extend(got)
        if len(got) < 1000:
            break
        offset += 1000
    CACHE.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return gpd.read_file(CACHE)


def load_tracts() -> gpd.GeoDataFrame:
    f = ROOT / "CONCURSO/data/strata/south-central-tx/south-central-tx-census-tracts.parquet"
    cols = ["GEOID", "COUNTYFP", "pct_urban", "pop_total", "geometry"]
    df = pq.read_table(f, columns=cols).to_pandas()
    df["GEOID"] = df["GEOID"].astype(str)
    df["geometry"] = shapely.from_wkb(df["geometry"].values)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    assert -107 < minx < -93 and 25 < miny < 37, f"CRS de tractos inesperado: {gdf.total_bounds}"
    return gdf


def load_features() -> pd.DataFrame:
    b = pq.read_table(ROOT / "datos/features_base/features_buildings.parquet",
                      columns=["GEOID", "bldg_count"]).to_pandas()
    r = pq.read_table(ROOT / "datos/features_base/features_roads.parquet",
                      columns=["GEOID", "km_named", "km_total_unfiltered"]).to_pandas()
    p = pq.read_table(ROOT / "datos/features_base/features_pois.parquet",
                      columns=["GEOID", "poi_total"]).to_pandas()
    s = pq.read_table(
        ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
        columns=["GEOID", "COUNTYFP", "pct_urban", "pop_total",
                 "usfs_BuildingCount_sum", "carbonplan_buildings", "svi_overall"],
    ).to_pandas()
    for d in (b, r, p, s):
        d["GEOID"] = d["GEOID"].astype(str)
    s = s.drop(columns=["pct_urban", "pop_total"])
    out = b.merge(r, on="GEOID", how="left").merge(p, on="GEOID", how="left")
    return out.merge(s, on="GEOID", how="left")


def build_table() -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    tracts = load_tracts()
    colonias = fetch_colonias()
    minx, miny, maxx, maxy = colonias.total_bounds
    assert -107 < minx < -93 and 25 < miny < 37, f"CRS de colonias inesperado: {colonias.total_bounds}"

    hits = gpd.sjoin(colonias, tracts[["GEOID", "geometry"]], how="inner", predicate="within")
    per_tract = hits.groupby("GEOID").size().rename("n_colonias")

    t = tracts.merge(per_tract, on="GEOID", how="left")
    t["n_colonias"] = t["n_colonias"].fillna(0).astype(int)
    t["has_colonia"] = t["n_colonias"] > 0
    t = t.merge(load_features(), on="GEOID", how="left", suffixes=("", "_f"))

    t["countyfp"] = t["GEOID"].str[:5]
    t["county"] = t["countyfp"].str[2:].map(TEXAS_COUNTY).fillna("TX-" + t["countyfp"].str[2:])
    t["rural"] = np.where(t["pct_urban"] < 0.5, "rural", "urban")

    t["ratio_overture_usfs"] = np.where(t["usfs_BuildingCount_sum"] > 0,
                                        t["bldg_count"] / t["usfs_BuildingCount_sum"], np.nan)
    t["ratio_overture_carbonplan"] = np.where(t["carbonplan_buildings"] > 0,
                                              t["bldg_count"] / t["carbonplan_buildings"], np.nan)
    t["pct_km_sin_nombre"] = np.where(t["km_total_unfiltered"] > 0,
                                     100 * (1 - t["km_named"] / t["km_total_unfiltered"]), np.nan)
    t["poi_per_1k"] = np.where(t["pop_total"] > 0, 1000 * t["poi_total"] / t["pop_total"], np.nan)
    t["bldg_per_1k"] = np.where(t["pop_total"] > 0, 1000 * t["bldg_count"] / t["pop_total"], np.nan)
    return t, tracts, hits


METRICS = ["bldg_per_1k", "ratio_overture_usfs", "ratio_overture_carbonplan",
           "pct_km_sin_nombre", "poi_per_1k"]


def stratum_effects(t: pd.DataFrame, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Diferencia de medianas (colonia - sin colonia) ponderada por cantidad de tractos, dentro de condado x ruralidad."""
    num, den, parts = {}, {}, {}
    for (county, rur), g in t.groupby(["county", "rural"]):
        a = g[g["has_colonia"]]
        b = g[~g["has_colonia"]]
        if len(a) == 0 or len(b) == 0:
            continue
        if rng is not None:
            a = a.iloc[rng.integers(0, len(a), len(a))]
            b = b.iloc[rng.integers(0, len(b), len(b))]
        w = len(a) + len(b)
        for m in METRICS:
            va, vb = a[m].dropna(), b[m].dropna()
            if len(va) == 0 or len(vb) == 0:
                continue
            num[m] = num.get(m, 0.0) + w * (va.median() - vb.median())
            den[m] = den.get(m, 0.0) + w
            parts.setdefault(m, []).append((county, rur, len(va), len(vb), va.median(), vb.median()))
    return {m: num[m] / den[m] for m in num}, parts


def main() -> None:
    t, tracts, hits = build_table()
    print("=" * 78)
    print("COBERTURA")
    print("=" * 78)
    print(f"tractos south-central-tx: {len(tracts)}  con >=1 colonia HUD: {int(t['has_colonia'].sum())}")
    print(f"comunidades colonia de Texas en la fuente: {hits['ID'].nunique()}  dentro de la region: {len(hits)}")
    print(f"condados en la region: {t['countyfp'].nunique()}  con colonias: {t.loc[t['has_colonia'], 'countyfp'].nunique()}")

    col = t[t["has_colonia"]].groupby("county").agg(
        tractos=("GEOID", "size"), colonias=("n_colonias", "sum"), poblacion=("pop_total", "sum"))
    col = col.sort_values("colonias", ascending=False)
    print("\nCondados de la region con colonias (tractos / comunidades / poblacion):")
    print(col.to_string())

    print("\nTop comunidades por nombre (primeras 12):")
    print(hits["NAME"].value_counts().head(12).to_string())

    work = t[(t["pop_total"] > 0) & t["county"].isin(col.index)].copy()
    print(f"\nDenominador del contraste: {len(work)} tractos con poblacion en condados con colonias "
          f"({int(work['has_colonia'].sum())} con colonia, {int((~work['has_colonia']).sum())} sin colonia)")

    print("\n" + "=" * 78)
    print("BRUTO (sin control): medianas colonia vs sin colonia")
    print("=" * 78)
    raw = {}
    for m in METRICS:
        a, b = work.loc[work["has_colonia"], m].dropna(), work.loc[~work["has_colonia"], m].dropna()
        raw[m] = a.median() - b.median()
        print(f"{m:28s} con={a.median():9.4f} (n={len(a):4d})  sin={b.median():9.4f} (n={len(b):4d})  dif={raw[m]:+8.4f}")

    eff, parts = stratum_effects(work)
    rng = np.random.default_rng(SEED)
    boots = {m: [] for m in METRICS}
    for _ in range(BOOT):
        e, _ = stratum_effects(work, rng)
        for m in METRICS:
            if m in e:
                boots[m].append(e[m])

    print("\n" + "=" * 78)
    print(f"ESTRATIFICADO condado x ruralidad (ponderado por n de tractos, bootstrap {BOOT}, semilla {SEED})")
    print("=" * 78)
    print(f"{'metrica':28s} {'bruto':>9s} {'ajustado':>9s} {'IC95 inf':>9s} {'IC95 sup':>9s}  signo IC")
    rows = []
    for m in METRICS:
        bs = np.array(boots[m])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        signo = "si" if (lo > 0 or hi < 0) else "no"
        print(f"{m:28s} {raw[m]:+9.4f} {eff.get(m, np.nan):+9.4f} {lo:+9.4f} {hi:+9.4f}  {signo}")
        rows.append(dict(metrica=m, bruto=raw[m], ajustado=eff.get(m), ic95_inf=lo, ic95_sup=hi,
                         significativo=signo, n_boot=len(bs)))

    print("\n" + "=" * 78)
    print("POR ESTRATO (solo condado x ruralidad con ambos grupos)")
    print("=" * 78)
    print(f"{'condado':12s} {'ruralidad':9s} {'n_con':>5s} {'n_sin':>5s} "
          f"{'ov/usfs_con':>11s} {'ov/usfs_sin':>11s} {'%sinnom_c':>9s} {'%sinnom_s':>9s} "
          f"{'poi_con':>8s} {'poi_sin':>8s}")
    strat_rows = []
    for county, rur, na, nb, ma, mb in parts["ratio_overture_usfs"]:
        g = work[(work["county"] == county) & (work["rural"] == rur)]
        A, B = g[g["has_colonia"]], g[~g["has_colonia"]]
        print(f"{county:12s} {rur:9s} {na:5d} {nb:5d} {ma:11.3f} {mb:11.3f} "
              f"{A['pct_km_sin_nombre'].median():9.1f} {B['pct_km_sin_nombre'].median():9.1f} "
              f"{A['poi_per_1k'].median():8.1f} {B['poi_per_1k'].median():8.1f}")
        strat_rows.append(dict(condado=county, ruralidad=rur, n_con=na, n_sin=nb,
                               ov_usfs_con=ma, ov_usfs_sin=mb,
                               pct_sinnom_con=A["pct_km_sin_nombre"].median(),
                               pct_sinnom_sin=B["pct_km_sin_nombre"].median(),
                               poi_con=A["poi_per_1k"].median(), poi_sin=B["poi_per_1k"].median(),
                               pop_con=int(A["pop_total"].sum()), pop_sin=int(B["pop_total"].sum()),
                               bldg1k_con=A["bldg_per_1k"].median(), bldg1k_sin=B["bldg_per_1k"].median(),
                               dif_ov_usfs=ma - mb))

    long = []
    for r in rows:
        long.append(dict(tipo="agregado", condado="TODOS", ruralidad="TODOS", metrica=r["metrica"],
                         n_con=int(work["has_colonia"].sum()), n_sin=int((~work["has_colonia"]).sum()),
                         pop_con=int(work.loc[work["has_colonia"], "pop_total"].sum()),
                         pop_sin=int(work.loc[~work["has_colonia"], "pop_total"].sum()),
                         med_con=np.nan, med_sin=np.nan, dif_bruto=r["bruto"],
                         dif_ajustado=r["ajustado"], ic95_inf=r["ic95_inf"], ic95_sup=r["ic95_sup"],
                         significativo=r["significativo"]))
    for s in strat_rows:
        long.append(dict(tipo="estrato", condado=s["condado"], ruralidad=s["ruralidad"],
                         metrica="ratio_overture_usfs", n_con=s["n_con"], n_sin=s["n_sin"],
                         pop_con=s["pop_con"], pop_sin=s["pop_sin"],
                         med_con=s["ov_usfs_con"], med_sin=s["ov_usfs_sin"],
                         dif_bruto=np.nan, dif_ajustado=s["dif_ov_usfs"],
                         ic95_inf=np.nan, ic95_sup=np.nan, significativo=""))
    csv = pd.DataFrame(long)[["tipo", "condado", "ruralidad", "metrica", "n_con", "n_sin",
                             "pop_con", "pop_sin", "med_con", "med_sin", "dif_bruto",
                             "dif_ajustado", "ic95_inf", "ic95_sup", "significativo"]]
    csv.to_csv(OUT / "l4_colonias.csv", index=False)
    print(f"\nescrito: {OUT / 'l4_colonias.csv'}")


if __name__ == "__main__":
    main()
