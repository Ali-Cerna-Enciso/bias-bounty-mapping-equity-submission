"""L5 - Brecha de representacion de salidas de evacuacion en Overture frente al riesgo de incendio.

Metrica por tracto: tramos de via que tocan el limite del tracto, separados en
ruteables_con_nombre (class en motorway..tertiary y names.primary no vacio) y todos.
brecha = 1 - ruteables/todos. Dos denominadores: todas las clases (brecha_all) y solo
clases vehiculares (brecha_veh), para separar la duda de "cruces sin nombre que son caminos irrelevantes".
"""
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO.as_posix()
OUT = REPO / "datos" / "bias_discovery" / "l5_evacuacion.csv"
REGIONS = ["northern-ca", "eastern-wa"]
HWY = "('motorway','trunk','primary','secondary','tertiary')"
NONVEH = "('footway','path','cycleway','steps','pedestrian','bridleway')"
SEED = 42
NBOOT = 2000

COUNTY = {
    "06007": "Butte CA", "06011": "Colusa CA", "06015": "Del Norte CA", "06017": "El Dorado CA",
    "06021": "Glenn CA", "06023": "Humboldt CA", "06033": "Lake CA", "06035": "Lassen CA",
    "06045": "Mendocino CA", "06049": "Modoc CA", "06055": "Napa CA", "06057": "Nevada CA",
    "06061": "Placer CA", "06063": "Plumas CA", "06067": "Sacramento CA", "06089": "Shasta CA",
    "06091": "Sierra CA", "06093": "Siskiyou CA", "06097": "Sonoma CA", "06101": "Sutter CA",
    "06103": "Tehama CA", "06105": "Trinity CA", "06113": "Yolo CA", "06115": "Yuba CA",
    "53001": "Adams WA", "53003": "Asotin WA", "53005": "Benton WA", "53007": "Chelan WA",
    "53013": "Columbia WA", "53017": "Douglas WA", "53019": "Ferry WA", "53021": "Franklin WA",
    "53023": "Garfield WA", "53025": "Grant WA", "53033": "King WA", "53037": "Kittitas WA",
    "53039": "Klickitat WA", "53041": "Lewis WA", "53043": "Lincoln WA", "53047": "Okanogan WA",
    "53051": "Pend Oreille WA", "53053": "Pierce WA", "53057": "Skagit WA", "53059": "Skamania WA",
    "53061": "Snohomish WA", "53063": "Spokane WA", "53065": "Stevens WA", "53071": "Walla Walla WA",
    "53073": "Whatcom WA", "53075": "Whitman WA", "53077": "Yakima WA",
}

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='8GB'")
con.execute("LOAD spatial")


def crossings(region):
    tr = f"{ROOT}/CONCURSO/data/strata/{region}/{region}-census-tracts.parquet"
    rd = f"{ROOT}/CONCURSO/data/reference/{region}/{region}-overture-roads-unfiltered.parquet"
    con.execute(f"""
        CREATE OR REPLACE TABLE tr AS
        SELECT GEOID, ST_Boundary(ST_Transform(geometry,'EPSG:4326','EPSG:5070', always_xy := true)) AS b
        FROM '{tr}' WHERE geometry IS NOT NULL
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE rd AS
        SELECT id, class, names.primary AS nm,
               ST_Transform(geometry,'EPSG:4326','EPSG:5070', always_xy := true) AS g
        FROM '{rd}' WHERE geometry IS NOT NULL
    """)
    con.execute("DROP INDEX IF EXISTS ri")
    con.execute("CREATE INDEX ri ON rd USING RTREE (g)")
    con.execute(f"""
        CREATE OR REPLACE TABLE cx AS
        SELECT t.GEOID, r.id, r.class,
               r.class NOT IN {NONVEH} AS veh,
               (r.class IN {HWY} AND r.nm IS NOT NULL AND trim(r.nm) <> '') AS rt,
               ST_NumGeometries(ST_Intersection(r.g, t.b)) AS np
        FROM tr t JOIN rd r ON ST_Intersects(r.g, t.b)
    """)
    df = con.execute("""
        SELECT GEOID,
               count(DISTINCT id) AS seg_all,
               count(DISTINCT CASE WHEN veh THEN id END) AS seg_veh,
               count(DISTINCT CASE WHEN rt THEN id END) AS seg_rout,
               count(DISTINCT CASE WHEN class IN ('track','service') THEN id END) AS seg_ts,
               count(DISTINCT CASE WHEN class IN ('motorway','trunk','primary','secondary','tertiary',
                        'residential','living_street','unclassified','unknown') THEN id END) AS seg_str,
               sum(np) AS pt_all,
               sum(CASE WHEN veh THEN np END) AS pt_veh,
               sum(CASE WHEN rt THEN np END) AS pt_rout
        FROM cx GROUP BY 1
    """).df()
    df["region"] = region
    byclass = con.execute("""
        SELECT GEOID, class, count(DISTINCT id) AS n, count(DISTINCT CASE WHEN rt THEN id END) AS n_rt
        FROM cx GROUP BY 1,2
    """).df()
    byclass["region"] = region
    return df, byclass


res = [crossings(r) for r in REGIONS]
cx = pd.concat([r[0] for r in res], ignore_index=True)
cls = pd.concat([r[1] for r in res], ignore_index=True)
for f in (cx, cls):
    f["GEOID"] = f["GEOID"].astype(str).str.zfill(11)

geo = pd.concat([
    con.execute(f"""SELECT GEOID, COUNTYFP, pop_total, pct_urban, ur_class
                    FROM '{ROOT}/CONCURSO/data/strata/{r}/{r}-census-tracts.parquet'""").df()
    for r in REGIONS
], ignore_index=True)
geo["GEOID"] = geo["GEOID"].astype(str).str.zfill(11)

strata = con.execute(f"""
    SELECT GEOID, usfs_CRPS_mean, usfs_WHP_mean, usfs_FLEP4_mean, svi_overall
    FROM '{ROOT}/CONCURSO/data/strata/national/national-strata-tract-table.parquet'
    WHERE substr(GEOID,1,2) IN ('06','53')
""").df()
strata["GEOID"] = strata["GEOID"].astype(str).str.zfill(11)

pois = con.execute(f"""
    SELECT GEOID, poi_fire, poi_ems FROM '{ROOT}/datos/features_base/features_pois.parquet'
""").df()
pois["GEOID"] = pois["GEOID"].astype(str).str.zfill(11)

d = (cx.merge(geo, on="GEOID", how="left")
       .merge(strata, on="GEOID", how="left")
       .merge(pois, on="GEOID", how="left"))
for c in ["seg_all", "seg_veh", "seg_rout", "seg_ts", "seg_str", "pt_all", "pt_veh", "pt_rout"]:
    d[c] = d[c].fillna(0).astype(int)

d["brecha_all"] = np.where(d.seg_all > 0, 1 - d.seg_rout / d.seg_all, np.nan)
d["brecha_veh"] = np.where(d.seg_veh > 0, 1 - d.seg_rout / d.seg_veh, np.nan)
d["brecha_str"] = np.where(d.seg_str > 0, 1 - d.seg_rout / d.seg_str, np.nan)
d["brecha_all_pt"] = np.where(d.pt_all > 0, 1 - d.pt_rout / d.pt_all, np.nan)
d["brecha_veh_pt"] = np.where(d.pt_veh > 0, 1 - d.pt_rout / d.pt_veh, np.nan)
d["pers_por_salida"] = np.where(d.seg_rout > 0, d.pop_total / d.seg_rout, np.nan)
d["pers_por_salida_veh"] = np.where(d.seg_veh > 0, d.pop_total / d.seg_veh, np.nan)
d = d[d.ur_class.notna()].copy()
d["ur"] = d.ur_class
d["condado"] = d.GEOID.str[:5].map(COUNTY).fillna(d.GEOID.str[:5])

# terciles DENTRO de region x ruralidad
d["tercil_riesgo"] = "SD"
d = d[d.usfs_CRPS_mean.notna()].copy()
for _, idx in d.groupby(["region", "ur"]).groups.items():
    d.loc[idx, "tercil_riesgo"] = pd.qcut(d.loc[idx, "usfs_CRPS_mean"], 3,
                                          labels=["T1 bajo", "T2 medio", "T3 alto"]).astype(str)


def boot_med(x, rng):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) < 3:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(x), (NBOOT, len(x)))
    m = np.median(x[idx], axis=1)
    return (np.percentile(m, 2.5), np.percentile(m, 97.5))


rng = np.random.default_rng(SEED)
rows = []
for (reg, ur, t), g in d.groupby(["region", "ur", "tercil_riesgo"], observed=True):
    lo_b, hi_b = boot_med(g.brecha_veh, rng)
    lo_p, hi_p = boot_med(g.pers_por_salida, rng)
    lo_a, hi_a = boot_med(g.brecha_all, rng)
    rows.append(dict(region=reg, ur=ur, tercil=t, n=len(g),
                     crps_mediana=g.usfs_CRPS_mean.median(),
                     salidas_ruteables_med=float(g.seg_rout.median()),
                     brecha_med=g.brecha_veh.median(), brecha_lo=lo_b, brecha_hi=hi_b,
                     brecha_todos_med=g.brecha_all.median(), brecha_todos_lo=lo_a, brecha_todos_hi=hi_a,
                     brecha_sin_ts_med=g.brecha_str.median(),
                     pers_salida_med=g.pers_por_salida.median(), pers_lo=lo_p, pers_hi=hi_p))
cel = pd.DataFrame(rows).sort_values(["region", "ur", "tercil"])

# desplazamiento T3-T1 dentro de region x ruralidad, con IC bootstrap (dos definiciones)
shift = []
for (reg, ur), g in d.groupby(["region", "ur"], observed=True):
    for col, nombre in [("brecha_veh", "veh"), ("brecha_str", "sin_track_service"),
                        ("brecha_all", "todos")]:
        a = g[g.tercil_riesgo == "T3 alto"][col].dropna()
        b = g[g.tercil_riesgo == "T1 bajo"][col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        r2 = np.random.default_rng(SEED)
        dd = [np.median(r2.choice(a, len(a))) - np.median(r2.choice(b, len(b))) for _ in range(NBOOT)]
        shift.append(dict(region=reg, ur=ur, metrica=nombre, n_alto=len(a), n_bajo=len(b),
                          delta_T3_T1=np.median(a) - np.median(b),
                          delta_lo=np.percentile(dd, 2.5), delta_hi=np.percentile(dd, 97.5),
                          brecha_T1=np.median(b), brecha_T3=np.median(a)))
sh = pd.DataFrame(shift)

crit = d[(d.tercil_riesgo == "T3 alto") & (d.pop_total >= 500) & (d.pers_por_salida > 0)].copy()
crit["crit_score"] = crit.pers_por_salida * crit.brecha_veh
top = crit.sort_values("crit_score", ascending=False).head(15)[[
    "GEOID", "region", "condado", "pop_total", "usfs_CRPS_mean", "seg_all", "seg_veh", "seg_rout",
    "brecha_all", "brecha_veh", "pers_por_salida", "poi_fire", "poi_ems", "svi_overall"]]

# control de plausibilidad: Paradise, CA (Butte) - centro 39.7596,-121.6219
par = con.execute("""
    SELECT GEOID FROM 'CONCURSO/data/strata/northern-ca/northern-ca-census-tracts.parquet'
    WHERE ST_Contains(geometry, ST_Point(-121.6219, 39.7596))
""").fetchall()
par_ids = [r[0] for r in par]
ctrl = d[d.GEOID.isin(par_ids)][["GEOID", "condado", "pop_total", "seg_all", "seg_veh", "seg_rout",
                                 "brecha_all", "brecha_veh", "pers_por_salida", "poi_fire", "poi_ems"]].copy()
ctrl["nota"] = "Paradise CA punto 39.7596,-121.6219"

pd.set_option("display.width", 250, "display.max_columns", 50)
print("=== A. Terciles de usfs_CRPS_mean por region x ruralidad (brecha = 1 - ruteables con nombre / vehiculares) ===")
print(cel.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== B. Desplazamiento T3 - T1 de la brecha (IC bootstrap 95%) ===")
print(sh.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== C. 15 tractos mas criticos (T3 de riesgo, pob>=500) ===")
print(top.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n=== D. Control Paradise, CA ===")
print(ctrl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nCorrelacion de Spearman brecha_veh vs usfs_CRPS_mean:")
print(d[["brecha_veh", "brecha_str", "brecha_all", "usfs_CRPS_mean", "usfs_WHP_mean"]].corr(method="spearman").to_string(float_format=lambda v: f"{v:.3f}"))

# E. diagnostico de composicion por clase: % de tramos con nombre dentro de cada clase
cl = cls.merge(d[["GEOID", "region", "ur", "tercil_riesgo"]], on=["GEOID", "region"], how="inner")
comp = (cl.groupby(["region", "tercil_riesgo", "class"], as_index=False)[["n", "n_rt"]].sum())
comp["pct_nombrado"] = comp.n_rt / comp.n
comp["pct_del_total"] = comp.n / comp.groupby(["region", "tercil_riesgo"]).n.transform("sum")
piv = comp.pivot_table(index=["region", "tercil_riesgo"], columns="class",
                       values="pct_nombrado").round(3)
print("\n=== E. % de tramos con names.primary, por clase (donde la clase existe) ===")
print(piv.to_string())
piv2 = comp.pivot_table(index=["region", "tercil_riesgo"], columns="class",
                        values="pct_del_total").round(3)
print("\n=== E2. Reparto de los cruces por clase ===")
print(piv2.to_string())

cel_out = cel.assign(bloque="A_terciles")
sh_out = sh.assign(bloque="B_shift")
top_out = top.assign(bloque="C_top15")
ctrl_out = ctrl.assign(bloque="D_control")
comp_out = comp.assign(bloque="E_clases")
out = pd.concat([cel_out, sh_out, top_out, ctrl_out, comp_out], ignore_index=True, sort=False)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print(f"\nescrito {OUT} ({len(out)} filas)")
