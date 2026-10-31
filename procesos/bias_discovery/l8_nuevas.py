"""L8: tractos de construccion reciente vs rezago de cobertura Overture.

Referencia independiente: ACS 2023 5-year, tablas B25034 (ano de construccion)
y B25024 (unidades por estructura), table-based summary file (sin API key).
Correr: .venv/Scripts/python.exe procesos/bias_discovery/l8_nuevas.py
"""
import pathlib
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BASE = pathlib.Path(__file__).resolve().parents[2]
CENSUS = BASE / "datos/externos/census_acs"            # descarga propia, consultado 2026-10-30
GAZ = BASE / "datos/externos/census_geo"               # gazetteer, consultado 2026-10-30
OUT = BASE / "datos/bias_discovery/l8_nuevas.csv"
STATES = {"04", "06", "35", "40", "48", "53"}
SEED = 42
MIN_STRUCT = 20
MIN_KM = 1.0


def load_inputs():
    b = pd.read_parquet(
        BASE / "datos/features_base/features_buildings.parquet",
        columns=["GEOID", "region", "bldg_count", "src_microsoft_share",
                 "confidence_mean", "housing_units", "pop_total", "ALAND",
                 "pct_urban", "ur_class"],
    )
    r = pd.read_parquet(
        BASE / "datos/features_base/features_roads.parquet",
        columns=["GEOID", "km_named", "km_total_unfiltered", "km_residential", "km_highway"],
    )
    s = pd.read_parquet(
        BASE / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
        columns=["GEOID", "COUNTYFP", "STATEFP", "svi_overall"],
    )
    for d in (b, r, s):
        d["GEOID"] = d["GEOID"].astype(str).str.zfill(11)
    df = b.merge(r, on="GEOID", how="left", validate="one_to_one")
    df = df.merge(s, on="GEOID", how="left", validate="one_to_one")
    return df


def read_acs(table, cols):
    path = CENSUS / f"acsdt5y2023-{table.lower()}.dat"
    if not path.exists():
        url = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
               f"table-based-SF/data/5YRData/acsdt5y2023-{table.lower()}.dat")
        CENSUS.mkdir(parents=True, exist_ok=True)
        print(f"  descargando {url}")
        with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
    d = pd.read_csv(path, sep="|", dtype=str, usecols=["GEO_ID"] + cols)
    d["GEOID"] = d["GEO_ID"].str[-11:]
    d = d[d["GEO_ID"].str.startswith("1400000US") & d["GEOID"].str[:2].isin(STATES)]
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d[["GEOID"] + cols]


def add_acs(df):
    t34 = [f"B25034_E{i:03d}" for i in (1, 2, 3)] + [f"B25034_M{i:03d}" for i in (1, 2)]
    t24 = [f"B25024_E{i:03d}" for i in (1, 2, 3, 10)]
    a34 = read_acs("b25034", t34)
    a24 = read_acs("b25024", t24)
    df = df.merge(a34, on="GEOID", how="left").merge(a24, on="GEOID", how="left")

    tot = df["B25034_E001"]
    df["share_2020p"] = df["B25034_E002"] / tot.where(tot > 0)
    df["share_1019"] = df["B25034_E003"] / tot.where(tot > 0)
    # MOE de la proporcion con aproximacion de Census (numerador contenido en denominador)
    m_num, m_den = df["B25034_M002"], df["B25034_M001"]
    p = df["share_2020p"]
    inner = m_num ** 2 - (p ** 2) * (m_den ** 2)
    df["moe_share_2020p"] = np.sqrt(inner.clip(lower=0)) / tot.where(tot > 0)

    df["struct_expected"] = df[["B25024_E002", "B25024_E003", "B25024_E010"]].sum(axis=1, min_count=3)
    df["bldg_per_struct"] = df["bldg_count"] / df["struct_expected"].where(df["struct_expected"] >= MIN_STRUCT)
    df["unnamed_share"] = 1 - df["km_named"] / df["km_total_unfiltered"].where(df["km_total_unfiltered"] >= MIN_KM)
    df["density"] = df["housing_units"] / (df["ALAND"] / 2.59e6).where(df["ALAND"] > 0)
    df["county"] = df["GEOID"].str[:5]
    df["urban"] = np.where(df["ur_class"].eq("Rural"), "rural", "urbano")
    dec = (
        df.groupby("county")["density"]
        .transform(lambda x: pd.qcut(x.rank(method="first", na_option="bottom"), 10,
                                     labels=False, duplicates="drop"))
    )
    df["county_decile"] = df["county"] + "_" + dec.fillna(-1).astype(int).astype(str)
    return df


def counties_names():
    g = pd.read_csv(GAZ / "2023_Gaz_counties_national.txt", sep="\t", dtype={"GEOID": str},
                    usecols=["GEOID", "NAME", "USPS"], engine="python")
    g["GEOID"] = g["GEOID"].str.strip().str.zfill(5)
    return dict(zip(g["GEOID"], g["NAME"].str.strip() + " (" + g["USPS"].str.strip() + ")"))


def fit(d, y, xvars):
    """EF de condado x decil absorbidos por demeaning; SE agrupados por condado.

    El demeaning da el estimador exacto de FE; la correccion de grados de libertad
    descuenta los grupos absorbidos (G) porque el OLS demeaneado no los ve.
    """
    d = d.dropna(subset=[y] + xvars).copy()
    if d["county"].nunique() < 4 or len(d) < 100:
        return None
    g = d.groupby("county_decile", observed=True)
    yd = (d[y] - g[y].transform("mean")).to_numpy()
    X = np.column_stack([(d[v] - g[v].transform("mean")).to_numpy() for v in xvars])
    n = len(d)
    G = d["county_decile"].nunique()
    k = X.shape[1]
    res = sm.OLS(yd, X).fit(cov_type="cluster", cov_kwds={"groups": d["county"].to_numpy()})
    scale = np.sqrt(n / (n - G - k))
    df_t = n - G - k
    out = []
    for i, v in enumerate(xvars):
        b = res.params[i]
        se = res.bse[i] * scale
        tstat = b / se
        p = 2 * stats.t.sf(abs(tstat), df_t)
        half = stats.t.ppf(0.975, df_t) * se
        out.append(dict(term=v, coef=b, se=se, lo=b - half, hi=b + half, p=p))
    return out, n, d["county"].nunique(), G


def main():
    df = add_acs(load_inputs())
    names = counties_names()
    print(f"tractos: {len(df)}  regiones: {df['region'].nunique()}  condados: {df['county'].nunique()}")
    print(f"con bldg_per_struct: {df['bldg_per_struct'].notna().sum()}  con unnamed_share: {df['unnamed_share'].notna().sum()}")
    print(f"share_2020p nulo: {df['share_2020p'].isna().sum()}  MOE>share (indistinguible de 0): "
          f"{(df['moe_share_2020p'] > df['share_2020p']).mean():.3f}  MOE mediano share: {df['moe_share_2020p'].median():.4f}")
    print(f"share_2020p medio: {df['share_2020p'].mean():.4f}  share_1019 medio: {df['share_1019'].mean():.4f}")

    rows = []

    def reg(tag, d, y, xvars):
        r = fit(d, y, xvars)
        if r is None:
            print(f"\n[{tag}] {y}: muestra insuficiente")
            return
        out, n, ncty, G = r
        for it in out:
            print(f"[{tag}] {y} ~ {it['term']}  coef={it['coef']:+.4f}  se={it['se']:.4f}  "
                  f"IC95=[{it['lo']:+.4f},{it['hi']:+.4f}]  p={it['p']:.4f}  n={n}  condados={ncty}")
            rows.append(dict(bloque=tag, metrica=y, termino=it["term"], coef=it["coef"], se=it["se"],
                             ci_lo=it["lo"], ci_hi=it["hi"], p=it["p"], n=n, n_condados=ncty))

    print("\n=== REGRESIONES (EF condado x decil de densidad, cluster condado) ===")
    print("-- edificios por estructura esperada (log) --")
    df["log_bps"] = np.log(df["bldg_per_struct"].where(df["bldg_per_struct"] > 0))
    reg("todo", df, "log_bps", ["share_2020p"])
    reg("todo", df, "log_bps", ["share_2020p", "share_1019"])
    reg("todo", df, "log_bps", ["share_1019"])
    for reg_name in sorted(df["region"].unique()):
        reg(f"region:{reg_name}", df[df["region"] == reg_name], "log_bps", ["share_2020p"])
    for u in ("urbano", "rural"):
        reg(f"ambito:{u}", df[df["urban"] == u], "log_bps", ["share_2020p"])

    print("-- % de km sin nombre --")
    reg("todo", df, "unnamed_share", ["share_2020p"])
    reg("todo", df, "unnamed_share", ["share_2020p", "share_1019"])
    for reg_name in sorted(df["region"].unique()):
        reg(f"region:{reg_name}", df[df["region"] == reg_name], "unnamed_share", ["share_2020p"])
    for u in ("urbano", "rural"):
        reg(f"ambito:{u}", df[df["urban"] == u], "unnamed_share", ["share_2020p"])

    print("\n=== TRAMOS DE SHARE 2020+ ===")
    df["tramo"] = pd.cut(df["share_2020p"], [-0.001, 0.0001, 0.05, 0.15, 1.0],
                         labels=["0", "0-5%", "5-15%", ">15%"])
    rng = np.random.default_rng(SEED)

    def boot_ci(tramo, metric):
        sub = df[(df["tramo"] == tramo) & df[metric].notna()]
        if len(sub) < 30:
            return (np.nan, np.nan)
        agg = sub.groupby("county")[metric].agg(["sum", "count"])
        s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
        idx = rng.integers(0, len(s), size=(500, len(s)))
        means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        return tuple(np.percentile(means, [2.5, 97.5]))

    for t in ["0", "0-5%", "5-15%", ">15%"]:
        d = df[df["tramo"] == t]
        bps = d["bldg_per_struct"].dropna()
        unm = d["unnamed_share"].dropna()
        lo_b, hi_b = boot_ci(t, "bldg_per_struct")
        lo_u, hi_u = boot_ci(t, "unnamed_share")
        print(f"{t:>6}: n={len(d):4d} condados={d['county'].nunique():2d} pob={int(d['pop_total'].sum()):>8,} "
              f"bps={bps.mean():.3f} [{lo_b:.3f},{hi_b:.3f}] (med {bps.median():.3f})  "
              f"sin_nombre={unm.mean():.3f} [{lo_u:.3f},{hi_u:.3f}]  "
              f"density_med={d['density'].median():.1f}  urbano={100*(d['ur_class']=='Urban').mean():.0f}%")
        rows.append(dict(bloque=f"tramo:{t}", metrica="n", coef=len(d), n=len(d),
                         n_condados=d["county"].nunique()))
        for metric, val, lo, hi in (("bldg_per_struct", bps.mean(), lo_b, hi_b),
                                    ("unnamed_share", unm.mean(), lo_u, hi_u)):
            rows.append(dict(bloque=f"tramo:{t}", metrica=metric, coef=val, ci_lo=lo, ci_hi=hi,
                             n=len(d), n_condados=d["county"].nunique()))

    print("\n--- tramos por region ---")
    for reg_name in sorted(df["region"].unique()):
        d = df[df["region"] == reg_name]
        parts = []
        for t in ["0", "0-5%", "5-15%", ">15%"]:
            dd = d[d["tramo"] == t]
            parts.append(f"{t}: n={len(dd):3d} bps={dd['bldg_per_struct'].mean():.3f} "
                         f"unm={dd['unnamed_share'].mean():.3f}")
        print(f"{reg_name:>16} " + " | ".join(parts))

    print("\n--- tramos por ambito ---")
    for u in ("urbano", "rural"):
        d = df[df["urban"] == u]
        parts = [f"{t}: n={len(d[d['tramo'] == t]):3d} bps={d[d['tramo'] == t]['bldg_per_struct'].mean():.3f} "
                 f"unm={d[d['tramo'] == t]['unnamed_share'].mean():.3f}" for t in ["0", "0-5%", "5-15%", ">15%"]]
        print(f"{u:>8} " + " | ".join(parts))

    print("\n=== 15 TRACTOS MAS EXTREMOS (residuo mas negativo del modelo FE, share 2020+ >= 15%) ===")
    dd = df.dropna(subset=["log_bps", "share_2020p"]).copy()
    gfe = dd.groupby("county_decile", observed=True)
    dd["resid"] = (dd["log_bps"] - gfe["log_bps"].transform("mean")).to_numpy() - \
        fit(dd, "log_bps", ["share_2020p"])[0][0]["coef"] * \
        (dd["share_2020p"] - gfe["share_2020p"].transform("mean")).to_numpy()
    top = dd[(dd["share_2020p"] >= 0.15) & (dd["struct_expected"] >= 100)].nsmallest(15, "resid")
    top["condado_nombre"] = top["county"].map(names)
    cols = ["GEOID", "condado_nombre", "region", "pop_total", "struct_expected", "bldg_count",
            "bldg_per_struct", "share_2020p", "unnamed_share", "resid"]
    print(top[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for _, r in top.iterrows():
        rows.append(dict(bloque="extremo", metrica="log_bps_resid", coef=r["resid"],
                         geo_id=r["GEOID"], condado=r["condado_nombre"], region=r["region"],
                         pop_total=r["pop_total"], struct_expected=r["struct_expected"],
                         bldg_count=r["bldg_count"], bldg_per_struct=r["bldg_per_struct"],
                         share_2020p=r["share_2020p"], unnamed_share=r["unnamed_share"]))

    print("\n--- 8 tractos con share 2020+ mas alto ---")
    hs = df.dropna(subset=["share_2020p"]).nlargest(8, "share_2020p")
    hs["condado_nombre"] = hs["county"].map(names)
    print(hs[["GEOID", "condado_nombre", "pop_total", "struct_expected", "bldg_count",
              "bldg_per_struct", "share_2020p", "moe_share_2020p", "unnamed_share"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== CONSTRUCCION RECIENTE POR REGION (share 2020+ medio y peso) ===")
    g = df.groupby("region").agg(n=("GEOID", "size"), share_2020p=("share_2020p", "mean"),
                                 share_1019=("share_1019", "mean"),
                                 bps=("bldg_per_struct", "mean"),
                                 unnamed=("unnamed_share", "mean"))
    print(g.round(4).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nescrito: {OUT}")


if __name__ == "__main__":
    main()
