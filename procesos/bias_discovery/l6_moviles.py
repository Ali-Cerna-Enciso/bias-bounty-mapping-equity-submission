"""L6: tractos con mucha vivienda movil tienen menos edificios en Overture de lo
esperado por su vivienda? Referencia independiente: ACS 2023 5yr tabla B25024.

Corre desde cualquier carpeta:  .venv/Scripts/python.exe procesos/bias_discovery/l6_moviles.py
"""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(RAIZ, "datos/externos/census_acs")        # descarga propia, consultado 2026-10-30
GAZ = os.path.join(RAIZ, "datos/externos/census_geo")        # gazetteer, consultado 2026-10-30
SALIDA = os.path.join(RAIZ, "datos/bias_discovery/l6_moviles.csv")

# ACS 2023 5-year, tabla B25024 (unidades por estructura), bulk table-based summary file.
URL_B25024 = ("https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
              "table-based-SF/data/5YRData/acsdt5y2023-b25024.dat")
URL_COUNTIES = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                "2023_Gazetteer/2023_Gaz_counties_national.zip")
SEMILLA = 42


def descargar(url, destino):
    if os.path.exists(destino) and os.path.getsize(destino) > 0:
        return destino
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    print(f"  descargando {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(destino, "wb") as f:
        f.write(r.read())
    return destino


def cargar_acs():
    os.makedirs(EXT, exist_ok=True)
    dat = descargar(URL_B25024, os.path.join(EXT, "acsdt5y2023-b25024.dat"))
    cols = ["GEO_ID", "B25024_E001", "B25024_E002", "B25024_E003", "B25024_E004",
            "B25024_E005", "B25024_E006", "B25024_E007", "B25024_E008",
            "B25024_E009", "B25024_E010", "B25024_E011"]
    acs = pd.read_csv(dat, sep="|", usecols=cols, dtype={"GEO_ID": str})
    acs = acs[acs.GEO_ID.str.startswith("1400000US")].copy()
    acs["GEOID"] = acs.GEO_ID.str.slice(9)
    acs = acs.rename(columns={"B25024_E001": "hu_acs", "B25024_E002": "det",
                              "B25024_E003": "att", "B25024_E010": "mh",
                              "B25024_E011": "rv"})
    acs["u25"] = acs[["B25024_E004", "B25024_E005"]].sum(axis=1)
    acs["u5p"] = acs[["B25024_E006", "B25024_E007", "B25024_E008",
                      "B25024_E009"]].sum(axis=1)
    acs["hu_multifam"] = acs.u25 + acs.u5p
    return acs.drop(columns=["B25024_E004", "B25024_E005", "B25024_E006",
                             "B25024_E007", "B25024_E008", "B25024_E009"])


def cargar_condados():
    z = descargar(URL_COUNTIES, os.path.join(GAZ, "2023_Gaz_counties_national.zip"))
    with zipfile.ZipFile(z) as zf:
        nombre = [n for n in zf.namelist() if n.lower().endswith(".txt")][0]
        gaz = pd.read_csv(io.BytesIO(zf.read(nombre)), sep="\t",
                          dtype={"GEOID": str}, encoding="latin-1")
    gaz.columns = [c.strip() for c in gaz.columns]
    return gaz[["GEOID", "NAME"]].rename(columns={"GEOID": "county", "NAME": "county_name"})


def construir():
    fb = pd.read_parquet(os.path.join(RAIZ, "datos/features_base/features_buildings.parquet"),
                         columns=["GEOID", "region", "bldg_count", "bldg_area_median_m2",
                                  "bldg_area_p90_m2", "housing_units", "pop_total",
                                  "ALAND", "pct_urban", "src_missing_share",
                                  "confidence_mean"])
    fb["GEOID"] = fb.GEOID.astype(str)
    fb["county"] = fb.GEOID.str[:5]

    strata = pd.read_parquet(
        os.path.join(RAIZ, "CONCURSO/data/strata/national/national-strata-tract-table.parquet"),
        columns=["GEOID", "COUNTYFP", "STATEFP", "svi_overall", "aiannh_name"])
    strata["GEOID"] = strata.GEOID.astype(str)

    df = (fb.merge(cargar_acs(), on="GEOID", how="left")
            .merge(strata, on="GEOID", how="left")
            .merge(cargar_condados(), on="county", how="left"))

    df["expected"] = df[["det", "att", "mh"]].sum(axis=1, min_count=3)
    df = df[df.expected.notna()].copy()
    df["mh_share"] = df.mh / df.expected
    df["log_exp"] = np.log(df.expected.clip(lower=1))
    df["log_bldg"] = np.log(df.bldg_count.clip(lower=1))
    df["bldg_per_exp"] = df.bldg_count / df.expected
    df["bldg_per_hu"] = df.bldg_count / df.housing_units.replace(0, np.nan)
    km2 = df.ALAND / 1e6
    df["hu_km2"] = df.housing_units / km2.replace(0, np.nan)
    df["band"] = pd.cut(df.mh_share, [-0.001, 0.05, 0.15, 0.30, 1.01],
                        labels=["<5%", "5-15%", "15-30%", ">30%"])

    uso = df[df.expected >= 10].copy()       # piso: menos de 10 estructuras es ruido ACS
    uso["mf_share"] = uso.hu_multifam / uso.hu_acs.replace(0, np.nan)
    uso["decil"] = _decil_densidad(uso)
    uso["grupo"] = uso.county + "|" + uso.decil
    # condados (o tramos) con menos de 3 tractos: el tramo se absorbe entero, así que
    # la FE queda a nivel condado y el tracto sigue aportando a la pendiente.
    for _ in range(2):
        chico = uso.groupby("grupo").GEOID.transform("size") < 3
        uso.loc[chico, "decil"] = "all"
        uso["grupo"] = uso.county + "|" + uso.decil
    return df, uso


def _decil_densidad(d):
    """Decil de densidad de vivienda dentro del condado, por rango (sin qcut: hay
    condados con <10 tractos y empates). Condados con <10 tractos: un solo grupo."""
    r = d.groupby("county").hu_km2.rank(method="first")
    n = d.groupby("county").hu_km2.transform("size")
    dec = np.minimum(np.floor((r - 1) / n * 10), 9).astype(int).astype(str)
    return dec.where(n >= 10, "all")


def _dentro_ols(y, X, g, cl):
    kg = g.max() + 1
    cnt = np.bincount(g, minlength=kg)

    def dentro(m):
        return m - (np.bincount(g, weights=m, minlength=kg) / cnt)[g]

    yw = dentro(y)
    Xw = np.column_stack([dentro(X[:, j]) for j in range(X.shape[1])])
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ Xw.T @ yw
    u = yw - Xw @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        m = cl == c
        s = Xw[m].T @ u[m]
        meat += np.outer(s, s)
    gc = len(np.unique(cl))
    n = len(y)
    k = X.shape[1] + kg
    V = XtX_inv @ meat @ XtX_inv * ((gc / (gc - 1)) * ((n - 1) / (n - k)))
    return b, np.sqrt(np.diag(V)), u


def ols_fe(df, fe_cols, x=("log_exp", "mh_share"), cluster="county"):
    """OLS con FE absorbidos (por dentro del grupo), SE agrupados por condado.
    cluster=None -> errores robustos a heterocedasticidad (cada fila su cluster)."""
    d = df.dropna(subset=[*x, "log_bldg"]).copy()
    g = (d.groupby(list(fe_cols), observed=True).ngroup().to_numpy() if fe_cols
         else np.zeros(len(d), int))
    cl = d[cluster].to_numpy() if cluster else np.arange(len(d))
    if cluster:
        d = d.dropna(subset=[cluster])
        g = (d.groupby(list(fe_cols), observed=True).ngroup().to_numpy() if fe_cols
             else np.zeros(len(d), int))
        cl = d[cluster].to_numpy()
    b, se, u = _dentro_ols(d.log_bldg.to_numpy(float), d[list(x)].to_numpy(float), g, cl)
    yv = d.log_bldg.to_numpy(float)
    kg = g.max() + 1
    cnt = np.bincount(g, minlength=kg)
    dentro_y = yv - (np.bincount(g, weights=yv, minlength=kg) / cnt)[g]
    ident = cnt[g] >= 2
    r2 = (np.sum((dentro_y ** 2 - u ** 2)[ident]) / np.sum(dentro_y[ident] ** 2)
          if ident.any() else np.nan)
    res = pd.DataFrame({"coef": b, "se": se}, index=list(x))
    res["t"] = res.coef / res.se
    d["_resid"] = u
    return res, d, dict(n=len(d), n_condados=d[cluster].nunique() if cluster else np.nan,
                        n_ident=int(ident.sum()), n_grupos=len(np.unique(g)), r2_dentro=r2)


def bootstrap_fe(df, fe_cols, reps=400, semilla=SEMILLA, x="mh_share", extra=None):
    """Cluster bootstrap por condado; las FE se rearman con la muestra de condados."""
    xs = ["log_exp", x] + ([extra] if extra else [])
    d = df.dropna(subset=[*xs, "log_bldg"]).copy()
    d["_dg"] = d.groupby(list(fe_cols), observed=True).ngroup()
    condados = d.county.unique()
    idx_c = {c: np.flatnonzero(d.county.to_numpy() == c) for c in condados}
    yv, X, dg = d.log_bldg.to_numpy(float), d[list(xs)].to_numpy(float), d._dg.to_numpy()
    rng = np.random.default_rng(semilla)
    est = np.empty(reps)
    for r in range(reps):
        elegidos = rng.choice(condados, size=len(condados), replace=True)
        idx = np.concatenate([idx_c[c] for c in elegidos])
        inst = np.concatenate([np.full(len(idx_c[c]), i) for i, c in enumerate(elegidos)])
        g = np.unique(inst.astype(np.int64) * 1000 + dg[idx], return_inverse=True)[1]
        try:
            est[r] = _dentro_ols(yv[idx], X[idx], g, inst)[0][1]
        except np.linalg.LinAlgError:
            est[r] = np.nan
    est = est[~np.isnan(est)]
    base = ols_fe(df, fe_cols, x=tuple(xs))[0].loc[x, "coef"]
    return base, np.sort(est)[[int(0.025 * len(est)), int(0.975 * len(est)) - 1]]


def tramos(d, etiqueta):
    filas = []
    for b, g in d.groupby("band", observed=True):
        filas.append({
            "grupo": etiqueta, "tramo_mh": str(b), "n": len(g),
            "poblacion": int(g.pop_total.sum()),
            "bldg_por_estructura": g.bldg_count.sum() / g.expected.sum(),
            "bldg_por_vivienda": g.bldg_count.sum() / g.housing_units.sum(),
            "mh_share_medio": g.mh_share.mean(), "svi_medio": g.svi_overall.mean(),
        })
    return pd.DataFrame(filas)


def main():
    print("=" * 78)
    print("L6 VIVIENDA MOVIL vs EDIFICIOS OVERTURE  (referencia ACS 2023 5yr B25024)")
    print("=" * 78)
    df, uso = construir()
    print(f"tractos del set: {len(df)}  | usados (ACS ok, esperadas>=10): {len(uso)}  | "
          f"condados: {uso.county.nunique()}")
    print(f"móviles ACS: {int(df.mh.sum()):,} de {int(df.expected.sum()):,} estructuras "
          f"esperadas = {df.mh.sum() / df.expected.sum():.3%}")
    print(f"edificios Overture: {int(uso.bldg_count.sum()):,} | viviendas ACS: "
          f"{int(uso.hu_acs.sum()):,}")

    print("\n--- 1. TASAS POR TRAMO DE SHARE DE MOVILES (todos) ---")
    t = tramos(uso, "todos")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n--- 2. REGRESION log(edificios) ~ log(esperadas) + share_movil "
          "(SE agrupados por condado) ---")
    filas_coef = []
    for etq, fe, sub in [
        ("sin FE", [], uso),
        ("FE condado", ["county"], uso),
        ("FE condado x decil densidad", ["county", "decil"], uso),
    ]:
        r, _, mx = ols_fe(sub, fe)
        print(f"\n[{etq}]  n={mx['n']} condados={mx['n_condados']} R2dentro={mx['r2_dentro']:.3f}")
        print(r.to_string(float_format=lambda v: f"{v:.5f}"))
        for k, row in r.iterrows():
            filas_coef.append({"modelo": etq, "var": k, "coef": row.coef, "se": row.se,
                               "t": row.t, **mx})

    print("\n--- 3. BOOTSTRAP agrupado por condado (400 reps, semilla 42) sobre share_movil, "
          "FE condado x decil ---")
    b, ic = bootstrap_fe(uso, ["county", "decil"])
    print(f"coef={b:.5f}  IC95%=[{ic[0]:.5f}, {ic[1]:.5f}]  excluye 0: "
          f"{'SI' if (ic[0] > 0 or ic[1] < 0) else 'NO'}")

    print("\n--- 3b. ROBUSTEZ: control por share multifamiliar y sin multifamiliar ---")
    r, _, mx = ols_fe(uso, ["county", "decil"],
                      x=("log_exp", "mh_share", "mf_share"))
    print(f"[+ control mf_share] n={mx['n']} condados={mx['n_condados']}")
    print(r.to_string(float_format=lambda v: f"{v:.5f}"))
    b3, ic3 = bootstrap_fe(uso, ["county", "decil"], reps=300, extra="mf_share")
    print(f"  bootstrap mh_share: {b3:.5f} IC95%=[{ic3[0]:.5f}, {ic3[1]:.5f}]")
    solo = uso[uso.mf_share.fillna(0) < 0.10]
    r, _, mx = ols_fe(solo, ["county", "decil"])
    print(f"\n[solo tractos con multifamiliar <10%] n={mx['n']} condados={mx['n_condados']}")
    print(r.to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n--- 4. RURAL / URBANO / MARICOPA (FE condado x decil) ---")
    cortes = [("rural pct_urban<0.5", uso.pct_urban < 0.5),
              ("urbano pct_urban>=0.5", uso.pct_urban >= 0.5),
              ("sin Maricopa", uso.county != "04013")]
    for etq, mask in cortes:
        sub = uso[mask]
        r, _, mx = ols_fe(sub, ["county", "decil"])
        bb, ii = bootstrap_fe(sub, ["county", "decil"], reps=300)
        print(f"\n[{etq}] n={mx['n']} condados={mx['n_condados']} "
              f"n_ident={mx['n_ident']} R2dentro={mx['r2_dentro']:.3f}")
        print(r.to_string(float_format=lambda v: f"{v:.5f}"))
        print(f"  bootstrap share_movil: {bb:.5f} IC95%=[{ii[0]:.5f}, {ii[1]:.5f}]")
        print(tramos(sub, etq).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[Maricopa 04013] un solo condado: FE decil densidad + errores HC (sin bootstrap "
          "por condado)")
    mar = uso[uso.county == "04013"]
    r, _, mx = ols_fe(mar, ["decil"], cluster=None)
    print(f"n={mx['n']} R2dentro={mx['r2_dentro']:.3f}")
    print(r.to_string(float_format=lambda v: f"{v:.5f}"))
    print(tramos(mar, "Maricopa").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n--- 4b. POR REGION (FE condado x decil) ---")
    for reg, g in uso.groupby("region", observed=True):
        if g.county.nunique() < 5:
            continue
        r, _, mx = ols_fe(g, ["county", "decil"])
        bb, ii = bootstrap_fe(g, ["county", "decil"], reps=300)
        print(f"  {reg:<16} n={mx['n']:5d} cond={mx['n_condados']:3d} "
              f"coef_mh={r.loc['mh_share', 'coef']:+.4f} (se {r.loc['mh_share', 'se']:.4f}) "
              f"IC95=[{ii[0]:+.4f},{ii[1]:+.4f}]  bldg/est={g.bldg_count.sum() / g.expected.sum():.2f}")

    print("\n--- 5. 15 TRACTOS MAS EXTREMOS (residuo del modelo FE condado x decil; "
          "share_movil>=15%, viviendas>=200, multifamiliar<20%) ---")
    _, fitted, _ = ols_fe(uso, ["county", "decil"])
    ext = fitted[(fitted.mh_share >= 0.15) & (fitted.housing_units >= 200)
                 & (fitted.mf_share.fillna(0) < 0.20)]
    abajo = ext.nsmallest(15, "_resid")
    cols = ["GEOID", "county_name", "region", "mh_share", "housing_units", "pop_total",
            "expected", "bldg_count", "bldg_per_exp", "_resid"]
    print(abajo[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  (contraste) extremos por exceso de edificios:")
    print(ext.nlargest(5, "_resid")[cols[:4] + ["county_name", "expected", "bldg_count",
                                               "bldg_per_exp", "_resid"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n--- 6. CHEQUEO DE FUSION DE FOOTPRINTS (área del edificio si se fusionan "
          "debería SUBIR con el share) ---")
    for b_, g in uso.groupby("band", observed=True):
        print(f"  {b_:>6}  n={len(g):5d}  area mediana edificio={g.bldg_area_median_m2.median():6.1f} m2"
              f"  p90={g.bldg_area_p90_m2.median():7.1f} m2"
              f"  src_missing={g.src_missing_share.mean():.3f}"
              f"  confidence={g.confidence_mean.mean():.3f}"
              f"  bldg/estructura={g.bldg_count.sum() / g.expected.sum():.2f}")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    uso.to_csv(SALIDA, index=False)
    print(f"\n-> {SALIDA}  ({len(uso)} filas)")


if __name__ == "__main__":
    main()
