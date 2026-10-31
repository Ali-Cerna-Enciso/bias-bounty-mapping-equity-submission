"""L3: cobertura Overture por nacion tribal concreta (AIANNH) contra grupos de
comparacion vecinos, y OTSA estadistica vs reserva legal.

Diseno (el "mismo condado y ruralidad" no existe en `eastern-ok`: cada OTSA ocupa
sus condados y los unicos tractos no tribales del estado estan en OKC/Cleveland):
  - nacion_estado  (primario): tractos de la nacion vs tractos de OTRAS naciones del
    mismo estado y misma banda de ruralidad. Aisla la senal entre naciones, que es lo
    que el bit tribal del scorecard no puede ver.
  - nacion_condado (control): mismo condado y banda, comp = otras naciones del condado.
    Solo existe donde dos OTSA comparten condado.
  - ref_no_tribal: OK tribal vs OK no tribal por banda. Se incluye para documentar el
    artefacto de composicion (los no tribales de OK son urbanos de OKC).
  - tipo_area: areas legales vs estadisticas, mismo estado y banda.

Uso:  .venv/Scripts/python.exe procesos/bias_discovery/l3_naciones.py
Salida: datos/bias_discovery/l3_naciones.csv + tabla por stdout.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
SALIDA = RAIZ / "datos" / "bias_discovery" / "l3_naciones.csv"
SEMILLA = 42
REPS = 4000
MIN_TRACTOS_NACION = 15
MIN_COMP = 5
MIN_GRUPO = 10
METRICAS = ["cov_overture_usfs", "pct_km_unnamed", "share_sin_bomberos_ems",
            "escuelas_por_1k", "pois_por_1k"]

COLS_NAC = ["GEOID", "STATEFP", "COUNTYFP", "aiannh_name", "aiannh_kind", "tribal_pct",
            "pct_urban", "pop_total", "usfs_BuildingCount_sum", "carbonplan_buildings"]


def cargar():
    def geo(df):
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        return df

    edi = geo(pq.read_table(RAIZ / "datos/features_base/features_buildings.parquet",
                            columns=["GEOID", "region", "bldg_count", "pop_total"]
                            ).to_pandas())
    via = geo(pq.read_table(RAIZ / "datos/features_base/features_roads.parquet",
                            columns=["GEOID", "km_named", "km_total_unfiltered"]).to_pandas())
    poi = geo(pq.read_table(RAIZ / "datos/features_base/features_pois.parquet",
                            columns=["GEOID", "poi_fire", "poi_ems", "poi_schools",
                                     "poi_total"]).to_pandas())
    nac = geo(pq.read_table(RAIZ / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
                            columns=COLS_NAC).to_pandas())
    df = edi.merge(via, on="GEOID", how="left", validate="1:1") \
            .merge(poi, on="GEOID", how="left", validate="1:1") \
            .merge(nac.drop(columns=["pop_total"]), on="GEOID", how="left", validate="1:1")
    df["condado"] = df["STATEFP"] + df["COUNTYFP"]
    df["estado"] = df["STATEFP"]
    df["tribal_pct"] = df["tribal_pct"].fillna(0.0)
    df["pct_urban"] = df["pct_urban"].fillna(0.0).clip(0, 1)
    df["tribal"] = df["tribal_pct"] > 0
    df["nacion"] = df["aiannh_name"].where(df["tribal"])
    return df


def construir_metricas(df):
    usfs = df["usfs_BuildingCount_sum"].where(df["usfs_BuildingCount_sum"] > 0)
    df["cov_overture_usfs"] = (df["bldg_count"] / usfs).clip(0, 3)

    km = df["km_total_unfiltered"].where(df["km_total_unfiltered"] >= 1.0)
    df["pct_km_unnamed"] = (1.0 - df["km_named"] / km).clip(0, 1)

    df["share_sin_bomberos_ems"] = ((df["poi_fire"].fillna(0)
                                     + df["poi_ems"].fillna(0)) == 0).astype(float)

    pop_ok = df["pop_total"].where(df["pop_total"] >= 100)
    df["escuelas_por_1k"] = 1000.0 * df["poi_schools"] / pop_ok
    df["pois_por_1k"] = 1000.0 * df["poi_total"] / pop_ok

    df["bin"] = pd.cut(df["pct_urban"], [-0.01, 0.05, 0.30, 0.70, 1.0],
                       labels=["rural", "bajo", "medio", "urbano"]).astype(object)
    return df


def estratos_por(df, col_geo, mask_grupo, mask_pool):
    """Pares (idx_grupo, idx_comp) por unidad geografica x banda de ruralidad."""
    grupo = df[mask_grupo]
    estratos = []
    for geo, g in grupo.groupby(col_geo):
        comp = df[mask_pool & (df[col_geo] == geo)]
        for b in pd.unique(g["bin"]):
            i_g = g.index[g["bin"] == b]
            i_c = comp.index[comp["bin"] == b]
            if len(i_c) >= MIN_COMP and len(i_g) > 0:
                estratos.append((np.asarray(i_g), np.asarray(i_c)))
    return estratos


def dif_estratificada(vals, estratos, rng=None, reps=0):
    pesos = [len(a) for a, _ in estratos]
    total = sum(pesos)
    m_g = sum(w * vals[a].mean() for (a, _), w in zip(estratos, pesos)) / total
    m_c = sum(w * vals[b].mean() for (_, b), w in zip(estratos, pesos)) / total
    if not reps:
        return m_g, m_c, None
    boot = np.zeros(reps)
    for (a, b), w in zip(estratos, pesos):
        va, vb = vals[a], vals[b]
        ra = rng.integers(0, len(va), size=(reps, len(va)))
        rb = rng.integers(0, len(vb), size=(reps, len(vb)))
        boot += w * (va[ra].mean(axis=1) - vb[rb].mean(axis=1))
    return m_g, m_c, boot / total


def probar(df, analisis, mask_grupo, mask_pool, rng):
    estratos = estratos_por(df, "condado" if analisis == "nacion_condado" else "estado",
                            mask_grupo, mask_pool)
    if not estratos:
        return [], 0
    idx_g = np.concatenate([a for a, _ in estratos])
    if len(idx_g) < MIN_GRUPO:
        return [], int(len(idx_g))
    filas = []
    for met in METRICAS:
        vals = df[met].to_numpy(dtype=float)
        est = [(a[np.isfinite(vals[a])], b[np.isfinite(vals[b])]) for a, b in estratos]
        est = [(a, b) for a, b in est if len(a) and len(b)]
        if not est:
            continue
        m_g, m_c, boot = dif_estratificada(vals, est, rng, REPS)
        ok = idx_g[np.isfinite(vals[idx_g])]
        filas.append({
            "analisis": analisis,
            "n": int(len(ok)),
            "n_comp": int(sum(len(b) for _, b in est)),
            "n_estratos": len(est),
            "n_condados": int(df.loc[ok, "condado"].nunique()),
            "poblacion": int(df.loc[ok, "pop_total"].sum()),
            "metrica": met,
            "valor_grupo": float(m_g),
            "valor_comp": float(m_c),
            "dif": float(m_g - m_c),
            "ic95_lo": float(np.percentile(boot, 2.5)),
            "ic95_hi": float(np.percentile(boot, 97.5)),
            "p_boot": float(min(1.0, max(2 * min((boot <= 0).mean(), (boot >= 0).mean()),
                                    1.0 / REPS))),
            "_boot": boot,
        })
    return filas, int(len(idx_g))


def holm(filas):
    m = len(filas)
    prev, orden = 0.0, sorted(range(m), key=lambda i: filas[i]["p_boot"])
    for rango, i in enumerate(orden, start=1):
        alpha_k = 0.05 / (m - rango + 1)
        prev = min(1.0, max(prev, filas[i]["p_boot"] * (m - rango + 1)))
        boot = filas[i].pop("_boot")
        filas[i].update(alpha_holm=alpha_k, p_holm=prev, rank_holm=rango,
                        ic_holm_lo=float(np.percentile(boot, 100 * alpha_k / 2)),
                        ic_holm_hi=float(np.percentile(boot, 100 * (1 - alpha_k / 2))),
                        sobrevive_holm=bool(prev < 0.05))
    return filas


def main():
    rng = np.random.default_rng(SEMILLA)
    df = construir_metricas(cargar())
    tribal = df[df["tribal"]]
    print(f"tractos en el set: {len(df)} | tribal_pct>0: {len(tribal)}")

    coexist = tribal.groupby("condado")["aiannh_kind"].nunique()
    print(f"condados del set con areas tribales: {len(coexist)} | con legal y estadistica "
          f"a la vez: {int((coexist > 1).sum())} ({list(coexist[coexist > 1].index)})")
    ok_no_trib = df[(df["estado"] == "40") & ~df["tribal"]]
    print(f"OK no tribal: {len(ok_no_trib)} tractos en {ok_no_trib['condado'].nunique()} "
          f"condados | urbanos(pct_urban>0.7): {int((ok_no_trib['pct_urban'] > 0.7).sum())}")

    conteo = tribal.groupby("aiannh_name").size().sort_values(ascending=False)
    pruebas = []
    naciones = [n for n, c in conteo.items() if c >= MIN_TRACTOS_NACION]

    ok_tribal = df["tribal"] & df["estado"].eq("40")
    ok_no_tribal = ~df["tribal"] & df["estado"].eq("40")
    filas, usados = probar(df, "ref_no_tribal", ok_tribal, ok_no_tribal, rng)
    for f in filas:
        f.update(nacion="OK tribal (todas)", tipo_area="mixto")
    pruebas += filas
    print(f"ref_no_tribal (solo OK): n={usados}")

    for nac in naciones:
        mask = df["nacion"].eq(nac).to_numpy()
        otras = df["tribal"].to_numpy() & ~mask
        tipo = tribal.loc[tribal["aiannh_name"] == nac, "aiannh_kind"].mode().iloc[0]
        for analisis in ("nacion_estado", "nacion_condado"):
            filas, usados = probar(df, analisis, mask, otras, rng)
            for f in filas:
                f.update(nacion=nac, tipo_area=tipo)
            pruebas += filas
            if filas:
                print(f"  {nac} [{analisis}]: n={usados}/{conteo[nac]} "
                      f"condados={filas[0]['n_condados']}")

    legal = df["tribal"] & df["aiannh_kind"].eq("legal")
    estad = df["tribal"] & df["aiannh_kind"].eq("statistical")
    filas, usados = probar(df, "tipo_area", legal, estad, rng)
    for f in filas:
        f.update(nacion="areas legales (todas)", tipo_area="legal")
    pruebas += filas
    print(f"  legal vs estadistica [tipo_area, comp=estadisticas del mismo estado]: "
          f"n={usados}/{int(legal.sum())}")

    pruebas = holm(pruebas)
    cols = ["analisis", "nacion", "tipo_area", "metrica", "n", "n_comp", "n_estratos",
            "n_condados", "poblacion", "valor_grupo", "valor_comp", "dif", "ic95_lo",
            "ic95_hi", "ic_holm_lo", "ic_holm_hi", "alpha_holm", "p_boot", "p_holm",
            "rank_holm", "sobrevive_holm"]
    out = pd.DataFrame(pruebas)[cols].sort_values(["analisis", "nacion", "metrica"])
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SALIDA, index=False, float_format="%.6f")

    print(f"\n=== {len(out)} pruebas, Holm sobre la familia completa ===")
    for (ana, nac), g in out.groupby(["analisis", "nacion"], sort=False):
        print(f"\n--- {ana} | {nac} | n={g['n'].iloc[0]} pob={g['poblacion'].iloc[0]:,} "
              f"condados={g['n_condados'].iloc[0]} estratos={g['n_estratos'].iloc[0]}")
        for _, r in g.iterrows():
            marca = "SI" if r["sobrevive_holm"] else " -"
            print(f"  {r['metrica']:<22} n={r['n']:>4} comp={r['n_comp']:>4} "
                  f"{r['valor_grupo']:>8.3f} vs {r['valor_comp']:>8.3f} "
                  f"dif={r['dif']:+.3f} IC[{r['ic95_lo']:+.3f},{r['ic95_hi']:+.3f}] "
                  f"IC_Holm[{r['ic_holm_lo']:+.3f},{r['ic_holm_hi']:+.3f}] "
                  f"p={r['p_boot']:.4f} p_holm={r['p_holm']:.4f} {marca}")

    print("\n=== cobertura Overture/USFS por nacion: tramo bajo y tractos nombrados ===")
    for nac in [n for n, c in conteo.items() if c >= MIN_TRACTOS_NACION]:
        s = tribal[tribal["aiannh_name"] == nac].dropna(subset=["cov_overture_usfs"])
        peores = s.nsmallest(3, "cov_overture_usfs")["GEOID"].tolist()
        print(f"  {nac}: n={len(s)} q25={s['cov_overture_usfs'].quantile(.25):.3f} "
              f"p10={s['cov_overture_usfs'].quantile(.10):.3f} "
              f"n<0.75={int((s['cov_overture_usfs'] < 0.75).sum())} peores={peores}")
    print(f"\nescrito: {SALIDA}")


if __name__ == "__main__":
    main()
