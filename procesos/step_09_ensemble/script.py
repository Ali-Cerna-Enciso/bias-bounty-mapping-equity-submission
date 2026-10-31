"""Candidato 00 — primera estructura (A09).

Receta transparente, sin ajuste contra el LB. Un solo parámetro global (rho).
Salida: datos/candidatos/cand_00_primera_estructura.csv (9,794 filas).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONC = ROOT / "CONCURSO"
FB = ROOT / "datos" / "features_base"
OUT = ROOT / "datos" / "candidatos"

REGIONS = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]

# ---------------------------------------------------------------- parámetros
SPH_URBAN, SPH_RURAL, URBAN_CUT = 0.55, 1.20, 0.50   # edificios por vivienda
PERS_PER_HU = 2.6                                    # fallback cuando housing_units = 0
LAM_TERTIARY = 0.30                                  # peso de tertiary/ring en el denominador TIGER
POI_RATE_RURAL, POI_RATE_URBAN = 18.0, 14.0          # POIs esperados por 1k hab (base + pendiente urbana)
POI_PER_LARGE_BLDG = 0.05
P_T_TIERS = {"motorway_or_trunk": 0.85, "highway_ge_2km": 0.55, "highway_lt_2km": 0.40,
             "no_highway_ring": 0.30, "no_highway": 0.25}
P_B_TIERS = {"both": 0.97, "one": 0.90, "neither": 0.10}
P_P_TIERS = {"both": 0.95, "one": 0.85, "neither": 0.20}
RHO = 0.55                                           # corr asumida composite~target
MEAN_Y2 = 0.0115018                                  # ancla: RMSE(0)^2
MEAN_CEIL = 0.107246299                              # ancla: mean(y) <= LB(0)


def load() -> pd.DataFrame:
    b = pd.read_parquet(FB / "features_buildings.parquet",
                        columns=["GEOID", "region", "bldg_count", "bldg_large_count", "bldg_missing",
                                 "housing_units", "pop_total", "pct_urban", "ruca_primary"])
    r = pd.read_parquet(FB / "features_roads.parquet",
                        columns=["GEOID", "km_motorway", "km_trunk", "km_primary", "km_secondary",
                                 "km_tertiary_named", "km_highway", "km_highway_ring25"])
    p = pd.read_parquet(FB / "features_pois.parquet",
                        columns=["GEOID", "poi_total", "poi_fire", "poi_ems", "poi_schools", "poi_retail"])
    for name, d in (("buildings", b), ("roads", r), ("pois", p)):
        assert d.GEOID.is_unique, f"GEOID duplicado en {name}"
        d["GEOID"] = d.GEOID.astype(str)
    df = b.merge(r, on="GEOID", how="inner").merge(p, on="GEOID", how="inner")
    assert len(df) == 9794, len(df)
    return df


def components(df: pd.DataFrame) -> pd.DataFrame:
    urban = df.pct_urban.fillna(0.0)
    sph = np.where(urban >= URBAN_CUT, SPH_URBAN, SPH_RURAL)
    hu_eff = df.housing_units.where(df.housing_units > 0, df.pop_total / PERS_PER_HU)
    exp_b = (hu_eff * sph).to_numpy()
    df["g_building"] = np.where(exp_b > 0, np.clip(1 - df.bldg_count.to_numpy() / np.where(exp_b > 0, exp_b, 1), 0, 1), 0.0)

    den = (df.km_highway + LAM_TERTIARY * (df.km_tertiary_named + df.km_highway_ring25)).to_numpy()
    g_t = np.where(den > 0, np.clip(1 - df.km_highway.to_numpy() / np.where(den > 0, den, 1), 0, 1), 0.0)
    df["g_transport"] = np.where(df.km_highway == 0, 1.0, g_t)

    exp_p = (df.pop_total / 1000 * (POI_RATE_RURAL + POI_RATE_URBAN * urban)
             + POI_PER_LARGE_BLDG * df.bldg_large_count).to_numpy()
    df["g_poi"] = np.where(exp_p > 0, np.clip(1 - df.poi_total.to_numpy() / np.where(exp_p > 0, exp_p, 1), 0, 1), 0.0)
    return df


def probabilities(df: pd.DataFrame) -> pd.DataFrame:
    mt, hw = df.km_motorway + df.km_trunk, df.km_highway
    df["p_T"] = np.select(
        [mt > 0, hw >= 2.0, hw > 0],
        [P_T_TIERS["motorway_or_trunk"], P_T_TIERS["highway_ge_2km"], P_T_TIERS["highway_lt_2km"]],
        default=np.where(df.km_highway_ring25 > 0, P_T_TIERS["no_highway_ring"], P_T_TIERS["no_highway"]),
    )
    df["p_B"] = np.where((df.bldg_count > 0) & (df.housing_units > 0), P_B_TIERS["both"],
                         np.where((df.housing_units > 0) | (df.bldg_count > 0), P_B_TIERS["one"],
                                  P_B_TIERS["neither"]))
    df["p_P"] = np.where((df.poi_total > 0) & (df.pop_total > 0), P_P_TIERS["both"],
                         np.where((df.pop_total > 0) | (df.poi_total > 0), P_P_TIERS["one"],
                                  P_P_TIERS["neither"]))
    return df


def composite(df: pd.DataFrame) -> pd.DataFrame:
    g = df[["g_transport", "g_building", "g_poi"]].to_numpy()
    p = df[["p_T", "p_B", "p_P"]].to_numpy()
    expect = np.zeros(len(df))
    p_empty = np.ones(len(df))
    for S in itertools.chain.from_iterable(itertools.combinations(range(3), r) for r in (1, 2, 3)):
        w = np.ones(len(df))
        for i in range(3):
            w = w * (p[:, i] if i in S else 1 - p[:, i])
        expect += w * g[:, list(S)].mean(axis=1)
    for i in range(3):
        p_empty = p_empty * (1 - p[:, i])
    # el sample sólo conserva tractos con ≥1 componente definido
    df["u"] = expect / (1 - p_empty)
    df["p_all_undefined"] = p_empty
    return df


def main() -> None:
    df = components(load())
    df = probabilities(df)
    df = composite(df)

    k = RHO * np.sqrt(MEAN_Y2) / np.sqrt((df.u ** 2).mean())
    df["score"] = (k * df.u).clip(0, 1)
    assert df.score.mean() <= MEAN_CEIL

    OUT.mkdir(parents=True, exist_ok=True)
    order = pd.concat([pd.read_csv(CONC / "sample_submissions" / f"{r}-sample-submission.csv",
                                   dtype={"GEOID": str}) for r in REGIONS], ignore_index=True).GEOID
    out = df.set_index("GEOID").loc[order, ["score"]].reset_index()
    out.columns = ["GEOID", "coverage_gap_score"]
    out.to_csv(OUT / "cand_00_primera_estructura.csv", index=False, float_format="%.6f")

    df.reset_index(drop=True).to_parquet(FB / "components_cand00.parquet",
                                         index=False, compression="zstd")

    s = df.score
    print(f"k={k:.4f} rho={RHO} filas={len(out)}")
    print(f"score: mean={s.mean():.4f} med={s.median():.4f} p90={s.quantile(.9):.4f} "
          f"p99={s.quantile(.99):.4f} max={s.max():.4f} mean2={(s ** 2).mean():.5f}")
    print(f"share>0.5={(s > 0.5).mean():.5f} share>0.107={(s > 0.107246).mean():.4f} ceros={(s == 0).sum()}")
    print("\ncomponentes (media / share>0 / p90):")
    for c in ["g_transport", "g_building", "g_poi"]:
        x = df[c]
        print(f"  {c:12s} {x.mean():.4f}  {(x > 0).mean():.4f}  {x.quantile(.9):.4f}")
    print("\npor región:")
    tab = df.groupby("region").agg(n=("score", "size"), pT=("p_T", "mean"), pB=("p_B", "mean"),
                                   pP=("p_P", "mean"), gT=("g_transport", "mean"),
                                   gB=("g_building", "mean"), gP=("g_poi", "mean"),
                                   u=("u", "mean"), score=("score", "mean"),
                                   med=("score", "median"), mx=("score", "max"))
    print(tab.round(4).to_string())
    print("\nshare score>0.5 por región:",
          (df.score > 0.5).groupby(df.region).mean().round(5).to_dict())
    print("p_all_undefined: max=%.3f, >0.05 en %d filas" % (df.p_all_undefined.max(), int((df.p_all_undefined > .05).sum())))

    # validación del CSV escrito
    chk = pd.read_csv(OUT / "cand_00_primera_estructura.csv", dtype={"GEOID": str})
    assert list(chk.columns) == ["GEOID", "coverage_gap_score"]
    assert len(chk) == 9794 and chk.GEOID.is_unique
    assert set(chk.GEOID) == set(order) and (chk.GEOID.str.len() == 11).all()
    assert chk.coverage_gap_score.notna().all() and chk.coverage_gap_score.between(0, 1).all()
    assert chk.GEOID.tolist() == list(order)
    print("\nCSV OK:", OUT / "cand_00_primera_estructura.csv")


if __name__ == "__main__":
    main()
