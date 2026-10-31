"""s04: deficit esperado de escuelas por tracto.

gap = (no hay escuela) x P(al menos una esperada por poblacion y ruralidad).
Tasa observada por 1,000 hab en celdas region x quintil de pct_urban.
Entrada: features_pois.parquet (Overture POIs + ACS de CONCURSO/data).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datos/features_base/features_pois.parquet"
OUT_DIR = ROOT / "datos/features_sonda"
OUT = OUT_DIR / "s04_schools.parquet"

SCHOOLS = ["poi_schools"]
ANY_EDU = ["poi_schools", "poi_education", "poi_college_university"]
LEVELS = ["poi_elementary", "poi_middle", "poi_high", "poi_public_school",
          "poi_private_school", "poi_school"]


def missing_risk(count, pop, region, urban_q):
    """P(al menos una escuela esperada) donde se observo cero; 0 donde hay alguna."""
    df = pd.DataFrame({"c": np.asarray(count), "p": np.asarray(pop, float),
                       "r": np.asarray(region), "q": np.asarray(urban_q)})
    cell = df.groupby(["r", "q"])[["c", "p"]].sum()
    reg = df.groupby("r")[["c", "p"]].sum()
    glob_rate = df["c"].sum() / df["p"].sum() * 1000
    tab = (cell["c"] / cell["p"].where(cell["p"] > 0) * 1000).rename("rate")
    reg_rate = (reg["c"] / reg["p"].where(reg["p"] > 0) * 1000).rename("rate_r")
    tab = tab.reset_index().merge(reg_rate.reset_index(), on="r", how="left")
    tab["rate"] = tab["rate"].fillna(tab["rate_r"]).fillna(glob_rate)
    rate = df.merge(tab[["r", "q", "rate"]], on=["r", "q"], how="left")["rate"].to_numpy()
    lam = rate * df["p"].to_numpy() / 1000.0
    risk = np.where(df["c"].to_numpy() == 0, 1.0 - np.exp(-lam), 0.0)
    return np.clip(risk, 0.0, 1.0)


def main():
    p = pd.read_parquet(SRC)
    p["GEOID"] = p["GEOID"].astype(str)
    assert p["GEOID"].is_unique and (p["GEOID"].str.len() == 11).all()

    urban = p["pct_urban"].astype(float)
    urban = urban.fillna(urban.groupby(p["region"]).transform("median"))
    urban_q = urban.groupby(p["region"]).transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    urban_q = urban_q.astype(int)

    pop = p["pop_total"].astype(float)
    out = pd.DataFrame({"GEOID": p["GEOID"]})

    c1 = p[SCHOOLS].sum(axis=1)
    out["h_school_missing_risk"] = missing_risk(c1, pop, p["region"], urban_q)

    c2 = p[ANY_EDU].sum(axis=1)
    out["h_school_any_missing_risk"] = missing_risk(c2, pop, p["region"], urban_q)

    c3 = p[LEVELS].sum(axis=1)
    level_same_mask = bool(((c3 == 0) == (c1 == 0)).all())
    if not level_same_mask:
        out["h_school_level_missing_risk"] = missing_risk(c3, pop, p["region"], urban_q)

    num = [c for c in out.columns if c != "GEOID"]
    for c in num:
        s = out[c]
        assert s.notna().all() and np.isfinite(s).all(), c
        assert s.nunique() > 1, f"{c} constante"
        assert s.between(0, 1).all(), c
        out[c] = s.round(6)

    out.to_parquet(OUT, compression="zstd", index=False)

    # diagnostico
    rate_glob = c1.sum() / pop.sum() * 1000
    print(f"GEOID n={len(out)} dup={out.GEOID.duplicated().sum()}")
    print(f"tasa global escuelas/1000hab = {rate_glob:.3f}")
    print(f"tractos con 0 poi_schools      = {(c1==0).sum()} ({(c1==0).mean()*100:.1f}%)")
    print(f"tractos con 0 school+edu+college= {(c2==0).sum()}")
    print(f"pasan de 0 a >0 al sumar vecinas= {((c1==0)&(c2>0)).sum()}")
    print(f"0 en levels vs 0 en poi_schools = {((c3==0)!=(c1==0)).sum()} discrepancias "
          f"(mismo cero={level_same_mask})")
    print(f"tasa global levels/1000hab      = {c3.sum()/pop.sum()*1000:.3f}")
    print("pop==0:", int((pop == 0).sum()), "| pct_urban rellenado:", 27)
    print(out[num].describe().loc[["mean", "min", "50%", "max"]].round(4).to_string())
    print("corr entre features:\n", out[num].corr().round(3).to_string())
    zero_rates = (out[num] == 0).mean()
    print("fraccion en 0:", zero_rates.round(3).to_dict())
    print(f"escrito {OUT}")


if __name__ == "__main__":
    main()
