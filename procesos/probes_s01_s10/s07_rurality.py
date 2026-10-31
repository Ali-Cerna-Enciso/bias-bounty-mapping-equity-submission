"""s07: features de ruralidad continua/geografica para el set oficial de 9,794 tractos.

Entradas:
  - CONCURSO/data/strata/national/national-strata-tract-table.parquet
  - CONCURSO/sample_submissions/*.csv  (GEOID del set oficial)
  - datos/features_base/components_cand00.parquet  (solo correlaciones)
Salidas:
  - datos/features_sonda/s07_rurality.parquet
  - datos/features_sonda/s07_rurality.md
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STRATA = ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet"
SAMPLES = sorted((ROOT / "CONCURSO/sample_submissions").glob("*.csv"))
COMPS = ROOT / "datos/features_base/components_cand00.parquet"
OUT = ROOT / "datos/features_sonda"
COLS = ["GEOID", "ruca_primary", "rucc_2023", "ALAND", "pop_total",
        "pct_urban", "tribal_any"]
FEATS = ["r_ruca", "r_rucc", "r_log_area_per_person", "r_tribal_any"]


def load():
    ids = pd.concat([pd.read_csv(f, dtype=str, usecols=["GEOID"]) for f in SAMPLES])
    ids = ids.GEOID.drop_duplicates()
    st = pd.read_parquet(STRATA, columns=COLS)
    st = st[st.GEOID.isin(set(ids))]
    assert len(st) == 9794 == st.GEOID.nunique(), f"filas {len(st)}"
    assert (st.GEOID.str.len() == 11).all()
    return st.set_index("GEOID").loc[ids].reset_index()


def build(st):
    f = pd.DataFrame({"GEOID": st.GEOID.astype(str)})
    # RUCA 99 = "Not coded" (2 tractos): se imputa con la mediana del set.
    ruca = st.ruca_primary.replace(99.0, np.nan)
    f["r_ruca"] = ruca.fillna(ruca.median())
    f["r_rucc"] = st.rucc_2023.astype(float)
    f["r_log_area_per_person"] = np.log1p(st.ALAND / st.pop_total.clip(lower=1))
    f["r_tribal_any"] = st.tribal_any.fillna(False).astype(int).astype(float)
    return f, ruca.isna().sum()


def check(f):
    assert f.GEOID.is_unique and len(f) == 9794
    assert (f.GEOID.str.len() == 11).all()
    assert not f[FEATS].isna().any().any()
    assert np.isfinite(f[FEATS].to_numpy()).all()
    for c in FEATS:
        assert f[c].nunique() > 1, f"{c} constante"
        assert f[c].std() > 0


def corrs(f):
    comp = pd.read_parquet(COMPS, columns=["GEOID", "pct_urban", "g_transport"])
    assert set(comp.GEOID) == set(f.GEOID)
    rural = (comp.pct_urban < 0.5).astype(float).where(comp.pct_urban.notna())
    rows = []
    for c in FEATS:
        rows.append({
            "feature": c,
            "pearson_rural": f[c].corr(rural),
            "pearson_g_transport": f[c].corr(comp.g_transport),
            "min": f[c].min(), "median": f[c].median(), "max": f[c].max(),
            "pct_missing": 0.0,
        })
    return pd.DataFrame(rows)


def main():
    st = load()
    f, n_ruca_imputed = build(st)
    check(f)
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT / "s07_rurality.parquet", index=False, compression="zstd")
    c = corrs(f)
    print(c.round(4).to_string(index=False))
    print(f"RUCA 99 imputados: {n_ruca_imputed}")

    lines = []
    lines.append("# s07 — ruralidad continua (9,794 tractos, entrada: strata national)")
    lines.append("| feature | Pearson vs rural(pct_urban<0.5) | Pearson vs g_transport | min / mediana / max | missing |")
    lines.append("|---|---:|---:|---|---:|")
    for r in c.itertuples():
        lines.append(f"| `{r.feature}` | {r.pearson_rural:+.3f} | "
                     f"{r.pearson_g_transport:+.3f} | "
                     f"{r.min:g} / {round(r.median, 3):g} / {r.max:g} | {r.pct_missing:.1%} |")
    lines.append(f"`r_ruca` imputa 2 tractos RUCA 99 (\"Not coded\") con la mediana "
                 f"({f.r_ruca.median():g}); `r_log_area_per_person` usa "
                 f"`pop_total.clip(lower=1)`. Sin duplicados, sin inf, sin constantes.")
    (OUT / "s07_rurality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
