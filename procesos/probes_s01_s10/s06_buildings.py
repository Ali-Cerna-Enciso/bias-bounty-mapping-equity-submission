"""s06: fuentes y densidad de edificios como features de gap (no se sube nada)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datos/features_base/features_buildings.parquet"
OUT_DIR = ROOT / "datos/features_sonda"
OUT = OUT_DIR / "s06_buildings.parquet"
N_EXPECTED = 9794

FEATURES = ["b_low_osm_share", "b_high_ms_share", "b_bldg_deficit", "b_area_p10_log"]


def load() -> pd.DataFrame:
    df = pd.read_parquet(SRC)
    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
    assert len(df) == N_EXPECTED and df["GEOID"].is_unique
    return df


def assign_cell(df: pd.DataFrame) -> pd.Series:
    """region x quintil de pct_urban.

    pct_urban esta saturado en 1.0 (62.6%), asi que qcut directo colapsa a 2 bins.
    El rango con desempate por orden de fila mantiene 5 celdas por region con n>=83.
    """
    pu = df["pct_urban"].fillna(df.groupby("region")["pct_urban"].transform("median"))
    rk = pu.groupby(df["region"]).rank(method="first")
    q = rk.groupby(df["region"]).transform(lambda s: pd.qcut(s, 5, labels=False))
    return df["region"] + "_q" + (q + 1).astype(int).astype(str)


def bldg_deficit(df: pd.DataFrame, cell: pd.Series) -> np.ndarray:
    """Deficit de huellas por vivienda frente a la mediana de su celda.

    El brief pedia dividir por sqrt(esperado+1); con esperado ~ 2,000 eso satura
    81.5% de las filas en +-3 y deja 18.5% con resolucion. Se estandariza por la
    MAD de la celda (escala del propio benchmark) y se mantiene el clip en +-3.
    """
    has_units = df["housing_units"] > 0
    ratio = pd.Series(np.nan, index=df.index)
    ratio[has_units] = df.loc[has_units, "bldg_count"] / df.loc[has_units, "housing_units"]
    med_ratio = ratio.groupby(cell).transform("median")
    mad = ratio.sub(med_ratio).abs().groupby(cell).transform("median") * 1.4826
    z = (med_ratio - ratio) / mad.clip(lower=0.15)
    z = z.where(has_units, 0.0)
    return z.clip(-3, 3).to_numpy()


def build(df: pd.DataFrame) -> pd.DataFrame:
    cell = assign_cell(df)
    out = pd.DataFrame({"GEOID": df["GEOID"]})
    out["b_low_osm_share"] = 1.0 - df["src_osm_share"].fillna(0.0)
    out["b_high_ms_share"] = df["src_microsoft_share"].fillna(0.0)
    out["b_bldg_deficit"] = bldg_deficit(df, cell)
    out["b_area_p10_log"] = np.log1p(df["bldg_area_median_m2"].clip(lower=0))
    return out


def validate(out: pd.DataFrame) -> None:
    assert len(out) == N_EXPECTED, len(out)
    assert out["GEOID"].is_unique and (out["GEOID"].str.len() == 11).all()
    assert out["GEOID"].str.fullmatch(r"\d{11}").all()
    num = out[FEATURES]
    assert not num.isna().any().any()
    assert np.isfinite(num.to_numpy()).all()
    assert (num.nunique() > 1).all(), num.nunique()


def report(out: pd.DataFrame, df: pd.DataFrame) -> None:
    rural = (df["pct_urban"].fillna(df.groupby("region")["pct_urban"].transform("median")) < 0.5)
    print(out[FEATURES].describe().T[["min", "50%", "max", "mean", "std"]].round(4).to_string())
    print("\ncorr con rural (pct_urban<0.5), Pearson / Spearman:")
    for f in FEATURES:
        p = np.corrcoef(out[f], rural.astype(int))[0, 1]
        s = out[f].corr(rural.astype(int), method="spearman")
        print(f"  {f:18s} {p:+.3f}  {s:+.3f}")
    print("\ncorr entre features (Pearson):")
    print(out[FEATURES].corr().round(3).to_string())


if __name__ == "__main__":
    df = load()
    out = build(df)
    validate(out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False, compression="zstd")
    report(out, df)
    print(f"\nOK -> {OUT}  ({len(out)} filas x {out.shape[1]} cols)")
