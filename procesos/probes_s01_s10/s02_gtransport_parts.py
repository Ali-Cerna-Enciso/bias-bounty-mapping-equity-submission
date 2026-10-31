"""s02 — descomponer g_transport en piezas con prefijo g2_.

Entradas: features_roads.parquet y components_cand00.parquet.
Salida: datos/features_sonda/s02_gtransport_parts.parquet
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AG = ROOT / "datos"
OUT = AG / "features_sonda"
LAM_TERTIARY = 0.30  # mismo lambda que step_09_ensemble/script.py


def g_transport(r: pd.DataFrame) -> np.ndarray:
    den = (r.km_highway + LAM_TERTIARY * (r.km_tertiary_named + r.km_highway_ring25)).to_numpy()
    g = np.where(den > 0, np.clip(1 - r.km_highway.to_numpy() / np.where(den > 0, den, 1), 0, 1), 0.0)
    return np.where(r.km_highway.to_numpy() == 0, 1.0, g)


def ratio(num: pd.Series, den: pd.Series) -> np.ndarray:
    d = den.to_numpy()
    v = np.where(d > 0, num.to_numpy() / np.where(d > 0, d, 1), 0.0)
    return np.clip(v, 0.0, 1.0)


def build(r: pd.DataFrame) -> pd.DataFrame:
    hw = r.km_highway
    return pd.DataFrame({
        "GEOID": r.GEOID.astype(str),
        "g2_zero_highway": (hw == 0).astype(float),
        "g2_unnamed_highway_share": ratio(hw - r.km_highway_named, hw),
        "g2_ring25_share": ratio(r.km_highway_ring25, hw),
        "g2_tertiary_refroute_share": ratio(r.km_tertiary_refroute, r.km_tertiary),
    })


def main() -> None:
    r = pd.read_parquet(AG / "features_base" / "features_roads.parquet")
    r["GEOID"] = r.GEOID.astype(str)
    out = build(r)
    assert len(out) == 9794 and out.GEOID.is_unique and out.GEOID.str.len().eq(11).all()

    gt = g_transport(r)
    ref = pd.read_parquet(AG / "features_base" / "components_cand00.parquet",
                          columns=["GEOID", "g_transport"])
    ref["GEOID"] = ref.GEOID.astype(str)
    chk = out[["GEOID"]].assign(mine=gt).merge(ref, on="GEOID", how="inner")
    print(f"filas={len(chk)} max|g_transport diff|={np.abs(chk.mine - chk.g_transport).max():.2e}")

    feats = [c for c in out.columns if c.startswith("g2_")]
    rural = (r.set_index("GEOID").pct_urban.reindex(out.GEOID).to_numpy() < 0.5).astype(float)
    print(f"\n{'feature':<28}{'mean':>8}{'sd':>8}{'%0':>7}{'%1':>7}{'r_gtrans':>10}{'r_rural':>9}")
    for c in feats:
        v = out[c].to_numpy()
        print(f"{c:<28}{v.mean():8.3f}{v.std():8.3f}{(v == 0).mean() * 100:6.1f}%"
              f"{(v == 1).mean() * 100:6.1f}%{np.corrcoef(v, gt)[0, 1]:10.3f}"
              f"{np.corrcoef(v, rural)[0, 1]:9.3f}")
    print(f"{'g_transport':<28}{gt.mean():8.3f}{gt.std():8.3f}{'':14}"
          f"{np.corrcoef(gt, rural)[0, 1]:10.3f}")

    assert not out[feats].isna().any().any() and np.isfinite(out[feats].to_numpy()).all()
    assert all(out[c].nunique() > 1 for c in feats)

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "s02_gtransport_parts.parquet"
    out.to_parquet(dest, compression="zstd", index=False)
    print(f"\n-> {dest} ({len(out)} x {len(out.columns)})")
    print("missing %:", out[feats].isna().mean().round(4).to_dict())


if __name__ == "__main__":
    main()
