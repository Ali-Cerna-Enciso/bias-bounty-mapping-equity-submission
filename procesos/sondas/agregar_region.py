"""Agrega el indicador de northern-ca. No toca ningún CSV de sonda.

Lo levanta sondas_v2.features() al leer datos/features_sonda/*.parquet.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]
order = pd.concat(
    [pd.read_csv(ROOT / "CONCURSO/sample_submissions" / f"{r}-sample-submission.csv",
                 dtype={"GEOID": str}) for r in REG],
    ignore_index=True,
).GEOID

comp = pd.read_parquet(ROOT / "datos/features_base/components_cand00.parquet")
comp["GEOID"] = comp.GEOID.astype(str).str.zfill(11)
ind = comp.set_index("GEOID").loc[order]
out = pd.DataFrame({
    "GEOID": order.values,
    "reg_northern_ca": (ind.region == "northern-ca").astype(float).values,
})
assert len(out) == 9794 and out.GEOID.str.len().eq(11).all() and out.reg_northern_ca.sum() == 591
path = ROOT / "datos/features_sonda/s11_region.parquet"
out.to_parquet(path, index=False)
print(f"{path.name}: {len(out)} filas, northern-ca={int(out.reg_northern_ca.sum())}")
