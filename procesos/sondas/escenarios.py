from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = 0.107246299**2          # mean(y^2) público
M0 = 0.099680211**2         # MSE cand_00
c = pd.read_parquet(ROOT / "datos/features_base/components_cand00.parquet")
f = c.score.values
Ef, Ef2 = f.mean(), (f**2).mean()
Efy = (A + Ef2 - M0) / 2
print(f"mean(f)={Ef:.5f} mean(f^2)={Ef2:.5f}  => mean(f*y)={Efy:.5f}")
print("cand_00 media por región:", c.groupby('region').score.mean().round(4).to_dict())
print(f"\n{'m=mean(y)':>10} {'RMSE const m':>13} {'corr cand00':>12} {'RMSE afín óptimo cand00':>24}")
for m in [0.02,0.03,0.04,0.05,0.06,0.07,0.08]:
    vy = A - m*m
    cov = Efy - Ef*m
    rho = cov/np.sqrt(vy*(Ef2-Ef*Ef))
    print(f"{m:10.2f} {np.sqrt(vy):13.5f} {rho:12.3f} {np.sqrt(vy*(1-rho**2)):24.5f}")
print("\nUmbral: la constante óptima le gana a cand_00 si mean(y) >", round(np.sqrt(A-M0),4))
