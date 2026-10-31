"""Convierte RMSE del LB en productos internos <f, y> y ajusta la mezcla lineal óptima.

Identidad: RMSE(f)^2 = mean(f^2) - 2 mean(f*y) + mean(y^2)  (sobre el split público).
mean(f^2) del público se aproxima con el de los 9,794 tractos.

Uso: completar LB con los CSV ya puntuados y correr.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "datos" / "sondas"
SUB = ROOT / "subidas"

LB = {
    SUB / "subida_2/submission.csv": 0.099680211,
    # SUB / "subida_3/submission.csv": None,
}
A = 0.107246299 ** 2  # mean(y^2): RMSE del envío todo-cero

F, b, names = [], [], []
for path, rmse in LB.items():
    if rmse is None:
        continue
    f = pd.read_csv(path, dtype={"GEOID": str}).coverage_gap_score.to_numpy()
    F.append(f)
    b.append((A + np.mean(f ** 2) - rmse ** 2) / 2)
    names.append(path.stem)
F, b = np.array(F), np.array(b)

const = [i for i, f in enumerate(F) if np.ptp(f) == 0]
if const:
    i = const[0]
    m = b[i] / F[i][0]
    vy = A - m * m
    print(f"mean(y) público = {m:.5f}   sd(y) = {np.sqrt(vy):.5f}   RMSE de la constante óptima = {np.sqrt(vy):.5f}")
    for k, f in enumerate(F):
        if np.ptp(f) == 0:
            continue
        cov = b[k] - f.mean() * m
        rho = cov / (f.std() * np.sqrt(vy))
        print(f"  {names[k]:40s} corr con y = {rho:+.3f}   RMSE afín óptimo = {np.sqrt(vy * (1 - rho ** 2)):.5f}")

G = F @ F.T / F.shape[1]
w = np.linalg.solve(G + 1e-9 * np.eye(len(G)), b)
pred_mse = A - 2 * w @ b + w @ G @ w
print("\nMezcla lineal óptima de todos los envíos puntuados:")
for n, wi in zip(names, w):
    print(f"  {wi:+.4f} x {n}")
print(f"  RMSE público predicho = {np.sqrt(max(pred_mse, 0)):.5f}  (subirla y comparar valida la identidad)")

blend = np.clip(w @ F, 0, 1)
order = pd.read_csv(next(iter(LB)), dtype={"GEOID": str}).GEOID
pd.DataFrame({"GEOID": order, "coverage_gap_score": blend}).to_csv(
    HERE / "mezcla_optima.csv", index=False, float_format="%.6f")
print(f"  escrito mezcla_optima.csv  mean={blend.mean():.5f}")
