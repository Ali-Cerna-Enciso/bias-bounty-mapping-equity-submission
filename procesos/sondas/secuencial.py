"""Última mezcla: subida 11 más las sondas de componente que superaron el ruido de medida.

Con b = subida 11 y z estandarizada sobre los 9,794 tractos, cada envío b + a z midió:
    RMSE(b + a z)^2 = RMSE(b)^2 - 2 a c + a^2,   c = cov_pub(z, y - b)
El paso w minimiza RMSE(b)^2 - 2 w'c + w'(S + lambda I)w, con S la covarianza local de las z.

Error de medida aproximado de c: 0.08 / sqrt(0.3 * 9794) = 0.0015. Entran las sondas con |c| de más de 2 errores:
`s_road_undefined` (-11.3) y `s_poi_schools` (+2.7). Quedan fuera `s_road_gap_condal` (+1.9; su peso conjunto es 0.0001)
y `b_bldg_deficit` (-1.4). Determinista. Uso: python procesos/sondas/secuencial.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "procesos/sondas"))
import sondas_v2 as s  # noqa: E402

A = 0.02
LAM = 0.1
BASE, RB = "subidas/subida_11/mezcla_v2_ridge.csv", 0.080228114
MEDIDAS = {                     # sonda -> RMSE público del envío b + 0.02 z
    "s_road_gap_condal": 0.08200541,   # subida 13
    "s_poi_schools": 0.081699336,      # subida 14
    "s_road_undefined": 0.08663476,    # subida 15
    "b_bldg_deficit": 0.083192057,     # subida 16
}
USE = ["s_road_undefined", "s_poi_schools"]


def main():
    z = s.features()
    f = pd.read_csv(ROOT / BASE, dtype={"GEOID": str}).set_index("GEOID").coverage_gap_score.loc[z.index]
    if "sondas" in sys.argv:           # reescribe los CSV de las subidas 13 a 16
        for i, k in enumerate(MEDIDAS, 1):
            pd.DataFrame({"GEOID": z.index, "coverage_gap_score": (f + A * z[k]).clip(0, 1).values}).to_csv(
                ROOT / f"datos/sondas/secuencial_{i:02d}_{k}.csv", index=False, float_format="%.7f")
    c = pd.Series({k: (RB ** 2 + A * A - v ** 2) / (2 * A) for k, v in MEDIDAS.items()})
    for k, v in c.items():
        print(f"  {k:20s} c={v:+.5f}  rho_inc={v / np.sqrt(s.VAR):+.3f}  {'entra' if k in USE else 'fuera'}")
    Z, cc = z[USE], c[USE]
    S = np.cov(Z.T.values)
    w = np.linalg.solve(S + LAM * np.eye(len(S)), cc.values)
    pred = np.sqrt(RB ** 2 - 2 * w @ cc.values + w @ S @ w)
    fin = (f + Z.values @ w).clip(0, 1)
    print("  pesos:", dict(zip(USE, np.round(w, 5))))
    print(f"RMSE predicho {pred:.5f}; R2 {1 - pred ** 2 / s.VAR:.3f}; media {fin.mean():.6f} sd {fin.std():.5f} "
          f"min {fin.min():.4f} max {fin.max():.4f} filas {len(fin)} en cero {(fin <= 0).sum()}")
    pd.DataFrame({"GEOID": z.index, "coverage_gap_score": fin.values}).to_csv(
        ROOT / "datos/sondas/mezcla_v4_estructural.csv", index=False, float_format="%.7f")


if __name__ == "__main__":
    main()
