"""Variantes de la mezcla de 5 para el 31/10. No reescribe mezcla_v2_ridge.csv.

Chequeo: la mezcla de 4 tiene que predecir ~0.08265 (subió 0.082736583).
Encogimiento s=0.85. s=0.5 tira el 25% de la ganancia y no se escribe.
"""
import numpy as np
import pandas as pd

import sondas_v2 as s

USE = ["rural", "g_transport", "t_route_lowclass_share", "h_fire_missing_risk", "r_tribal_any"]
SIN_FIRE = [k for k in USE if k != "h_fire_missing_risk"]


def escribir(nombre, valores):
    f = np.clip(np.asarray(valores, float), 0, 1)
    pd.DataFrame({"GEOID": s.order.values, "coverage_gap_score": f}).to_csv(
        s.HERE / nombre, index=False, float_format="%.7f")
    print(f"{nombre}: mean={f.mean():.6f} filas={len(f)}")


def main():
    z, c = s.features(), s.covs()
    _, r4 = s.fit(z[USE[:4]], c[USE[:4]], 0.1)
    beta, r5 = s.fit(z[USE], c[USE], 0.1)
    if abs(r4 - 0.08265) > 0.0003 or abs(r5 - 0.08002) > 0.0003:
        raise SystemExit(f"cordura falló: mezcla4={r4:.5f} mezcla5={r5:.5f}. No escribo nada.")
    print(f"cordura ok: mezcla4={r4:.5f} mezcla5={r5:.5f}")

    b3, r3 = s.fit(z[USE], c[USE], 0.3)
    escribir("mezcla_v3_lam03.csv", s.M + z[USE].values @ b3)
    print(f"  predicho lam=0.3: {r3:.5f}")

    shrink = 0.85
    escribir("mezcla_v3_shrink85.csv", s.M + shrink * (z[USE].values @ beta))
    g = s.VAR - r5 ** 2
    print(f"  predicho shrink 0.85: {np.sqrt(s.VAR - (2 * shrink - shrink ** 2) * g):.5f}")

    b, rr = s.fit(z[SIN_FIRE], c[SIN_FIRE], 0.1)
    escribir("mezcla_v3_sin_fire.csv", s.M + z[SIN_FIRE].values @ b)
    print(f"  predicho sin h_fire: {rr:.5f}")


if __name__ == "__main__":
    main()
