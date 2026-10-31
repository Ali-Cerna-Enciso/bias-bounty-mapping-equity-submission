"""s13 — estimador estructural de los componentes del gap, solo con el paquete del concurso.

No lee ninguna capa retirada. Cada término es una esperanza calculada con datos de Overture y estratos:

  transport: T_hat = km de Overture con ruta numerada (US:I, US:US, US:<ST>, US:<ST>:FM|RM|...), en cualquier clase.
             O     = km motorway..secondary. Definido si T_hat > UMBRAL_KM. gap = 1 - min(1, O / T_hat).
             Calibración contra lo publicado: % de tractos sin T_hat por región frente a % con componente indefinido.
  poi:       para cada tipo (fire, ems, schools) se ajusta O ~ Poisson(lambda(x)). Con cobertura c, lo que falta es
             M ~ Poisson(lambda*(1-c)/c), independiente de O. E[gap | O] = E[M/(O+M); O+M>0].
  building:  se trata como 0 cuando el tracto tiene edificios (peso de divisor sí, gap no).

Resultado: datos/features_sonda/s13_estructural.parquet (GEOID + una columna por candidata; el nombre es la sonda).
Determinista: PoissonRegressor es convexo, sin semilla. Uso: python procesos/sondas/s13_estructural.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

ROOT = Path(__file__).resolve().parents[2]
UMBRAL_KM = 0.25
PESO_CONDAL = 0.6
COBERTURA = {"fire": 0.60, "ems": 0.85, "schools": 0.80}   # supuesto; el LB calibra la amplitud del término
RED = ["interstate", "us", "estatal", "estatal_sec"]


def vias():
    l = pd.read_parquet(ROOT / "datos/intermedios/s12_vias_red_largo.parquet")
    t = l.pivot_table(index="GEOID", columns="grupo_red", values="km", aggfunc="sum", fill_value=0.0)
    alta = l[l.grupo_clase == "alta"].groupby("GEOID").km.sum().rename("O")
    return t.join(alta)


def e_gap(O, lam_m):
    """E[M/(O+M)] con M ~ Poisson(lam_m), truncando en 30."""
    m = np.arange(0, 31)
    pm = poisson.pmf(m[None, :], lam_m[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(O[:, None] + m[None, :] > 0, m[None, :] / (O[:, None] + m[None, :]), np.nan)
    defin = np.where(O[:, None] + m[None, :] > 0, 1.0, 0.0)
    return np.nansum(pm * g, axis=1), np.sum(pm * defin, axis=1)


def main():
    b = pd.read_parquet(ROOT / "datos/features_base/features_buildings.parquet").set_index("GEOID")
    p = pd.read_parquet(ROOT / "datos/features_base/features_pois.parquet").set_index("GEOID")
    v = vias().reindex(b.index).fillna(0.0)
    d = pd.DataFrame(index=b.index)
    d["region"] = b.region
    d["rural"] = (b.pct_urban.fillna(0) < .5) * 1.0
    d["lpop"] = np.log1p(b.pop_total.fillna(0))
    d["larea"] = np.log1p(b.ALAND.fillna(0) / 1e6)
    d["lbld"] = np.log1p(b.bldg_count.fillna(0))
    X = pd.concat([d[["rural", "lpop", "larea", "lbld"]], pd.get_dummies(d.region, dtype=float)], axis=1).values

    # --- transporte
    T = v[RED].sum(axis=1)
    O = v["O"]
    defT = (T > UMBRAL_KM).values
    gap_t = np.where(defT, np.clip(1 - O / T.replace(0, np.nan), 0, 1), 0.0)
    # Red condal: 25,253 km en eastern-ok. Con peso 1 la razón O/T de esa región da 0.57 y el mínimo publicado es 0.71;
    # 0.71 se alcanza con peso 0.6, si eastern-ok es la región del mínimo (supuesto).
    Tc = T + PESO_CONDAL * v["condal"]
    gap_tc = np.where((Tc > UMBRAL_KM).values, np.clip(1 - O / Tc.replace(0, np.nan), 0, 1), 0.0)

    # --- POI por tipo
    out, pesos = {}, {}
    for tipo, col in (("fire", "poi_fire"), ("ems", "poi_ems"), ("schools", "poi_schools")):
        o = p[col].reindex(b.index).fillna(0).values.astype(float)
        lam = PoissonRegressor(alpha=1e-3, max_iter=2000).fit(X, o).predict(X)
        c = COBERTURA[tipo]
        g, pdef = e_gap(o, lam * (1 - c) / c)
        out[tipo], pesos[tipo] = g, pdef
    num = sum(out.values())
    den = sum(pesos.values())
    hifld = np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)
    poi = 0.5 * hifld                              # la mitad CBP no mostró señal medible (subida 8): se deja en 0

    # --- compuesto: media sobre definidos; building definido y en 0 cuando hay edificios
    hay_b = (b.bldg_count.fillna(0) > 0).values * 1.0
    hay_p = np.ones(len(b))
    for nombre, g, dt in (("s_struct", gap_t, defT), ("s_struct_condal", gap_tc, (Tc > UMBRAL_KM).values)):
        d[nombre] = (g * dt + poi) / np.maximum(dt * 1.0 + hay_b + hay_p, 1.0)
    d["s_road_gap"] = gap_t
    d["s_road_gap_condal"] = gap_tc
    d["s_road_div"] = gap_t * defT / np.maximum(defT * 1.0 + hay_b + hay_p, 1.0)
    d["s_road_undefined"] = (~defT) * 1.0
    d["s_poi_hifld"] = hifld
    d["s_poi_fire"] = out["fire"]
    d["s_poi_schools"] = out["schools"]
    d["s_poi_ems"] = out["ems"]
    cols = [c for c in d.columns if c.startswith("s_")]
    d[cols].reset_index().astype({"GEOID": str}).to_parquet(
        ROOT / "datos/features_sonda/s13_estructural.parquet", compression="zstd", index=False)
    print(d.groupby("region")[cols].mean().round(3).T)
    print(d.groupby("rural")[cols].mean().round(3).T)
    print(d[cols].describe().T[["mean", "std", "50%", "max"]].round(3))


if __name__ == "__main__":
    main()
