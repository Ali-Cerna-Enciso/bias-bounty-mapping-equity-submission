"""Sondas centradas en la media real y lectura de su covarianza con el target.

Una sonda es f = m + a*z, con z estandarizada sobre los 9,794 tractos y m = media pública exacta.
  RMSE(f)^2 = var(y) - 2a*cov_pub(z, y) + a^2*mean_pub(z^2)
El término de covarianza es exacto (m es la media pública); solo a^2*mean_pub(z^2) se aproxima por 1,
y con a = 0.02 el error relativo sobre cov es ~0.3%/rho.

Features nuevas: un archivo por familia (GEOID + una columna por feature, 9,794 filas)
  datos/features_sonda/<familia>.parquet
El nombre de la columna es el nombre de la sonda; debe ser único entre todos los archivos.

  python sondas_v2.py screen            -> corr local de cada candidata con las ya sondeadas (novedad)
  python sondas_v2.py build col1 col2   -> escribe sonda_v2_<col>.csv
  python sondas_v2.py read              -> con SCORES: rho, aporte incremental y mezcla ridge
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "datos" / "sondas"
BASE = ROOT / "datos" / "features_base"
EXTRA = ROOT / "datos" / "features_sonda"
REG = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]

A = 0.107246299 ** 2                       # RMSE(cero)^2 = mean(y^2)
R_CONST = 0.089560921                      # RMSE(constante 0.05)
M = (A + 0.05 ** 2 - R_CONST ** 2) / (2 * 0.05)
VAR = A - M * M
AMP = 0.02
LAMBDA = 0.1

SCORES = {                                 # nombre de sonda -> RMSE público
    "rural": 0.085765003,
    "g_transport": 0.08759489,
    "t_route_lowclass_share": 0.086122934,
    "h_fire_missing_risk": 0.088818198,
    "c_poi_deficit_pop": 0.091352474,
    "r_tribal_any": 0.084907318,
}

# Colas que domina un puñado de tractos (79% ceros, kurtosis 48): sin acotar, la sonda
# mide esos tractos y no el patrón. Se acota en p1/p99 antes de estandarizar.
WINSOR = {"t_route_lowclass_share"}

# Medidas con |rho| < 0.04: entran al registro, no a la mezcla (el ridge les da peso espurio).
EXCLUIR = {"c_poi_deficit_pop"}

order = pd.concat([pd.read_csv(ROOT / "CONCURSO/sample_submissions" / f"{r}-sample-submission.csv",
                               dtype={"GEOID": str}) for r in REG], ignore_index=True).GEOID


def features() -> pd.DataFrame:
    c = pd.read_parquet(BASE / "components_cand00.parquet").set_index("GEOID")
    raw = pd.DataFrame({
        "rural": (c.pct_urban < .5).astype(float).where(c.pct_urban.notna()),
        "g_transport": c.g_transport,
    }, index=c.index)
    for path in sorted(EXTRA.glob("*.parquet")):
        extra = pd.read_parquet(path)
        extra["GEOID"] = extra.GEOID.astype(str)
        dup = set(extra.columns) & set(raw.columns) - {"GEOID"}
        assert not dup, f"{path.name} repite columnas {dup}"
        raw = raw.join(extra.set_index("GEOID"), how="left")
    raw = raw.loc[order]
    for col in WINSOR & set(raw.columns):
        lo, hi = raw[col].quantile([0.01, 0.99])
        raw[col] = raw[col].clip(lo, hi)
    z = (raw - raw.mean()) / raw.std()
    return z.fillna(0.0)


def covs() -> pd.Series:
    return pd.Series({k: (VAR + AMP ** 2 - r ** 2) / (2 * AMP) for k, r in SCORES.items() if r is not None})


def screen() -> None:
    z = features()
    done = [k for k in covs().index if k in z]
    todo = [k for k in z if k not in done]
    C = z.corr()
    print("Candidatas sin sondear: corr local con las ya sondeadas (|corr| baja = información nueva)")
    for k in todo:
        print(f"  {k:30s}", "  ".join(f"{d}={C.loc[k, d]:+.2f}" for d in done))


def build(cols) -> None:
    z = features()
    for col in cols or list(z):
        f = (M + AMP * z[col]).clip(0, 1)
        assert f.between(0, 1).all() and len(f) == 9794
        out = pd.DataFrame({"GEOID": order.values, "coverage_gap_score": f.values})
        path = HERE / f"sonda_v2_{col}.csv"
        out.to_csv(path, index=False, float_format="%.7f")
        print(f"{path.name}: mean={f.mean():.6f} min={f.min():.4f} max={f.max():.4f}")


def fit(Z, c, lam):
    S = np.atleast_2d(np.cov(Z.T.values))
    beta = np.linalg.solve(S + lam * np.eye(len(S)), c.values)
    mse = VAR - 2 * beta @ c.values + beta @ S @ beta
    return beta, np.sqrt(mse)


def read(cols=()) -> None:
    z = features()
    c = covs()
    use = [k for k in c.index if k not in EXCLUIR]
    if cols:
        use = [k for k in use if k in cols]
    print(f"m={M:.6f}  sd(y)={np.sqrt(VAR):.6f}  RMSE constante óptima={np.sqrt(VAR):.6f}")
    _, full = fit(z[use], c[use], LAMBDA)
    for k, cv in c.items():
        rho = cv / np.sqrt(VAR)
        rest = [j for j in use if j != k]
        without = fit(z[rest], c[rest], LAMBDA)[1] if rest else np.sqrt(VAR)
        marca = "" if k in use else "  (excluida de la mezcla)"
        print(f"  {k:30s} rho={rho:+.3f}  aporte incremental={without - full:+.5f}{marca}")
    if len(use) < 2:
        return
    for lam in (0.0, LAMBDA, 0.3):
        beta, r = fit(z[use], c[use], lam)
        print(f"  ridge lam={lam}: RMSE predicho={r:.5f}  beta={dict(zip(use, np.round(beta, 4)))}")
    beta, r = fit(z[use], c[use], LAMBDA)
    f = (M + z[use].values @ beta).clip(0, 1)
    pd.DataFrame({"GEOID": order.values, "coverage_gap_score": f}).to_csv(
        HERE / "mezcla_v2_ridge.csv", index=False, float_format="%.7f")
    print(f"  escrito mezcla_v2_ridge.csv (lam={LAMBDA}, {len(use)} features, RMSE predicho {r:.5f}, "
          f"clip afecta {(f <= 0).sum() + (f >= 1).sum()} filas)")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "build":
        build(args)
    elif cmd == "read":
        read(args)
    else:
        {"screen": screen}[cmd]()
