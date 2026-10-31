"""s03: riesgo de estacion de bomberos/EMS ausente en el tracto.

Un poi_fire==0 / poi_ems==0 en un tracto con poblacion dispersa puede ser
gap real (la estacion mas cercana cae fuera del tracto) y no ausencia real.
Se estima la tasa de estaciones por 1,000 hab. en cada celda region x quintil
de ruralidad y con ella la probabilidad esperada de al menos una estacion.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
POIS = ROOT / "datos/features_base/features_pois.parquet"
BLDG = ROOT / "datos/features_base/features_buildings.parquet"
OUT_DIR = ROOT / "datos/features_sonda"
OUT_PARQUET = OUT_DIR / "s03_fire_ems.parquet"

MIN_CELL_POP = 50_000
MIN_CELL_TRACTS = 20


def rural_quintile(df: pd.DataFrame) -> pd.Series:
    """Quintil de ruralidad dentro de cada region.

    pct_urban empata en 1.0 en ~55% de los tractos, asi que el desempate es por
    densidad de poblacion (mas disperso = quintil mas bajo = mas rural).
    """
    dens = df["pop_total"] / df["ALAND"].replace(0, np.nan)
    key = pd.DataFrame(
        {"region": df["region"], "u": df["pct_urban"].fillna(-1.0), "dens": dens}
    ).sort_values(["region", "u", "dens"], na_position="first")
    n = key.groupby("region")["u"].transform("size")
    q = (key.groupby("region").cumcount() * 5 // n).astype(int)
    return q.reindex(df.index)


def cell_rates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    cell = (
        df.groupby(["region", "q"])
        .agg(n=("GEOID", "size"), pop=("pop_total", "sum"), stations=(col, "sum"))
        .reset_index()
    )
    region = (
        df.groupby("region")
        .agg(pop_r=("pop_total", "sum"), st_r=(col, "sum"))
        .reset_index()
    )
    pop = cell["pop"].where(cell["pop"] > 0)
    cell["rate"] = cell["stations"] * 1000.0 / pop
    cell = cell.merge(region, on="region", how="left")
    region_rate = region["st_r"] * 1000.0 / region["pop_r"].where(region["pop_r"] > 0)
    cell["rate_region"] = cell["region"].map(dict(zip(region["region"], region_rate)))
    thin = (cell["pop"] < MIN_CELL_POP) | (cell["n"] < MIN_CELL_TRACTS)
    cell["rate"] = cell["rate"].where(~thin, cell["rate_region"])
    cell["fallback"] = thin
    return cell[["region", "q", "n", "pop", "rate", "fallback"]]


def missing_risk(df: pd.DataFrame, col: str) -> pd.Series:
    rates = cell_rates(df, col)
    rate = (
        df[["region", "q"]]
        .merge(rates[["region", "q", "rate"]], on=["region", "q"], how="left")["rate"]
        .fillna(0.0)
        .to_numpy()
    )
    exposure = rate * df["pop_total"].clip(lower=0).to_numpy() / 1000.0
    p_expected = 1.0 - np.exp(-exposure)
    absent = (df[col] == 0).astype(float).to_numpy()
    return pd.Series(absent * p_expected, index=df.index)


def main() -> None:
    pois = pd.read_parquet(POIS)
    bldg = pd.read_parquet(BLDG, columns=["GEOID", "rucc_2023", "bldg_count"])

    df = pois.merge(bldg, on="GEOID", how="inner")
    assert df["GEOID"].is_unique and len(df) == 9794, (len(df), df["GEOID"].nunique())

    df["q"] = rural_quintile(df)
    df["poi_emergency_any"] = (
        df["poi_fire"] + df["poi_ems"] + df["poi_emergency"]
    ).astype(float)

    out = pd.DataFrame(
        {
            "GEOID": df["GEOID"].astype(str),
            "h_fire_missing_risk": missing_risk(df, "poi_fire"),
            "h_ems_missing_risk": missing_risk(df, "poi_ems"),
            "h_emergency_any_missing_risk": missing_risk(df, "poi_emergency_any"),
        }
    )

    assert out["GEOID"].str.len().eq(11).all()
    assert not out["GEOID"].duplicated().any()
    assert np.isfinite(out.iloc[:, 1:].to_numpy()).all()
    for c in out.columns[1:]:
        assert out[c].nunique() > 1, c
        assert out[c].between(0, 1).all(), c

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, compression="zstd", index=False)

    info = {
        "filas": len(out),
        "fire0": int((df["poi_fire"] == 0).sum()),
        "ems0": int((df["poi_ems"] == 0).sum()),
        "any0": int((df["poi_emergency_any"] == 0).sum()),
        "celdas_fallback": int(
            cell_rates(df, "poi_fire")["fallback"].sum()
            + cell_rates(df, "poi_ems")["fallback"].sum()
        ),
    }
    print(info)
    print(out.describe().T.to_string())


if __name__ == "__main__":
    main()
