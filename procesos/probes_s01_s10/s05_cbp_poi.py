"""s05: deficit relativo de POIs (observado vs esperado por region x ruralidad).

Entrada: features_pois.parquet + features_buildings.parquet (GEOID).
Salida:  datos/features_sonda/s05_cbp_poi.parquet
Entrada: Overture POIs/edificios ya agregados en CONCURSO/data/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
POIS = RAIZ / "datos/features_base/features_pois.parquet"
BLDG = RAIZ / "datos/features_base/features_buildings.parquet"
SALIDA = RAIZ / "datos/features_sonda/s05_cbp_poi.parquet"

UMBRAL_CELDA = 20  # tractos minimos para fiarse de la tasa de la celda


def cargar():
    p = pd.read_parquet(POIS)
    b = pd.read_parquet(BLDG)[["GEOID", "bldg_count", "bldg_large_count"]]
    df = p.merge(b, on="GEOID", how="left", validate="one_to_one")
    assert len(df) == 9794 and df.GEOID.str.len().eq(11).all() and df.GEOID.is_unique
    assert df.bldg_count.notna().all()
    return df


def bin_urbano(s):
    # los quintiles de pct_urban colapsan: 6,129 de 9,767 tractos valen exactamente 1.0.
    return pd.cut(
        s, bins=[-0.001, 0.001, 0.25, 0.6, 0.99, 1.0],
        labels=["0", "0-25", "25-60", "60-99", "100"],
        include_lowest=True,
    )


def tasa_esperada(df, num, den, escala):
    """POIs por (escala) habitantes/edificios: mediana de la celda region x bin urbano.

    Mediana y no total: el conteo de POIs por tracto es muy asimetrico y el total
    lo dominan unos pocos tractos con aglomeracion comercial.
    """
    ratio = df[num] / (df[den] / escala).replace(0, np.nan)
    df = df.assign(_r=ratio)
    df["_celda"] = df["region"].astype(str) + "|" + df["urb_bin"].astype(str)
    med_celda = df.groupby("_celda")["_r"].transform("median")
    n_celda = df.groupby("_celda")[den].transform("size")
    med_region = df.groupby("region")["_r"].transform("median")
    tasa = med_celda.where(n_celda >= UMBRAL_CELDA).fillna(med_region).fillna(ratio.median())
    return tasa, n_celda


def main():
    df = cargar()
    df["urb_bin"] = bin_urbano(df["pct_urban"])

    tasa_pop, n_celda = tasa_esperada(df, "poi_total", "pop_total", 1000.0)
    tasa_bldg, _ = tasa_esperada(df, "poi_total", "bldg_count", 100.0)
    esp_pop = tasa_pop * df["pop_total"] / 1000.0
    esp_bldg = tasa_bldg * df["bldg_count"] / 100.0

    # Deficit relativo en escala log: >0 = menos POIs de los esperados para la celda.
    def deficit(esp, obs):
        return np.clip(np.log((esp + 1.0) / (obs + 1.0)), -3.0, 3.0)

    out = pd.DataFrame(
        {
            "GEOID": df["GEOID"].astype(str),
            "c_poi_deficit_pop": deficit(esp_pop, df["poi_total"]),
            "c_poi_deficit_bldg": deficit(esp_bldg, df["poi_total"]),
            # pct_with_confidence == 100.0 en las 9,792 filas con dato (constante):
            # la cuota de POIs sin confianza es 0 en todo el set. Se usa 1 - confidence_mean.
            "c_low_confidence_mean": np.where(
                df["poi_total"] > 0, 1.0 - df["confidence_mean"], 0.0
            ),
        }
    )
    num = ["c_poi_deficit_pop", "c_poi_deficit_bldg", "c_low_confidence_mean"]
    out[num] = out[num].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")

    assert len(out) == 9794 and out.GEOID.is_unique and out.GEOID.str.len().eq(11).all()
    assert np.isfinite(out[num].to_numpy()).all()
    for c in num:
        assert out[c].nunique() > 1, f"{c} constante"

    out.to_parquet(SALIDA, index=False, compression="zstd")

    u = df["pct_urban"].fillna(df["pct_urban"].median())
    print("filas", len(out), "| regiones", out.GEOID.map(df.set_index("GEOID").region).nunique())
    print("celdas<umbral:", int((n_celda < UMBRAL_CELDA).sum()), "tractos")
    print("missing insumo:", {"pct_urban": int(df.pct_urban.isna().sum()),
                              "pct_with_confidence": int(df.pct_with_confidence.isna().sum()),
                              "pop<=0": int((df.pop_total <= 0).sum()),
                              "bldg==0": int((df.bldg_count == 0).sum()),
                              "conf_mean_nan": int(df.confidence_mean.isna().sum())})
    for c in num:
        print(f"{c}: miss={out[c].isna().sum()} corr_urb={out[c].corr(u):+.3f} "
              f"min={out[c].min():+.3f} med={out[c].median():+.3f} max={out[c].max():+.3f}")
    print("corr entre features:\n", out[num].corr().round(3).to_string())
    print("corr con poi_total/pop crudo:", round(df["poi_total"].div(df["pop_total"].replace(0, np.nan)).corr(u), 3))


if __name__ == "__main__":
    main()
