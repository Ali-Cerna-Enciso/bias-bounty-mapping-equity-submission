"""Inventario de sample submissions + baseline cero (4 y 5 regiones).

Uso: ../../.venv/Scripts/python.exe build_baseline.py
"""

import csv
import filecmp
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
CONCURSO = AQUI.parents[1] / "CONCURSO"
SAMPLE_DIR = CONCURSO / "sample_submissions"
REF_DIR = CONCURSO / "data" / "reference"

OFICIALES = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]
EXTRA = ["eastern-wa"]

ESPERADO = {
    "eastern-ok": 1192,
    "maricopa-az": 1593,
    "northern-ca": 591,
    "south-central-tx": 6003,
    "eastern-wa": 415,
}


def leer_sample(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: los GEOID conservan el cero inicial de Maricopa.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def inventariar():
    print("== 1. Inventario de sample submissions ==")
    frames = {}
    for region in OFICIALES + EXTRA:
        path = SAMPLE_DIR / f"{region}-sample-submission.csv"
        df = leer_sample(path)
        frames[region] = df

        dup = int(df["GEOID"].duplicated().sum())
        malos = df.loc[~df["GEOID"].str.fullmatch(r"\d{11}"), "GEOID"].tolist()
        prefijos = sorted(df["GEOID"].str[:2].unique())
        cero_inicial = int((df["GEOID"].str[0] == "0").sum())

        ref = REF_DIR / region / f"{region}-sample-submission.csv"
        igual = filecmp.cmp(path, ref, shallow=False) if ref.exists() else None

        print(f"\n{region}")
        print(f"  archivo        : {path.name}")
        print(f"  filas          : {len(df)}  (esperado {ESPERADO[region]})")
        print(f"  columnas       : {list(df.columns)}")
        print(f"  duplicados     : {dup}")
        print(f"  GEOID invalidos: {len(malos)}")
        print(f"  prefijos FIPS  : {prefijos}")
        print(f"  GEOID con 0 ini: {cero_inicial}")
        print(f"  identico a ref : {igual}")
        if dup or malos:
            print(f"  !! PROBLEMA dup={dup} malos={malos[:5]}")
    return frames


def buscar_oficial():
    print("\n== 2. Busqueda del SampleSubmission.csv oficial ==")
    # Solo el nombre oficial de Zindi; los "<region>-sample-submission.csv" no cuentan.
    candidatos = sorted(
        p
        for p in CONCURSO.parent.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and p.name.lower().replace("-", "").replace("_", "").replace(" ", "") == "samplesubmission.csv"
    )
    if not candidatos:
        print("  NO existe SampleSubmission.csv en CONCURSO/ ni en la raiz del repo.")
        print("  -> Ambiguedad 4 vs 5 regiones no resoluble localmente.")
    else:
        for p in candidatos:
            print(f"  encontrado: {p}")
    return candidatos


def escribir_baseline(frames, regiones, destino: Path):
    geoids = []
    for region in regiones:
        geoids.extend(frames[region]["GEOID"].tolist())

    df = pd.DataFrame({"GEOID": geoids, "coverage_gap_score": 0.0})
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, lineterminator="\n", float_format="%.1f")
    return df


def validar(destino: Path, esperado: int):
    with destino.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    df = pd.read_csv(destino, dtype=str, keep_default_na=False)
    scores = pd.to_numeric(df["coverage_gap_score"], errors="coerce")

    checks = {
        "header exacto": header == ["GEOID", "coverage_gap_score"],
        "filas": len(df) == esperado,
        "GEOID texto 11 digitos": bool(df["GEOID"].str.fullmatch(r"\d{11}").all()),
        "GEOID unicos": not df["GEOID"].duplicated().any(),
        "sin blancos": not (df == "").any().any(),
        "score parseable": not scores.isna().any(),
        "score en [0,1]": bool(((scores >= 0) & (scores <= 1)).all()),
        "score todos 0.0": bool((scores == 0.0).all()),
    }

    print(f"\n{destino.name}  ({len(df)} filas)")
    for nombre, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALLA'}] {nombre}")
    return all(checks.values())


def main():
    frames = inventariar()
    buscar_oficial()

    print("\n== 3. Solapamiento entre regiones ==")
    todos = []
    for region in OFICIALES + EXTRA:
        todos.extend(frames[region]["GEOID"].tolist())
    print(f"  suma de las 5 regiones : {len(todos)}")
    print(f"  GEOID distintos        : {len(set(todos))}")
    print(f"  [{'OK' if len(todos) == len(set(todos)) else 'FALLA'}] un tracto pertenece a una sola region")

    print("\n== 4. Baselines ==")
    b4 = escribir_baseline(frames, OFICIALES, AQUI.parents[1] / "datos/candidatos/baseline_cero_4reg.csv")
    b5 = escribir_baseline(frames, OFICIALES + EXTRA, AQUI.parents[1] / "datos/candidatos/baseline_cero_5reg.csv")
    print(f"  baseline_cero_4reg.csv : {len(b4)} filas")
    print(f"  baseline_cero_5reg.csv : {len(b5)} filas")

    delta = set(b5["GEOID"]) - set(b4["GEOID"])
    wa = set(frames["eastern-wa"]["GEOID"])
    print(f"  [{'OK' if delta == wa else 'FALLA'}] 5reg - 4reg == eastern-wa ({len(delta)} vs {len(wa)})")

    print("\n== 5. Validacion ==")
    ok4 = validar(AQUI.parents[1] / "datos/candidatos/baseline_cero_4reg.csv", 9379)
    ok5 = validar(AQUI.parents[1] / "datos/candidatos/baseline_cero_5reg.csv", 9794)
    print(f"\nValidacion global: {'OK' if ok4 and ok5 else 'FALLA'}")


if __name__ == "__main__":
    main()
