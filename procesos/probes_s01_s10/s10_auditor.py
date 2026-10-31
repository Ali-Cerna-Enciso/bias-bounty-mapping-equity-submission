"""s10: valida los CSV de envio/sonda contra el sample submission oficial.

Chequeo dirigido: 9,794 filas, GEOID string de 11 caracteres, unicos,
conjunto = union de los 5 sample submissions, score float en [0,1] sin NaN/inf,
y orden de filas igual al canonico (ok -> az -> ca -> tx -> wa).

Uso: ./.venv/Scripts/python.exe procesos/probes_s01_s10/s10_auditor.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "CONCURSO" / "sample_submissions"
# Orden canonico: concatenacion de los 5 sample submissions en orden alfabetico de
# region (ok -> wa -> az -> ca -> tx). Es el que usan las sondas y `cand_00`, ambos
# puntuados por Zindi. Zindi cruza por GEOID; el orden es control de consistencia
# interna para no desalinear las mezclas.
REGION_ORDER = ["eastern-ok", "eastern-wa", "maricopa-az", "northern-ca", "south-central-tx"]

TARGETS = sorted((ROOT / "subidas").glob("subida_*/submission.csv")) + sorted(
    p for p in (ROOT / "datos" / "sondas").glob("*.csv")
)


def expected():
    frames = []
    for region in REGION_ORDER:
        df = pd.read_csv(SAMPLE_DIR / f"{region}-sample-submission.csv", dtype={"GEOID": str})
        frames.append(df[["GEOID"]])
    return pd.concat(frames, ignore_index=True)["GEOID"]


def check(path, canon):
    row = {"archivo": path.parent.name + "/" + path.name if path.name == "submission.csv" else path.name, "fails": []}
    raw = pd.read_csv(path, dtype={"GEOID": str}, keep_default_na=False)
    row["cols"] = ",".join(raw.columns)
    fail = row["fails"]

    if list(raw.columns[:2]) != ["GEOID", "coverage_gap_score"]:
        fail.append(f"columnas={list(raw.columns)}")
    if len(raw) != len(canon):
        fail.append(f"filas={len(raw)} (esperado {len(canon)})")

    geoid = raw["GEOID"].astype(str)
    if not geoid.str.fullmatch(r"\d{11}").all():
        bad = geoid[~geoid.str.fullmatch(r"\d{11}")].head(3).tolist()
        fail.append(f"GEOID no 11 digitos: {bad}")
    if geoid.duplicated().any():
        fail.append(f"GEOID duplicados={int(geoid.duplicated().sum())}")

    set_sub, set_canon = set(geoid), set(canon)
    if set_sub != set_canon:
        fail.append(f"faltan={len(set_canon - set_sub)} sobran={len(set_sub - set_canon)}")
    if list(geoid) != list(canon):
        if set_sub == set_canon and len(geoid) == len(canon):
            fail.append("orden distinto al canonico")

    score = pd.to_numeric(raw["coverage_gap_score"], errors="coerce")
    if score.isna().any():
        fail.append(f"score no numerico={int(score.isna().sum())}")
    else:
        if not np.isfinite(score.to_numpy()).all():
            fail.append("score con inf")
        if score.min() < 0 or score.max() > 1:
            fail.append(f"score fuera de [0,1]: [{score.min():.6f},{score.max():.6f}]")
        row["media"] = float(score.mean())

    row["n"] = len(raw)
    row["pass"] = not fail
    return row


def main():
    canon = expected()
    canon11 = bool(canon.str.fullmatch(r"\d{11}").all())
    print(f"sample canonico: {len(canon)} filas, {canon.nunique()} unicos, 11 chars={canon11}")
    results = [check(p, canon) for p in TARGETS]

    print("\n| archivo | filas | PASS | detalle |")
    print("|---|---:|---|---|")
    for r in results:
        detail = "; ".join(r["fails"]) if r["fails"] else "ok"
        print(f"| {r['archivo']} | {r['n']} | {'PASS' if r['pass'] else 'FAIL'} | {detail} |")

    print("\nmedias:")
    for r in results:
        if "media" in r:
            print(f"  {r['archivo']:<45} {r['media']:.6f}")
    n_pass = sum(r["pass"] for r in results)
    print(f"\nTOTAL: {n_pass} PASS / {len(results) - n_pass} FAIL de {len(results)}")


if __name__ == "__main__":
    main()
