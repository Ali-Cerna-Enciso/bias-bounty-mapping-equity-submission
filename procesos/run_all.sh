#!/usr/bin/env bash
# Corre el pipeline completo, en orden.
# No descarga nada ni sube nada: lee CONCURSO/data/ y escribe en datos/.
# Uso: bash procesos/run_all.sh            (desde cualquier carpeta)
#      bash procesos/run_all.sh --con-hallazgo   (+ L7 y L9 de Best Bias Discovery, ~2 min)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/Scripts/python.exe}"
cd "$ROOT"

paso() { echo "== $* =="; }

# datos/ no versiona carpetas vacias: los pasos 1 y 9 escriben ahi.
mkdir -p datos/candidatos datos/intermedios

paso "1/9 baseline cero"
"$PY" procesos/step_01_contract/build_baseline.py

paso "2/9 features de edificios (DuckDB, 5 regiones)"
"$PY" procesos/step_02_buildings/script.py

paso "3/9 features de vias (DuckDB, 5 regiones)"
"$PY" procesos/step_03_roads/script.py

paso "4/9 features de POIs (DuckDB, 5 regiones)"
"$PY" procesos/step_04_pois/script.py

paso "5/9 components_cand00 + cand_00"
"$PY" procesos/step_09_ensemble/script.py

paso "6/9 features de sonda s01-s10"
for s in procesos/probes_s01_s10/s*.py; do
    echo "-- $(basename "$s")"
    "$PY" "$s"
done

paso "7/9 sondas individuales"
"$PY" procesos/sondas/sondas_v2.py build \
    rural g_transport t_route_lowclass_share h_fire_missing_risk c_poi_deficit_pop r_tribal_any

paso "8/9 mezcla ridge"
"$PY" procesos/sondas/sondas_v2.py read

paso "9/9 terminos estructurales y mezcla final"
mkdir -p datos/intermedios
"$PY" procesos/sondas/s12_vias_red.py
"$PY" procesos/sondas/s13_estructural.py
"$PY" procesos/sondas/secuencial.py sondas

echo "listo: datos/sondas/mezcla_v4_estructural.csv"

if [[ "${1:-}" == "--con-hallazgo" ]]; then
    paso "opcional: hallazgo L7 cuidado + L9 hospitales"
    PYTHONIOENCODING=utf-8 "$PY" procesos/bias_discovery/l7_cuidado.py
    PYTHONIOENCODING=utf-8 "$PY" procesos/bias_discovery/l9_hospitales.py
fi
