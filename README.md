# Bias Bounty Mapping Equity Challenge — code and writeups

Code, intermediate data and the two written entries for the Zindi *Bias Bounty Mapping Equity
Challenge* (Humane Intelligence, Radiant Earth). The task was to predict a
per-tract `coverage_gap_score` in `[0, 1]` for 9,794 census tracts in five US regions, scored
by RMSE against the organiser's hidden reference scores.

This repository contains everything needed to reproduce the submitted CSV from the competition
data package, plus the nine diagnostics written for the Best Bias Discovery prize.

## Result

| | Public RMSE |
|---|---:|
| Zero submission | 0.107246299 |
| Best constant (the standard deviation of the target) | 0.089022 |
| **Final submission** (`KDVKc3sY`) | **0.078391501** |

The submission was a ridge blend of five tract-level signals plus two structural terms derived
from the published formula. Coefficient weights were calibrated from public leaderboard
responses. `subidas/README.md` lists all 17
submissions with their scores and files.

## The two writeups

The two prize writeups are submitted through the challenge form, so the compiled PDFs are not
published here. Their full LaTeX sources, tables and figures are:

- `docs/2_premios/writeup/documentation.tex` — Best Documentation: method, the two leaderboard
  identities, edge cases, alternative weightings, reproducibility.
- `docs/2_premios/writeup/bias_discovery.tex` — Best Bias Discovery: the Overture `hospital`
  category against CMS/VA/IHS/state records, and why a deficit-only score cannot see a layer that
  over-counts.

Each compiles in place with `pdflatex` run twice.

## Data

The scored CSV reads the competition package: Overture buildings, roads, roads-unfiltered, rail,
POIs and infrastructure (release 2026-08-19.0), ACS housing, the strata tables and the 2020 census
tracts. `datos/externos/` holds dated snapshots of CMS, VA, IHS, HRSA, Texas HHSC, Arizona AEMS,
Oklahoma OSDH, Census and OpenStreetMap responses for the bias discovery work, with the URLs in
`bias_discovery.tex`.

## Repository layout

```
procesos/                       all pipeline scripts (only l6 and l8 fetch two ACS tables)
  step_01_contract/           sample-submission inventory and the zero baseline
  step_02_buildings/          Overture buildings + ACS -> features_buildings.parquet
  step_03_roads/               Overture roads, 25 m buffer, rail -> features_roads.parquet
  step_04_pois/               Overture POIs by category -> features_pois.parquet
  step_09_ensemble/           cand_00: components, mask, first CSV
  probes_s01_s10/           the ten probe feature families
  sondas/                       the method: sondas_v2.py (build/read/screen), s12, s13,
                                secuencial.py, variantes_robustas.py, lb_identidad.py
  bias_discovery/               l1..l9 diagnostics, tables and figures for the writeups
  run_all.sh                    the whole chain, in order
datos/
  features_base/*.parquet       9,794 rows each: buildings (33 cols), roads (29), POIs (41),
                                components_cand00 (30)
  features_sonda/*.parquet      one parquet per probe family
  intermedios/                  s12_vias_red_largo.parquet (km by tract, road class and route)
  sondas/*.csv                  every probe and blend that was submitted
  bias_discovery/*.csv          the nine diagnostics' outputs
  externos/                     dated public snapshots used only by the discovery scripts
subidas/                        one folder per submission, the CSV and its score
docs/2_premios/writeup/         the two PDFs, their .tex, tablas/ and figuras/
CONCURSO/sample_submissions/    the five sample submission CSVs (the only competition files
                                small enough to version; they define the canonical row order)
```

## Requirements

Python 3.12 with the versions in `requirements.txt` (`pandas`, `numpy`, `pyarrow`, `duckdb`,
`geopandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`; `geopandas` pulls `pyproj`,
`pyogrio` and `shapely`). DuckDB pulls its `spatial` extension on first use; that is the only
network call the scored pipeline makes.

```bash
pip install -r requirements.txt
```

All scripts resolve the repository root from `__file__`, so they run from any directory.

## Downloading the competition data

The 5.8 GB package is public and needs no account. It is not in this repository
(`CONCURSO/data/` is in `.gitignore`). Fetch it into `CONCURSO/data/` keeping the bucket paths:

```bash
# whole package (~5.8 GB)
aws s3 sync --no-sign-request \
  s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/ \
  CONCURSO/data/

# or one file, over https
curl -O https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/strata/national/national-strata-tract-table.parquet
```

The scripts read exactly these files, for each `R` in `eastern-ok`, `eastern-wa`,
`maricopa-az`, `northern-ca`, `south-central-tx`:

```
CONCURSO/data/reference/R/R-overture-buildings.parquet
CONCURSO/data/reference/R/R-overture-roads.parquet
CONCURSO/data/reference/R/R-overture-roads-unfiltered.parquet
CONCURSO/data/reference/R/R-overture-rail.parquet
CONCURSO/data/reference/R/R-overture-infrastructure.parquet
CONCURSO/data/reference/R/R-overture-pois.parquet
CONCURSO/data/reference/R/R-census-acs-housing.parquet
CONCURSO/data/strata/R/R-census-tracts.parquet
CONCURSO/data/strata/R/R-strata-tract-table.parquet
CONCURSO/data/strata/national/national-strata-tract-table.parquet
```

Two ACS bulk tables (46 and 48 MB) are not versioned either: `l6_moviles.py` and `l8_nuevas.py`
download them from `www2.census.gov` into `datos/externos/census_acs/` if the file is missing.
Nothing else in `datos/externos/` is fetched at run time.

## Running the pipeline

```bash
bash procesos/run_all.sh                  # steps 1-9
bash procesos/run_all.sh --con-hallazgo   # plus the two heaviest discovery scripts (L7, L9)
```

`run_all.sh` uses `.venv/Scripts/python.exe` by default; set `PY` to use another interpreter,
e.g. `PY=python3 bash procesos/run_all.sh`.

Measured wall-clock times, from a clean clone with the package supplied separately (a laptop,
other work running at the same time):

| Stage | Script | Time |
|---|---|---:|
| 1 | `step_01_contract/build_baseline.py` | seconds |
| 2 | `step_02_buildings/script.py` | 835 s |
| 3 | `step_03_roads/script.py` | 2,070 s |
| 4 | `step_04_pois/script.py` | 72 s |
| 5 | `step_09_ensemble/script.py` | 2 s |
| 6 | `probes_s01_s10/s01..s10*.py` | 376 s (s01 alone 314 s) |
| 7 | `sondas/sondas_v2.py build ...` | 3 s |
| 8 | `sondas/sondas_v2.py read` | 2 s |
| 9 | `sondas/s12_vias_red.py`, `s13_estructural.py`, `secuencial.py sondas` | ~4 min |
| | **total** | **~60 min** |

Steps 2–4 are the expensive ones: DuckDB over the 5.8 GB package, with `threads=2` and
`memory_limit='4GB'` fixed inside the scripts. Everything after the base parquets is seconds.

The nine diagnostics of the Best Bias Discovery work are separate and cost about four minutes
in total (the times recorded when they were written: L1 7 s, L2 2 s, L3 2 s, L4 63 s, L5 34 s,
L6 14 s, L7 82 s, L8 5 s, L9 25 s; on a machine under load they can take two to three times
longer):

```bash
export PYTHONIOENCODING=utf-8
python procesos/bias_discovery/l1_incendios.py
python procesos/bias_discovery/l2_fuentes.py
python procesos/bias_discovery/l3_naciones.py
python procesos/bias_discovery/l4_colonias.py
python procesos/bias_discovery/l5_evacuacion.py
python procesos/bias_discovery/l6_moviles.py
python procesos/bias_discovery/l7_cuidado.py
python procesos/bias_discovery/l8_nuevas.py
python procesos/bias_discovery/l9_hospitales.py
python procesos/bias_discovery/l9_sensibilidad_ref.py
python procesos/bias_discovery/tablas_writeup.py    # tables for documentation.tex
```

`l9_referencia.py` and `l9_salidas.py` are modules, imported by `l9_hospitales.py` and not run
on their own: the first builds and caches the external reference (it uses `l9_geo_respaldo.py`,
which re-queries Nominatim only when the geocoding cache is missing), the second formats the
tables and the figure.

## Shortcuts that skip the recomputation

- **No competition data needed.** `secuencial.py` reads `datos/` and `subidas/subida_11/`, both in the repository.
  `python procesos/sondas/secuencial.py sondas` regenerates the final CSV
  `datos/sondas/mezcla_v4_estructural.csv` in seconds; it has to match
  `subidas/subida_17/mezcla_v4_estructural.csv` byte for byte.
- **No recomputation at all.** The CSVs in `datos/sondas/` and `subidas/` are the artefacts
  themselves; the two PDFs and their tables are checked in as well. Nothing here needs to run
  to be read.
- **Discovery only.** `l7_cuidado.py` and `l9_hospitales.py` need the package's POIs, census
  tracts and national strata table, but not buildings or roads; L7 and L9 also run on the
  snapshots in `datos/externos/`.

## Which command writes which CSV

| Output | Command |
|---|---|
| `datos/candidatos/baseline_cero_{4,5}reg.csv` | `step_01_contract/build_baseline.py` |
| `datos/features_base/features_buildings.parquet` | `step_02_buildings/script.py` (`[region]` or `--merge`) |
| `datos/features_base/features_roads.parquet` | `step_03_roads/script.py` |
| `datos/features_base/features_pois.parquet` | `step_04_pois/script.py` |
| `datos/features_base/components_cand00.parquet`, `datos/candidatos/cand_00_primera_estructura.csv` | `step_09_ensemble/script.py` |
| `datos/features_sonda/s01_routes.parquet` … `s09_interactions.parquet` | `probes_s01_s10/sNN_*.py` |
| `datos/sondas/sonda_v2_<feature>.csv` | `sondas/sondas_v2.py build <feature>` |
| `datos/sondas/mezcla_v2_ridge.csv` | `sondas/sondas_v2.py read` |
| `datos/sondas/mezcla_v3_{lam03,shrink85,sin_fire}.csv` | `sondas/variantes_robustas.py` |
| `datos/intermedios/s12_vias_red_largo.parquet` | `sondas/s12_vias_red.py` |
| `datos/features_sonda/s13_estructural.parquet` | `sondas/s13_estructural.py` |
| `datos/sondas/secuencial_0N_*.csv`, `mezcla_v4_estructural.csv` | `sondas/secuencial.py sondas` |
| `datos/bias_discovery/l1..l9_*.csv`, `docs/2_premios/writeup/{tablas,figuras}` | the `bias_discovery/l*.py` scripts |
| `docs/2_premios/writeup/tablas/{edge_cases,features,alt_weights}.tex` | `bias_discovery/tablas_writeup.py` |

`secuencial.py` and `variantes_robustas.py` import `sondas_v2`; run them through `run_all.sh`
or add `procesos/sondas` to `PYTHONPATH`.

One warning: `sondas_v2.py read` **overwrites** `datos/sondas/mezcla_v2_ridge.csv`. It is the
last step of its block for that reason. The four-feature blend of submission 9 is the same
command restricted to four features —
`sondas_v2.py read rural g_transport t_route_lowclass_share h_fire_missing_risk` — which also
writes to `mezcla_v2_ridge.csv`; copy the result aside before restoring the five-feature file.

## Reproduction checks

The chain was re-run from a clean clone on 30 October 2026 (steps 1–8) and the later half again
on 31 October 2026 (steps 7–9 and the discovery scripts) from a second copy of the repository,
with the results compared against the artefacts checked in here:

- `mezcla_v2_ridge.csv`, the six probe CSVs, the four sequential CSVs and
  `mezcla_v4_estructural.csv` reproduce with an identical SHA-256.
- `features_roads.parquet` and `components_cand00.parquet` are bit-identical;
  `features_buildings.parquet` differs by at most 6.3e-09 on `bldg_area_total_m2`, and
  `features_pois.parquet` by 3.0e-15 on `confidence_mean` (floating-point summation order).
- `l2_fuentes.py`, `l3_naciones.py`, `l4_colonias.py`, `l7_cuidado.py`, `l9_hospitales.py`,
  `l9_sensibilidad_ref.py` and `tablas_writeup.py` reproduce their CSVs and `.tex` tables
  exactly. The Matplotlib figures can differ in PDF metadata across library versions, so the
  figures are published as generated.
- Recompiling `documentation.tex` and `bias_discovery.tex` from this folder gives the same page
  count, the same page text and the same file size as the checked-in PDFs; only the internal
  PDF ID and timestamp change.

## Seeds

The scored pipeline is deterministic: no sampling, no stochastic optimiser and no train/test
split, so there is no seed to fix. The Poisson regression of `s13_estructural.py` is convex and
starts from zero. The discovery scripts use `seed=42`: `random_state=42` for the three 100-POI
validation samples in `l9_hospitales.py` and `seed=42` for the 2,000-draw county bootstrap, the
same seed in `l6_moviles.py` and `l8_nuevas.py`. The 300 labels of the validation samples are
stored in `datos/bias_discovery/l9_etiquetas_validacion.csv`, so the sampling is not re-run.

## Licences and attribution

- **Overture Maps** buildings, places, transportation, infrastructure and divisions: Overture
  Maps Foundation, released under ODbL v1.0 and CDLA-Permissive-2.0 depending on the layer
  (https://overturemaps.org/download/). Derived columns and CSVs here keep those terms.
- **US federal data** in the package (Census ACS and TIGER/Line geometries used by the
  organiser, CDC/ATSDR SVI, CDC/EPHT heat, USDA RUCA and RUCC, NCHS, USFS Wildfire Risk to
  Communities, US Drought Monitor, NASA, CarbonPlan): public domain or openly licensed, as
  distributed in the package's `strata/` tables. Attribution and vintages are in the package
  README.
- **CMS, VA, IHS, HRSA** provider records: US government works, public domain.
- **Texas HHSC, Arizona AEMS, Oklahoma OSDH** rosters: state public records without an explicit
  licence; dated copies are kept in `datos/externos/fsed/` and cited in `bias_discovery.tex`.
- **OpenStreetMap Nominatim** (236 geocoded addresses): ODbL, © OpenStreetMap contributors.

Code and text in this repository are published for the review of this competition entry; no
licence is granted beyond that, and reuse of the data must follow the terms above.
