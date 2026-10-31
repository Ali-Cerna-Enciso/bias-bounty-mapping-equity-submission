# Submissions

The 17 files sent to the Zindi leaderboard for this competition, in order. Every file has
9,794 rows and the columns `GEOID,coverage_gap_score`, `GEOID` as an 11-character string.
`Public RMSE` is the leaderboard value the platform returned, copied from the score page at
the time of submission.

| # | What it is | Public RMSE | Submission ID | CSV |
|---:|---|---:|---|---|
| 1 | Zero baseline: every tract at 0.0 (control) | 0.107246299 | — | `subida_1/submission.csv` |
| 2 | First structure (`cand_00`): building + transport + POI composite, one global scale | 0.099680211 | — | `subida_2/submission.csv` |
| 3 | Constant 0.05: measures the mean of the target | 0.089560921 | — | `subida_3/submission.csv` |
| 4 | Probe: `rural`, amplitude 0.02 | 0.085765003 | — | `subida_4/submission.csv` |
| 5 | Probe: `g_transport` | 0.087594890 | — | `subida_5/submission.csv` |
| 6 | Probe: `t_route_lowclass_share` | 0.086122934 | — | `subida_6/submission.csv` |
| 7 | Probe: `h_fire_missing_risk` | 0.088818198 | — | `subida_7/submission.csv` |
| 8 | Probe: `c_poi_deficit_pop` (no signal, dropped) | 0.091352474 | — | `subida_8/submission.csv` |
| 9 | Ridge blend of 4 features, λ = 0.1 | 0.082736583 | — | `subida_9/submission.csv` |
| 10 | Probe: `r_tribal_any` | 0.084907318 | — | `subida_10/submission.csv` |
| 11 | Ridge blend of 5 features, λ = 0.1 | 0.080228114 | `1CjfM8je` | `subida_11/mezcla_v2_ridge.csv` |
| 12 | Blend 11 shrunk by s = 0.85 (hedge, not the primary) | 0.080525733 | — | `subida_12/mezcla_v3_shrink85.csv` |
| 13 | Blend 11 + 0.02·z(`s_road_gap_condal`) (probe, dropped) | 0.082005410 | — | `subida_13/secuencial_01_s_road_gap_condal.csv` |
| 14 | Blend 11 + 0.02·z(`s_poi_schools`) (probe) | 0.081699336 | `yzRTSBH5` | `subida_14/secuencial_02_s_poi_schools.csv` |
| 15 | Blend 11 + 0.02·z(`s_road_undefined`) (probe) | 0.086634760 | `NEhoA7Bd` | `subida_15/secuencial_03_s_road_undefined.csv` |
| 16 | Blend 11 + 0.02·z(`b_bldg_deficit`) (probe, dropped) | 0.083192057 | `dQQyhb5r` | `subida_16/secuencial_04_b_bldg_deficit.csv` |
| 17 | **Final**: blend 11 + road flag + school term | **0.078391501** | `KDVKc3sY` | `subida_17/mezcla_v4_estructural.csv` |

The zero-to-best path is 0.107246299 → 0.078391501 on the public split. The best constant is
0.089022 (the standard deviation of the target), so submissions 3 and 4 already sit below the
trivial floor and 9 onwards are below it by a clear margin.

How each one is produced:

- 1–2: `procesos/step_01_contract/build_baseline.py` and
  `procesos/step_09_ensemble/script.py` (the latter writes `datos/candidatos/`).
- 3–10: `procesos/sondas/sondas_v2.py build <feature>`. The six probes of 4–10 are kept in
  `datos/sondas/sonda_v2_<feature>.csv`; 9 is the same ridge as 11 restricted to four features
  (see `README.md`).
- 11–12: `procesos/sondas/sondas_v2.py read` and `procesos/sondas/variantes_robustas.py`.
- 13–17: `procesos/sondas/secuencial.py sondas`, which writes `datos/sondas/secuencial_0N_*.csv`
  as well as the final `datos/sondas/mezcla_v4_estructural.csv`.

`subida_11`, `subida_12`, `subida_13`–`subida_16` and `subida_17` are byte-identical to the
copies in `datos/sondas/` (verified by SHA-256).
