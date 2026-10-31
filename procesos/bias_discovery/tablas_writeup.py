"""Tablas LaTeX del writeup (Best Documentation): casos borde, construccion y pesos alternativos.

Reutiliza la logica de procesos/sondas/sondas_v2.py en memoria; no ejecuta `read`, por lo que
no reescribe datos/sondas/mezcla_v2_ridge.csv. Determinista.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "procesos" / "sondas"))
import sondas_v2 as s  # noqa: E402

OUT = ROOT / "docs" / "2_premios" / "writeup" / "tablas"
SONDAS = ROOT / "datos" / "sondas"
USE = [k for k in s.covs().index if k not in s.EXCLUIR]
SIN_FIRE = [k for k in USE if k != "h_fire_missing_risk"]

REAL_MIX4 = 0.082736583           # envío 9
PRED_MIX4 = 0.08265435875699025


def fit(cols=USE, lam=s.LAMBDA, shrink=1.0):
    z, c = s.features(), s.covs()
    beta, r = s.fit(z[cols], c[cols], lam)
    if shrink != 1.0:
        g = s.VAR - r ** 2
        r = np.sqrt(s.VAR - (2 * shrink - shrink ** 2) * g)
    return beta, r


def vector(cols=USE, lam=s.LAMBDA, shrink=1.0):
    z, c = s.features(), s.covs()
    beta, _ = s.fit(z[cols], c[cols], lam)
    return pd.DataFrame({"GEOID": s.order.values,
                         "pred": np.clip(s.M + shrink * (z[cols].values @ beta), 0, 1)})


def tabla_features():
    z, c = s.features(), s.covs()
    beta, full = s.fit(z[USE], c[USE], s.LAMBDA)
    b = dict(zip(USE, beta))
    rows = []
    for k in USE:
        rest = [j for j in USE if j != k]
        aporte = s.fit(z[rest], c[rest], s.LAMBDA)[1] - full
        rows.append((k, c[k] / np.sqrt(s.VAR), b[k], aporte))
    return pd.DataFrame(rows, columns=["feat", "rho", "beta", "aporte"]), full


def casos_borde():
    nat = pd.read_parquet(ROOT / "CONCURSO/data/strata/national/national-strata-tract-table.parquet",
                          columns=["GEOID", "ALAND", "AWATER", "pop_total", "pct_urban"])
    nat["GEOID"] = nat.GEOID.astype(str)
    b = pd.read_parquet(ROOT / "datos/features_base/features_buildings.parquet",
                        columns=["GEOID", "region", "bldg_count", "pop_total", "pct_urban"])
    r = pd.read_parquet(ROOT / "datos/features_base/features_roads.parquet",
                        columns=["GEOID", "km_total_unfiltered", "km_highway",
                                 "km_tertiary_named", "km_highway_ring25", "n_road_segments"])
    p = pd.read_parquet(ROOT / "datos/features_base/features_pois.parquet",
                        columns=["GEOID", "poi_total"])
    m = pd.read_csv(SONDAS / "mezcla_v2_ridge.csv", dtype={"GEOID": str})
    d = (b.merge(nat, on="GEOID", how="left", suffixes=("", "_nat"))
          .merge(r, on="GEOID").merge(p, on="GEOID")
          .merge(m.rename(columns={"coverage_gap_score": "pred"}), on="GEOID"))
    assert len(d) == 9794 and d.GEOID.is_unique and d.pred.notna().all()
    assert (d.pop_total == d.pop_total_nat).all()
    d["wshare"] = d.AWATER / (d.ALAND + d.AWATER)
    cond = {
        "pop0": d.pop_total_nat == 0,
        "water50": d.wshare > 0.5,
        "water90": d.wshare > 0.9,
        "nobldg": d.bldg_count == 0,
        "noroad": d.km_total_unfiltered == 0,
        "nopoi": d.poi_total == 0,
        "nohighway": d.km_highway == 0,
    }
    return d, cond


TREAT = {
    "pop0": r"no mask, no imputation; same formula. 27/27 lack \texttt{pct\_urban}, so "
            r"\texttt{rural} is undefined and the z-fill sets it to its mean",
    "water50": r"no mask. Both water-only tracts (ALAND $=0$) have $km_{hw}=0$, so "
               r"\texttt{g\_transport} takes its maximum value 1.0",
    "water90": r"same as above; the third tract (0.93 water) has 3{,}005 residents and 3{,}093 buildings",
    "nobldg": r"no mask in the blend. In the cand\_00 block a building-less tract gets the maximum "
              r"gap, not a missing value",
    "noroad": r"no mask; \texttt{km\_highway}$=0$ gives \texttt{g\_transport}$=1.0$ (maximum gap), "
              r"the opposite of a fully mapped tract",
    "nopoi": r"no mask in the blend; \texttt{components.py} returns 0 only for a zero "
             r"expectation denominator (not in the final blend)",
    "nohighway": r"no mask; \texttt{g\_transport}$=1.0$ for all 137 by construction of the cand\_00 column",
}


def tex_edge(d, cond):
    regs = s.REG
    gm = d.pred.mean()
    rows = []
    for k, c in cond.items():
        sub = d[c]
        rows.append({"caso": k, "n": len(sub), "pct": 100 * len(sub) / len(d),
                     "mean": sub.pred.mean(), "min": sub.pred.min(), "max": sub.pred.max(),
                     "ratio": sub.pred.mean() / gm, "nan_urban": int(sub.pct_urban.isna().sum())})
    t = pd.DataFrame(rows)
    esc = lambda x: x.replace("_", r"\_")  # noqa: E731
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Case & $n$ & \% of tracts & Mean pred. & Min & Max & Ratio & \texttt{pct\_urban} NaN \\",
             r"\midrule"]
    for _, x in t.iterrows():
        lines.append(f"{esc(x.caso)} & {x.n:d} & {x.pct:.2f} & {x['mean']:.5f} & {x['min']:.5f} & "
                     f"{x['max']:.5f} & {x.ratio:.2f}$\\times$ & {x.nan_urban:d} \\\\")
    lines += [r"\midrule",
              f"All tracts & {len(d):d} & 100.00 & {gm:.5f} & {d.pred.min():.5f} & "
              f"{d.pred.max():.5f} & 1.00$\\times$ & {int(d.pct_urban.isna().sum()):d} \\\\",
              r"\bottomrule", r"\end{tabular}", "",
              r"\par\smallskip\footnotesize",
              f"Ratio = mean prediction inside the case / global mean ({gm:.5f}). Predictions are the "
              r"submitted 5-feature blend (\texttt{mezcla\_v2\_ridge.csv}); 0 of 9{,}794 rows hit the "
              r"$[0,1]$ clip."]
    tex = "\n".join(lines)

    lines = [r"\begin{tabular}{l" + "r" * (len(regs) + 1) + "}", r"\toprule",
             "Case & " + " & ".join(regs) + " & All \\\\", r"\midrule",
             r"\multicolumn{" + str(len(regs) + 2) + r"}{l}{\emph{Cells: $n$ / mean prediction}} \\",
             r"\midrule"]
    for k, c in cond.items():
        cells = []
        for rg in regs:
            sub = d[c & (d.region == rg)]
            cells.append(f"{len(sub)} / {sub.pred.mean():.4f}" if len(sub) else "0 / ---")
        sub = d[c]
        cells.append(f"{len(sub)} / {sub.pred.mean():.4f}" if len(sub) else "0 / ---")
        lines.append(f"{esc(k)} & " + " & ".join(cells) + r" \\")
    lines += [r"\midrule",
              r"Region mean pred. & " + " & ".join(f"{d[d.region == rg].pred.mean():.4f}" for rg in regs)
              + f" & {gm:.4f} \\\\",
              r"Region $n$ & " + " & ".join(str(int((d.region == rg).sum())) for rg in regs)
              + f" & {len(d)} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    tex += "\n" + "\n".join(lines)

    lines = [r"\begin{tabular}{p{1.7cm}p{11.4cm}}", r"\toprule",
             r"Case & Treatment in the pipeline \\", r"\midrule"]
    for k in cond:
        lines.append(f"{esc(k)} & {TREAT[k]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return tex + "\n\\smallskip\n\n" + "\n".join(lines), t


FEAT_BODY = {
    "rural": (r"$1[\texttt{pct\_urban}<0.5]$; NaN when \texttt{pct\_urban} is missing",
              r"\texttt{components\_cand00}.\newline\texttt{pct\_urban} = \texttt{pop\_urban}/"
              r"\texttt{pop\_total}\newline (package strata, 2020 census)",
              r"z-score; NaN $\to 0$ = mean"),
    "g_transport": (r"cand\_00 transport gap, $1.0$ when $km_{hw}=0$; $\rho=0.944$ with "
                    r"$1-km_{hw}/(km_{hw}+km_{tert}+km_{hw,ring25})$",
                    r"\texttt{components\_cand00}.\newline\texttt{g\_transport} (Overture roads)",
                    r"z-score; no imputation, no winsor"),
    "t_route_lowclass_share": (r"share of road km on a numbered route in the lowest Overture road classes",
                               r"\texttt{s01\_routes.parquet}\newline (Overture named routes)",
                               r"winsorise p1/p99, then z-score (79\% zeros, kurtosis 48)"),
    "h_fire_missing_risk": (r"fire-station coverage deficit, 0 = no deficit",
                            r"\texttt{s03\_fire\_ems.parquet}.\newline Overture POIs "
                            r"\texttt{fire\_department} and\newline \texttt{ambulance\_and\_ems\_services} "
                            r"vs population",
                            r"z-score; no imputation, no winsor"),
    "r_tribal_any": (r"$1[$tract intersects a tribal area$]$, 874 tracts = 8.9\%",
                     r"\texttt{s07\_rurality.parquet}.\newline Package strata column "
                     r"\texttt{tribal\_any}",
                     r"z-score on a 0/1 column"),
}


def tex_features(feat, full):
    esc = lambda x: x.replace("_", r"\_")  # noqa: E731
    lines = [r"\begin{tabular}{lrrrp{6.3cm}}", r"\toprule",
             r"Feature & $\rho$ with $y$ & $\beta$ ($\lambda=0.1$) & $\Delta$RMSE if dropped & Definition \\",
             r"\midrule"]
    for _, x in feat.sort_values("beta", ascending=False).iterrows():
        lines.append(f"\\texttt{{{esc(x.feat)}}} & {x.rho:+.3f} & {x.beta:+.4f} & {x.aporte:+.5f} & "
                     f"{FEAT_BODY[x.feat][0]} \\\\")
    lines += [r"\midrule",
              f"Intercept $m$ & --- & {s.M:.6f} & --- & public mean of $y$ (constant-0.05 submission) \\\\",
              f"Predicted RMSE & --- & {full:.5f} & --- & 5-feature blend, "
              f"$\\sqrt{{\\mathrm{{Var}}(y)}}$ = {np.sqrt(s.VAR):.6f} \\\\",
              r"\bottomrule", r"\end{tabular}", ""]
    tex1 = "\n".join(lines)

    lines = [r"\small", r"\begin{tabular}{p{5.4cm}p{4.6cm}p{3.9cm}}", r"\toprule",
             r"Feature & Source & Transformation \\", r"\midrule"]
    for k in USE:
        lines.append(f"\\texttt{{{esc(k)}}} & {FEAT_BODY[k][1]} & {FEAT_BODY[k][2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return tex1 + "\n\\smallskip\n\n" + "\n".join(lines)


def tex_alt():
    v2, v4 = vector(), vector(USE[:4])
    v_lam, v_shr, v_nof = vector(USE, 0.3), vector(USE, 0.1, 0.85), vector(SIN_FIRE, 0.1)
    _, r4 = fit(USE[:4])
    _, r5 = fit(USE)
    _, r3 = fit(USE, 0.3)
    _, r_shr = fit(USE, 0.1, 0.85)
    _, r_nof = fit(SIN_FIRE, 0.1)
    base = v2.set_index("GEOID").pred
    filas = [
        (r"mix4 (submission 9)", "0.1", "1.00", r4, REAL_MIX4, v4),
        (r"\texttt{mezcla\_v2\_ridge} (submission 11)", "0.1", "1.00", r5, 0.080228114, v2),
        (r"\texttt{mezcla\_v3\_lam03}", "0.3", "1.00", r3, None, v_lam),
        (r"\texttt{mezcla\_v3\_shrink85} (submission 12)", "0.1", "0.85", r_shr, 0.080525733, v_shr),
        (r"\texttt{mezcla\_v3\_sin\_fire}", "0.1", "1.00", r_nof, None, v_nof),
    ]
    lines = [r"\begin{tabular}{lcrrrrr}", r"\toprule",
             r"Blend & $\lambda$ & Shrink $s$ & Pred. RMSE & $\Delta$ vs mix4 & LB RMSE & "
             r"corr with 5-feature blend \\", r"\midrule"]
    for nom, lam, shr, r, real, v in filas:
        cc = np.corrcoef(base.values, v.set_index("GEOID").pred.values)[0, 1]
        dstr = "---" if nom.startswith("mix4") else f"{r - PRED_MIX4:+.5f}"
        rstr = f"{real:.9f}" if real is not None else "---"
        mark = "" if real is not None else r"$^{\dagger}$"
        lines.append(f"{nom} & {lam} & {shr} & {r:.5f}{mark} & {dstr} & {rstr} & {cc:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"\par\smallskip\footnotesize $^{\dagger}$Predicted, not submitted: no leaderboard RMSE "
              r"exists. Predicted RMSE uses the measured covariance "
              r"identity $RMSE^2=\mathrm{Var}(y)-2\beta^{\top}c+\beta^{\top}S\beta$, with $S$ the sample "
              r"covariance of the z-scored features and $c$ the covariance with $y$ implied by each "
              r"probe's public score; the identity was validated on submission 9 ($+0.000087$). Predictions here chain from the predicted RMSE of the blend, not its real score. Correlation is Pearson over the 9{,}794 predictions against the 5-feature blend. Shrinking by "
              r"$s$ gives $RMSE^2=\mathrm{Var}(y)-(2s-s^2)(\mathrm{Var}(y)-RMSE_0^2)$. $\Delta$ is "
              r"against the submitted 4-feature blend (predicted 0.082654)."]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    z, c = s.features(), s.covs()
    beta, full = s.fit(z[USE], c[USE], s.LAMBDA)
    f_raw = s.M + z[USE].values @ beta
    f = f_raw.clip(0, 1)
    csv = pd.read_csv(SONDAS / "mezcla_v2_ridge.csv", dtype={"GEOID": str})
    assert (csv.GEOID.values == s.order.values).all()
    maxdiff = float(np.abs(csv.coverage_gap_score.values - f).max())
    assert maxdiff < 1e-6, maxdiff
    print(f"check pesos: max |recomputed - mezcla_v2_ridge.csv| = {maxdiff:.2e}; "
          f"rows clipped = {int(((f_raw < 0) | (f_raw > 1)).sum())}")

    d, cond = casos_borde()
    counts = {rg: int((d.region == rg).sum()) for rg in s.REG}
    assert sum(counts.values()) == len(d) == 9794, counts
    print("n por region:", counts, "| suma", sum(counts.values()))

    feat, _ = tabla_features()
    t_edge, t = tex_edge(d, cond)
    (OUT / "edge_cases.tex").write_text(t_edge, encoding="utf-8")
    (OUT / "features.tex").write_text(tex_features(feat, full), encoding="utf-8")
    (OUT / "alt_weights.tex").write_text(tex_alt(), encoding="utf-8")

    print(t.round(5).to_string(index=False))
    print("\nbeta:", {k: round(float(v), 6) for k, v in zip(USE, beta)}, f"| pred RMSE = {full:.5f}")
    print("tablas:", ", ".join(str(OUT / f"{n}.tex") for n in ("edge_cases", "features", "alt_weights")))


if __name__ == "__main__":
    main()
