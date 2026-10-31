"""Best Bias Discovery: cuidado institucional (nursing homes, dialisis, FQHC) fuera de la grilla POI.

Fuentes externas (solo diagnostico del premio, nunca el CSV puntuado):
  CMS Nursing Home Provider Information  data.cms.gov/provider-data/dataset/4pq5-n9py  (NH_ProviderInfo_Oct2026.csv)
  CMS Dialysis Facility Listing          data.cms.gov/provider-data/dataset/23ew-n7w9  (DFC_FACILITY_2.csv)
  HRSA Health Center Service Delivery    data.hrsa.gov/data/download                    (Health_Center_Service_Delivery_and_LookAlike_Sites.csv)
  Census Geocoder (batch)                geocoding.geo.census.gov/geocoder/locations/addressbatch

Entradas internas: Overture POIs y census-tracts de las 5 regiones + strata nacional.
Salidas: l7_cuidado.csv, tablas/care_missing.tex, tablas/care_rates.tex, figuras/care_rates.pdf

Corre:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe procesos/bias_discovery/l7_cuidado.py
"""

import math
import os
import re
import unicodedata
from difflib import SequenceMatcher

import duckdb
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATOS = os.path.join(ROOT, "datos")
EXT_CMS = os.path.join(DATOS, "externos", "cms")      # consultado 2026-10-30
EXT_HRSA = os.path.join(DATOS, "externos", "hrsa")    # consultado 2026-10-30
CSV_OUT = os.path.join(DATOS, "bias_discovery", "l7_cuidado.csv")
TEX_TAB = os.path.join(ROOT, "docs", "2_premios", "writeup", "tablas")
TEX_FIG = os.path.join(ROOT, "docs", "2_premios", "writeup", "figuras")

REGIONS = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx", "eastern-wa"]
STATES = {"AZ", "CA", "OK", "TX", "WA"}

NH_CSV = os.path.join(EXT_CMS, "NH_ProviderInfo_Oct2026.csv")
DF_CSV = os.path.join(EXT_CMS, "DFC_FACILITY_2.csv")
DF_GEO = os.path.join(EXT_CMS, "df_geo_out.csv")
HC_CSV = os.path.join(EXT_HRSA, "Health_Center_Service_Delivery_and_LookAlike_Sites.csv")

OVERTURE_DATE = pd.Timestamp("2026-08-19")
CMS_DATE = pd.Timestamp("2026-10-28")
CUT_DATE = pd.Timestamp("2025-01-01")

LAX_CATS = {
    "health_and_medical", "medical_center", "hospital", "childrens_hospital",
    "public_health_clinic", "community_health_center", "skilled_nursing", "hospice",
    "dialysis_clinic", "assisted_living_facility", "retirement_home", "rehabilitation_center",
    "home_health_care", "medical_service_organizations", "diagnostic_services",
    "diagnostic_imaging", "blood_and_plasma_donation_center", "urgent_care_clinic",
    "walk_in_clinic", "mobile_clinic", "senior_citizen_services", "pharmacy",
    "medical_supply", "physical_therapy", "occupational_therapy",
    "counseling_and_mental_health", "womens_health_clinic", "prenatal_perinatal_care",
    "physician_assistant", "clinical_laboratories", "wellness_program", "health_coach",
    "memory_care", "nursing_home", "surgeon", "health_department",
    "addiction_rehabilitation_center", "foster_care_services", "medical_transportation",
    "medical_research_and_development", "health_and_wellness_club",
    # practicas clinicas y especialidades (anadidas tras revisar la lista completa de categorias)
    "family_practice", "internal_medicine", "pediatrician", "obstetrician_and_gynecologist",
    "geriatric_medicine", "sports_medicine", "emergency_medicine", "occupational_medicine",
    "preventive_medicine", "optometrist", "ophthalmologist", "psychologist", "psychiatrist",
    "psychotherapist", "speech_therapist", "laboratory_testing", "laboratory",
    "ambulance_and_ems_services", "disability_services_and_support_organization",
    "geriatric_psychiatry", "child_psychiatrist", "vascular_medicine",
    "pediatric_cardiology", "pediatric_gastroenterology", "pediatric_surgery",
    "pediatric_endocrinology", "pediatric_pulmonology", "pediatric_neurology",
    "pediatric_orthopedic_surgery", "pediatric_oncology", "pediatric_nephrology",
    "pediatric_anesthesiology", "pediatric_radiology", "emergency_medicine",
}
STATE_TOKENS = {"TEXAS", "OKLAHOMA", "ARIZONA", "CALIFORNIA", "WASHINGTON", "TX", "OK",
                "AZ", "CA", "WA"}


NAME_SIM = 0.6
STRICT_R = 250.0
LAX_CAT_R = 100.0
LAX_NAME_R = 1000.0
ADDR_R = 50.0

# Falsos faltantes medidos en la muestra estratificada de la seccion 6
# (50 rurales + 50 urbanos, semilla 42). Se actualiza tras leer la muestra.
VALIDACION = {"rural": (1, 35), "urban": (1, 50)}

_LEGAL = {"INC", "LLC", "LP", "LLP", "LTD", "CORP", "CO", "PC", "PLLC", "PA", "DBA"}
_STOP = {"THE", "OF", "AND", "A", "AN", "AT", "FOR", "ON", "IN"}
_ABBR = {
    "CTR": "CENTER", "CNTR": "CENTER", "HLTH": "HEALTH", "MED": "MEDICAL",
    "HOSP": "HOSPITAL", "REHAB": "REHABILITATION", "COMM": "COMMUNITY",
    "NURS": "NURSING", "MNOR": "MANOR", "HLTHCARE": "HEALTHCARE",
    "SPEC": "SPECIALTY", "ASSN": "ASSOCIATION", "DIST": "DISTRICT",
    "MEM": "MEMORIAL", "REG": "REGIONAL", "SVCS": "SERVICES", "SVC": "SERVICE",
    "CNTY": "COUNTY", "ST": "SAINT", "MT": "MOUNT", "UNIV": "UNIVERSITY",
    "CLIN": "CLINIC", "FAM": "FAMILY", "PEDS": "PEDIATRIC", "GEN": "GENERAL",
}
# Cadenas del sector: el operador aparece abreviado en un lado y completo en el otro.
_BRAND = [
    (re.compile(r"\bFRESENIUS\b|\bFKC\b|\bFMC\b"), "FRESENIUS"),
    (re.compile(r"\bDAVITA\b"), "DAVITA"),
    (re.compile(r"\bU S RENAL\b|\bUSRENAL\b|\bUSRC\b|\bU S R C\b"), "USRENAL"),
    (re.compile(r"\bBIOMEDICAL APPLICATIONS\b|\bBMA\b"), "BMA"),
    (re.compile(r"\bENCOMPASS\b"), "ENCOMPASS"),
    (re.compile(r"\bLIFE CARE CENTERS?\b|\bLCCA\b"), "LIFECARE"),
    (re.compile(r"\bSIGNATURE HEALTHCARE\b"), "SIGNATURE"),
    (re.compile(r"\bCONSULATE\b"), "CONSULATE"),
]
# Palabras de modalidad o tipo: no identifican el centro, solo el servicio.
_GENERIC = {
    "NURSING", "NURS", "REHABILITATION", "REHAB", "CENTER", "CTR", "CNTR", "HEALTH",
    "HEALTHCARE", "HLTH", "CARE", "CLINIC", "CLIN", "MEDICAL", "MED", "MEDICINE",
    "HOSPITAL", "HOSP", "FACILITY", "DIALYSIS", "KIDNEY", "UNIT", "DEPT", "DEPARTMENT",
    "SERVICES", "SERVICE", "SVCS", "SVC", "GROUP", "HEALTHCARECENTER", "REHABCARE",
    "TRANSITIONAL", "SPECIALTY", "SKILLED", "LIVING", "ASSISTED",
}


def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pat, rep in _BRAND:
        s = pat.sub(" " + rep + " ", s)
    toks = [_ABBR.get(t, t) for t in s.split()]
    toks = [t for t in toks if t and t not in _STOP and t not in _LEGAL]
    return " ".join(toks)


def norm_key(s):
    """Clave de identidad: cadena canonica + sin palabras de modalidad."""
    toks = [t for t in norm_name(s).split() if t not in _GENERIC]
    return " ".join(toks) if toks else norm_name(s)


def sim(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def contained(a, b):
    """Un nombre esta contenido en el otro (con >=2 tokens en el mas corto)."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) < 2 or len(" ".join(short)) < 10:
        return False
    return all(t in long_ for t in short)


_cache = {}


def key_of(s):
    if s not in _cache:
        _cache[s] = norm_key(s)
    return _cache[s]


# ---------------------------------------------------------------- fuentes externas
def load_nursing_homes():
    d = pd.read_csv(NH_CSV, dtype=str, keep_default_na=False)
    d = d[d["State"].isin(STATES)].copy()
    out = pd.DataFrame({
        "id": "NH_" + d["CMS Certification Number (CCN)"],
        "nombre": d["Provider Name"],
        "direccion": d["Provider Address"] + ", " + d["City/Town"] + ", " + d["State"] + " " + d["ZIP Code"],
        "ciudad": d["City/Town"],
        "estado": d["State"],
        "condado_nombre": d["County/Parish"],
        "lat": pd.to_numeric(d["Latitude"], errors="coerce"),
        "lon": pd.to_numeric(d["Longitude"], errors="coerce"),
        "tipo": "nursing_home",
        "camas_nh": pd.to_numeric(d["Number of Certified Beds"], errors="coerce"),
        "estaciones_dx": np.nan,
        "residentes_dia_nh": pd.to_numeric(d["Average Number of Residents per Day"], errors="coerce"),
        "pacientes_dx": np.nan,
        "fecha": pd.to_datetime(d["Date First Approved to Provide Medicare and Medicaid Services"],
                                errors="coerce"),
        "fuente": "CMS NH Provider Information (4pq5-n9py, rel. 2026-10-28)",
    })
    return out.dropna(subset=["lat", "lon"])


def load_dialysis():
    d = pd.read_csv(DF_CSV, dtype=str, keep_default_na=False)
    d = d[d["State"].isin(STATES)].copy()
    g = pd.read_csv(DF_GEO, header=None, dtype=str, keep_default_na=False,
                    names=["id", "addr_in", "match", "type", "addr_out", "xy", "hash", "side"])
    g = g[g["match"] == "Match"].copy()
    g["lon"] = pd.to_numeric(g["xy"].str.split(",").str[0], errors="coerce")
    g["lat"] = pd.to_numeric(g["xy"].str.split(",").str[1], errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).drop_duplicates("id")
    d = d.merge(g[["id", "lat", "lon", "match"]], left_on="CMS Certification Number (CCN)",
                right_on="id", how="left")
    out = pd.DataFrame({
        "id": "DF_" + d["CMS Certification Number (CCN)"],
        "nombre": d["Facility Name"],
        "direccion": d["Address Line 1"] + ", " + d["City/Town"] + ", " + d["State"] + " " + d["ZIP Code"],
        "ciudad": d["City/Town"],
        "estado": d["State"],
        "condado_nombre": d["County/Parish"],
        "lat": d["lat"],
        "lon": d["lon"],
        "tipo": "dialysis",
        "camas_nh": np.nan,
        "estaciones_dx": pd.to_numeric(d["# of Dialysis Stations"], errors="coerce"),
        "residentes_dia_nh": np.nan,
        "pacientes_dx": pd.to_numeric(d["Number of patients included in hospitalization summary"],
                                     errors="coerce"),
        "fecha": pd.to_datetime(d["Certification Date"], errors="coerce", format="%m/%d/%Y"),
        "fuente": "CMS Dialysis Facility Listing (23ew-n7w9, rel. 2026-10-28) + Census Geocoder",
    })
    geo_ok = d["match"].eq("Match").sum()
    print(f"[geo] dialisis: {geo_ok}/{len(d)} geocodificadas ({100*geo_ok/len(d):.1f}%); "
          f"{len(d)-geo_ok} sin coordenada quedan fuera")
    return out.dropna(subset=["lat", "lon"])


def load_health_centers():
    d = pd.read_csv(HC_CSV, dtype=str, keep_default_na=False)
    d = d[d["Site State Abbreviation"].isin(STATES)].copy()
    d = d[d["Health Center Type Description"] != "Administrative"].copy()
    d = d[d["Health Center Location Type Description"].isin(["Permanent", "Seasonal"])].copy()
    out = pd.DataFrame({
        "id": "HC_" + d["BPHC Assigned Number"].astype(str) + "_" + d.index.astype(str),
        "nombre": d["Site Name"],
        "direccion": d["Site Address"] + ", " + d["Site City"] + ", " + d["Site State Abbreviation"] + " " + d["Site Postal Code"],
        "ciudad": d["Site City"],
        "estado": d["Site State Abbreviation"],
        "condado_nombre": d["Complete County Name"],
        "lat": pd.to_numeric(d["Geocoding Artifact Address Primary Y Coordinate"], errors="coerce"),
        "lon": pd.to_numeric(d["Geocoding Artifact Address Primary X Coordinate"], errors="coerce"),
        "tipo": "health_center",
        "camas_nh": np.nan,
        "estaciones_dx": np.nan,
        "residentes_dia_nh": np.nan,
        "pacientes_dx": np.nan,
        "fecha": pd.to_datetime(d["Site Added to Scope this Date"], errors="coerce", format="%m/%d/%Y"),
        "fuente": "HRSA Health Center Service Delivery Sites (data.hrsa.gov, desc. 2026-10-30)",
    })
    return out.dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------- geometria
def build_tracts(con):
    parts = []
    for r in REGIONS:
        p = os.path.join(ROOT, "CONCURSO", "data", "strata", r, f"{r}-census-tracts.parquet")
        parts.append(
            f"SELECT GEOID, '{r}' AS region, STATEFP, COUNTYFP, NAMELSAD, pop_total, pct_urban, "
            f"bbox.xmin AS x0, bbox.xmax AS x1, bbox.ymin AS y0, bbox.ymax AS y1, geometry "
            f"FROM read_parquet('{p.replace(os.sep, '/')}')"
        )
    con.execute("CREATE TABLE tracts AS " + " UNION ALL ".join(parts))
    con.execute("""CREATE TABLE tract_pt AS
        SELECT f.id, t.GEOID AS tracto, t.region, t.NAMELSAD AS tracto_nombre,
               ROW_NUMBER() OVER (PARTITION BY f.id ORDER BY t.GEOID) rn
        FROM fac f JOIN tracts t
          ON f.lon BETWEEN t.x0 AND t.x1 AND f.lat BETWEEN t.y0 AND t.y1
         AND ST_Contains(t.geometry, ST_Point(f.lon, f.lat))""")
    con.execute("""CREATE TABLE fac_geo AS
        SELECT f.*, tp.tracto, tp.region, tp.tracto_nombre FROM fac f
        LEFT JOIN (SELECT id, tracto, region, tracto_nombre FROM tract_pt WHERE rn = 1) tp USING (id)""")


def load_pois(con):
    files = []
    for r in REGIONS:
        p = os.path.join(ROOT, "CONCURSO", "data", "reference", r, f"{r}-overture-pois.parquet")
        files.append(f"'{p.replace(os.sep, '/')}'")
    con.execute(f"""CREATE TABLE pois AS
        SELECT names.primary AS pnom, categories.primary AS pcat,
               ST_Y(geometry) AS plat, ST_X(geometry) AS plon
        FROM read_parquet([{",".join(files)}])
        WHERE geometry IS NOT NULL""")


def build_pairs(con):
    con.execute("""CREATE TABLE pairs AS
        SELECT g.id, p.pnom, p.pcat, p.plat, p.plon,
               2*6371008.8*ASIN(SQRT(POWER(SIN(RADIANS(p.plat-g.lat)/2),2)
                 + COS(RADIANS(g.lat))*COS(RADIANS(p.plat))
                   *POWER(SIN(RADIANS(p.plon-g.lon)/2),2))) AS d
        FROM fac_geo g JOIN pois p
          ON ABS(p.plat - g.lat) < 0.012 AND ABS(p.plon - g.lon) < 0.016
        WHERE g.lat IS NOT NULL""")


def match_pairs(con, namemap, citymap):
    pr = con.execute("SELECT id, pnom, pcat, d FROM pairs WHERE d <= 1000").df()
    fk = pr["id"].map(namemap)
    pk = pr["pnom"].map(lambda s: key_of(s) if isinstance(s, str) else "")
    pr["sim"] = [sim(a, b) for a, b in zip(fk, pk)]
    pr["cont"] = [contained(a, b) for a, b in zip(fk, pk)]
    pr["nameok"] = (pr["sim"] >= NAME_SIM) | pr["cont"]
    # Regla de identidad: mismo predio mapeado con otro rotulo. Comparte >=2 palabras
    # distintivas (>=5 letras, fuera de ciudad/estado/modalidad) a <=250 m, o >=1 a <=100 m.
    near = pr["d"] <= 250
    ft = pr.loc[near, "id"].map(lambda i: _toks(namemap[i], citymap[i]))
    pt = pr.loc[near, "pnom"].map(lambda s: _toks(key_of(s) if isinstance(s, str) else "", ""))
    shared = np.array([len(a & b) for a, b in zip(ft, pt)], dtype=int)
    dd = pr.loc[near, "d"].to_numpy()
    hit = (shared >= 2) | ((shared >= 1) & (dd <= 100))
    pr["ident"] = False
    pr.loc[near, "ident"] = hit
    pr["nameok"] = pr["nameok"] | pr["ident"]
    pr["ok_strict"] = ((pr["sim"] >= NAME_SIM) | pr["cont"]) & (pr["d"] <= STRICT_R)
    pr["ok_lax"] = (pr["pcat"].isin(LAX_CATS) & (pr["d"] <= LAX_CAT_R)) \
        | (pr["nameok"] & (pr["d"] <= LAX_NAME_R))
    return pr


def _toks(key, city):
    out = set()
    for t in key.split():
        if len(t) < 5 or t in _GENERIC or t in STATE_TOKENS or t.isdigit():
            continue
        out.add(t[:6])
    for t in (city or "").split():
        if len(t) >= 3:
            out.discard(t)
    return out


# ---------------------------------------------------------------- estadistica
def mh_stats(t, seed=42):
    """OR_MH + IC95% (Robins-Breslow-Greenland) + bootstrap por estrato."""
    t = t[(t.a + t.b + t.c + t.e) >= 2].copy()
    if t.empty:
        return None
    t["n"] = t.a + t.b + t.c + t.e
    R = (t.a * t.e / t.n).sum()
    S = (t.b * t.c / t.n).sum()
    k_inf = int((t[["a", "b", "c", "e"]] > 0).all(axis=1).sum())
    base = dict(k=len(t), k_inf=k_inf, n=int(t.n.sum()),
                a=float(t.a.sum()), b=float(t.b.sum()),
                c=float(t.c.sum()), e=float(t.e.sum()))
    if R <= 0 or S <= 0:
        return dict(base, or_mh=np.nan, lo=np.nan, hi=np.nan, boot=None)
    or_mh = R / S
    P = ((t.a + t.e) * t.a * t.e / t.n ** 2).sum()
    Q = ((t.a + t.e) * t.b * t.c / t.n ** 2).sum()
    se = math.sqrt(max(P / (2 * R ** 2) + Q / (2 * S ** 2) + (P + Q) / (2 * R * S), 0))
    return dict(base, or_mh=or_mh, lo=math.exp(math.log(or_mh) - 1.96 * se),
                hi=math.exp(math.log(or_mh) + 1.96 * se), se=se,
                boot=bootstrap_or(t, seed=seed))


def mh_or(df, expo, case="faltante"):
    d = df.dropna(subset=[expo, "estrato"]).copy()
    d[expo] = d[expo].astype(bool)
    d[case] = d[case].astype(bool)
    rows = []
    for s, g in d.groupby("estrato"):
        rows.append((s, int((g[expo] & g[case]).sum()), int((g[expo] & ~g[case]).sum()),
                     int((~g[expo] & g[case]).sum()), int((~g[expo] & ~g[case]).sum())))
    return mh_stats(pd.DataFrame(rows, columns=["estrato", "a", "b", "c", "e"]))


def bootstrap_or(t, nboot=2000, seed=42):
    """IC percentil del OR_MH resampleando condados con reemplazo."""
    A, B, C, E = (t[["a", "b", "c", "e"]].to_numpy(dtype=float).T)
    strata = t["n"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(nboot, len(A)))
    a, b, c, e, n = A[idx], B[idx], C[idx], E[idx], strata[idx]
    R = (a * e / n).sum(1)
    S = (b * c / n).sum(1)
    ok = (R > 0) & (S > 0)
    vals = np.full(nboot, np.nan)
    vals[ok] = R[ok] / S[ok]
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper(k, n, alpha=0.05):
    from scipy.stats import beta
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def fmt_mh(tag, r, width=34):
    if r is None:
        print(f"  {tag:<{width}} sin estratos informativos")
        return
    if not np.isfinite(r["or_mh"]):
        print(f"  {tag:<{width}} OR indefinido (n={r['n']}, estratos={r['k']}, inf.={r['k_inf']})")
        return
    sig = "IC excluye 1" if (r["lo"] > 1 or r["hi"] < 1) else "IC incluye 1"
    bt = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
    print(f"  {tag:<{width}} OR_MH={r['or_mh']:.2f} IC95%[{r['lo']:.2f},{r['hi']:.2f}]{bt} "
          f"condados={r['k']} (inf.={r['k_inf']}) n={r['n']} -> {sig}")


# ---------------------------------------------------------------- main
def main():
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='8GB'")
    con.execute("INSTALL spatial; LOAD spatial;")

    nh = load_nursing_homes()
    dx = load_dialysis()
    hc = load_health_centers()
    fac = pd.concat([nh, dx, hc], ignore_index=True).dropna(subset=["lat", "lon"])
    fac = fac.drop_duplicates("id")
    print(f"[src] nursing={len(nh)} dialisis={len(dx)} health_center={len(hc)} -> {len(fac)} en 5 estados")
    con.register("fac_df", fac)
    con.execute("CREATE TABLE fac AS SELECT * FROM fac_df")

    build_tracts(con)
    facg = con.execute("SELECT * FROM fac_geo").df()
    nat = os.path.join(ROOT, "CONCURSO", "data", "strata", "national",
                       "national-strata-tract-table.parquet").replace(os.sep, "/")
    strata = con.execute(f"""SELECT GEOID, COUNTYFP, STATEFP, pct_urban, pop_total,
                                    svi_overall, tribal_pct, aiannh_name
                             FROM read_parquet('{nat}')""").df()
    geo = facg.dropna(subset=["tracto"]).merge(
        strata, left_on="tracto", right_on="GEOID", how="left", suffixes=("", "_n"))
    geo["condado"] = geo["STATEFP"].astype(str) + geo["COUNTYFP"].astype(str)
    print(f"[geo] dentro del AOI: {len(geo)}/{len(fac)}")

    load_pois(con)
    con.register("fac_ll", geo[["id", "lat", "lon"]])
    con.execute("CREATE OR REPLACE TABLE fac_geo AS SELECT * FROM fac_ll")
    build_pairs(con)
    pr = match_pairs(con, dict(zip(geo["id"], geo["nombre"].map(key_of))),
                     dict(zip(geo["id"], geo["ciudad"].map(key_of))))
    print(f"[poi] pares <=1km evaluados: {len(pr)}")

    agg = pr.groupby("id").agg(
        encontrado_estricto=("ok_strict", "max"),
        encontrado_laxo=("ok_lax", "max"),
        mejor_poi_dist=("d", "min"),
        sim_max=("sim", "max"),
        dens_1km=("d", "size"),
        any50=("d", lambda s: bool((s <= ADDR_R).any())),
    )
    near = pr.sort_values("d").groupby("id").first()[["pnom", "pcat", "d"]]
    near.columns = ["poi_cercano", "poi_cat", "poi_cercano_dist"]
    care = pr[pr["pcat"].isin(LAX_CATS)].sort_values("d").groupby("id").first()[["pnom", "d"]]
    care.columns = ["poi_cuidado", "poi_cuidado_dist"]

    out = geo.merge(agg, left_on="id", right_index=True, how="left") \
             .merge(near, left_on="id", right_index=True, how="left") \
             .merge(care, left_on="id", right_index=True, how="left")
    for c_ in ["encontrado_estricto", "encontrado_laxo", "any50"]:
        out[c_] = out[c_].fillna(False).astype(bool)
    out["encontrado"] = out["encontrado_laxo"]
    out["faltante"] = ~out["encontrado"]
    out["rural"] = out["pct_urban"] < 0.5
    out["tribal"] = out["tribal_pct"].fillna(0) > 0
    out["svi_alto"] = out["svi_overall"] >= 0.75
    out["svi_bajo"] = out["svi_overall"] <= 0.25
    out["estrato"] = out["condado"]

    # ---------------- seccion 2: desfase temporal
    print("\n=== 0. DESFASE TEMPORAL (Overture 2026-08-19 vs CMS 2026-10-28) ===")
    sin_fecha = out["fecha"].isna().sum()
    viejos = out["fecha"] <= CUT_DATE
    print(f"  con fecha de alta: {out['fecha'].notna().sum()}; sin fecha: {sin_fecha}")
    print(f"  excluidos por alta > {CUT_DATE.date()}: {(~viejos & out['fecha'].notna()).sum()}")
    for k, g in out.groupby("tipo"):
        vg = g["fecha"] <= CUT_DATE
        print(f"    {k:<14} n={len(g):>4} sin fecha={g['fecha'].isna().sum():>3} "
              f"excluidos={(~vg & g['fecha'].notna()).sum():>3} "
              f"faltante antes={g['faltante'].mean()*100:5.1f}% "
              f"despues={g.loc[vg | g['fecha'].isna(), 'faltante'].mean()*100:5.1f}%")
    out_all = out.copy()
    out = out[viejos | out["fecha"].isna()].copy()
    print(f"  base final: {len(out)} centros (de {len(out_all)})")
    fmt_mh("rural, base completa (sin filtro)", mh_or(out_all, "rural"))
    fmt_mh("rural, base filtrada por fecha", mh_or(out, "rural"))

    # ---------------- seccion 1
    m1 = out[out["faltante"]]
    # Correccion por nombre: si existe un POI con nombre casi identico pero mas lejos que
    # 1 km, el centro SI esta en Overture y lo que falla es el par de coordenadas.
    con.register("qm", m1[["id", "lat", "lon"]])
    far_all = con.execute("""SELECT q.id, p.pnom,
              2*6371008.8*ASIN(SQRT(POWER(SIN(RADIANS(p.plat-q.lat)/2),2)
                + COS(RADIANS(q.lat))*COS(RADIANS(p.plat))
                  *POWER(SIN(RADIANS(p.plon-q.lon)/2),2))) AS d
        FROM qm q JOIN pois p ON ABS(p.plat-q.lat) < 0.056 AND ABS(p.plon-q.lon) < 0.068""").df()
    far_all = far_all[far_all.d > LAX_NAME_R]
    kn = dict(zip(m1["id"], m1["nombre"].map(key_of)))
    far_all["sim"] = [sim(kn.get(i, ""), key_of(n) if isinstance(n, str) else "")
                      for i, n in zip(far_all["id"], far_all["pnom"])]
    far_all = far_all[(far_all.sim >= 0.85) & (far_all.d <= 6000)]
    corr = far_all.sort_values("d").groupby("id").first()
    out["nombre_lejos"] = out["id"].map(corr["pnom"])
    out["dist_lejos"] = out["id"].map(corr["d"]).round(0)
    out["faltante_c"] = out["faltante"] & ~out["id"].isin(corr.index)

    print("\n=== 1. TASA DE FALTANTE ===")
    print(f"  r={len(out)} centros  laxa={out.faltante.mean()*100:.1f}% ({out.faltante.sum()})  "
          f"estricta={(~out.encontrado_estricto).mean()*100:.1f}% ({(~out.encontrado_estricto).sum()})")
    print(f"  de {int(out.faltante.sum())} faltantes, "
          f"{int((out.faltante & ~out['faltante_c']).sum())} tienen POI de nombre casi identico "
          f"a <=6 km -> tasa corregida {out['faltante_c'].mean()*100:.1f}% "
          f"({int(out['faltante_c'].sum())})")
    lax_base = (pr["pcat"].isin(LAX_CATS) & (pr["d"] <= LAX_CAT_R)) | \
        (((pr["sim"] >= NAME_SIM) | pr["cont"]) & (pr["d"] <= LAX_NAME_R))
    rescat = pr.loc[pr["ok_lax"] & ~lax_base, "id"].nunique()
    print(f"  alias de cadena + regla de identidad rescataron {rescat} centros "
          f"({rescat/len(out)*100:.1f}% de la base) frente al matcher de solo difflib")
    for k, g in out.groupby("tipo"):
        lo, hi = wilson(g.faltante.sum(), len(g))
        loc, hic = wilson(g["faltante_c"].sum(), len(g))
        print(f"    {k:<14} n={len(g):>4} laxa={g.faltante.mean()*100:5.1f}% [{lo*100:.1f},{hi*100:.1f}] "
              f"corregida={g['faltante_c'].mean()*100:5.1f}% [{loc*100:.1f},{hic*100:.1f}]")

    print("\n=== 2. POR REGION Y TIPO ===")
    reg = out.groupby("region").agg(n=("id", "size"), falt=("faltante", "sum"),
                                    faltc=("faltante_c", "sum"))
    reg["tasa"] = (reg["falt"] / reg["n"] * 100).round(1)
    reg["tasa_corr"] = (reg["faltc"] / reg["n"] * 100).round(1)
    reg[["lo", "hi"]] = [np.round(np.array(wilson(r.falt, r.n)) * 100, 1) for r in reg.itertuples()]
    print(reg.to_string())
    print("\n  tasa laxa (%) / corregida (%) por region x tipo (n entre parentesis):")
    for r_ in REGIONS:
        row = []
        for k in ["nursing_home", "dialysis", "health_center"]:
            g = out[(out.region == r_) & (out.tipo == k)]
            row.append(f"{k[:6]}={g.faltante.mean()*100:5.1f}/{g['faltante_c'].mean()*100:4.1f}"
                       f"({len(g)})" if len(g) else f"{k[:6]}=  n/a(0)")
        print(f"    {r_:<17} " + "  ".join(row))

    # ---------------- seccion 3
    print("\n=== 3. DISPARIDAD: MH + BOOTSTRAP POR CONDADO ===")
    res = {}
    res["rural"] = mh_or(out, "rural")
    fmt_mh("rural vs urbano", res["rural"])
    res["rural_c"] = mh_or(out, "rural", case="faltante_c")
    fmt_mh("rural vs urbano (corregido)", res["rural_c"])
    for col, lab in [("tribal", "tribal vs no tribal"),
                     ("svi_alto", "SVI alto vs bajo")]:
        dd = out[out["svi_alto"] | out["svi_bajo"]] if col == "svi_alto" else out
        res[col] = mh_or(dd, col)
        fmt_mh(lab, res[col])
    print("  --- por tipo de centro (condado como estrato) ---")
    for k, g in out.groupby("tipo"):
        res["rural_" + k] = mh_or(g, "rural")
        fmt_mh("rural | " + k, res["rural_" + k])
        res["ruralc_" + k] = mh_or(g, "rural", case="faltante_c")
        fmt_mh("rural | " + k + " (corr.)", res["ruralc_" + k])
    print("  --- por region (condado como estrato) ---")
    for r_ in REGIONS:
        g = out[out["region"] == r_]
        res["rural_" + r_] = mh_or(g, "rural")
        fmt_mh("rural | " + r_, res["rural_" + r_])
        res["ruralc_" + r_] = mh_or(g, "rural", case="faltante_c")
        fmt_mh("rural | " + r_ + " (corr.)", res["ruralc_" + r_])

    print("\n=== 3b. LECTURA CON DENSIDAD DE POIs (posible mediador) ===")
    out["dens_tercil"] = pd.qcut(out["dens_1km"], 3, labels=["baja", "media", "alta"])
    print(f"  POIs <=1km por centro: mediana rural={out[out.rural]['dens_1km'].median():.0f} "
          f"urbano={out[~out.rural]['dens_1km'].median():.0f}")
    dt = out.pivot_table(index="dens_tercil", columns="rural", values="faltante_c", aggfunc="mean") * 100
    dn = out.pivot_table(index="dens_tercil", columns="rural", values="faltante_c", aggfunc="size")
    print(pd.concat([dt.round(1).add_suffix("_tasa%"), dn.add_suffix("_n")], axis=1).to_string())
    dd = out.copy()
    dd["estrato"] = dd["dens_tercil"]
    res["rural_dens"] = mh_or(dd, "rural", case="faltante_c")
    fmt_mh("rural, densidad (corregido)", res["rural_dens"])

    # ---------------- seccion 4: especificidad
    print("\n=== 4. ESPECIFICIDAD: cuidado vs densidad general de POIs ===")
    m = out[out["faltante_c"]]
    fnd = out[~out["faltante_c"]]
    print(f"  (sobre los {len(m)} faltantes corregidos; los {int(out['faltante'].sum())-len(m)} "
          f"faltantes de la regla de 1 km con POI homonimo a <=6 km quedan fuera)")
    print(f"  faltantes con algun POI a <=50 m (direccion mapeada, institucion ausente): "
          f"{m.any50.mean()*100:.1f}% ({m.any50.sum()}/{len(m)})")
    print(f"  encontrados con algun POI a <=50 m: {fnd.any50.mean()*100:.1f}% ({fnd.any50.sum()}/{len(fnd)})")
    print(f"  distancia al POI mas cercano: faltantes mediana={m.poi_cercano_dist.median():.0f} m "
          f"(p25={m.poi_cercano_dist.quantile(.25):.0f}, p75={m.poi_cercano_dist.quantile(.75):.0f}); "
          f"encontrados mediana={fnd.poi_cercano_dist.median():.0f} m")
    print(f"  faltantes sin NINGUN POI a <=100 m: {(m.poi_cercano_dist > 100).mean()*100:.1f}%")
    print("  tasa por tercil de densidad x tipo (si la densidad explica todo, se iguala):")
    tt = out.pivot_table(index="dens_tercil", columns="tipo", values="faltante_c", aggfunc="mean") * 100
    tn = out.pivot_table(index="dens_tercil", columns="tipo", values="faltante_c", aggfunc="size")
    print(pd.concat([tt.round(1).add_suffix("%"), tn.add_suffix("_n")], axis=1).to_string())
    dd = out[out.dens_tercil.notna()].copy()
    dd["estrato"] = dd["dens_tercil"]
    dd["nh_"] = dd["tipo"].eq("nursing_home")
    res["nh_dens"] = mh_or(dd, "nh_", case="faltante_c")
    fmt_mh("NH vs resto, densidad (corregido)", res["nh_dens"])
    dm = m[m.any50]
    print(f"  de los {len(dm)} faltantes con direccion mapeada: "
          f"{(dm.poi_cercano_dist <= 25).mean()*100:.0f}% tiene POI a <=25 m")

    # ---------------- seccion 5: unidades
    print("\n=== 5. CARGA AFECTADA (unidades separadas, no se suman) ===")
    for k, g in m.groupby("tipo"):
        print(f"  {k:<14} n={len(g):>3}  camas_certificadas={g['camas_nh'].sum():>7.0f}  "
              f"estaciones_dialisis={g['estaciones_dx'].sum():>5.0f}  "
              f"residentes_dia={g['residentes_dia_nh'].sum():>7.0f}  "
              f"pacientes_dx_1anio={g['pacientes_dx'].sum():>7.0f}")
    print(f"  TOTAL faltantes n={len(m)}")
    chk = out[out["nombre"].str.contains("Gateway", na=False) & out["ciudad"].eq("Chandler")]
    for _, r_ in chk.iterrows():
        print(f"  [unidad] {r_['nombre']} ({r_['ciudad']},{r_['estado']}): estaciones="
              f"{r_['estaciones_dx']:.0f} pacientes_dx_1anio={r_['pacientes_dx']:.0f} "
              f"-> el 763 del campo 'survival summary' es cohorte multi-anio, NO censo concurrente")

    # ---------------- seccion 6: validacion estratificada
    print("\n=== 6. VALIDACION ESTRATIFICADA 50 RURALES + 50 URBANOS (semilla 42) ===")
    print("  Criterio de falso faltante, escrito ANTES de mirar la muestra. El centro SI esta en")
    print("  Overture si existe un POI que sea la misma institucion: (a) nombre normalizado con")
    print("  difflib>=0.6 o contencion de tokens a <=1 km; (b) misma cadena canonica (Fresenius/FKC/")
    print("  FMC, DaVita, US Renal/USRC, BMA, Encompass, LifeCare); (c) >=2 palabras distintivas")
    print("  compartidas (>=5 letras, fuera de ciudad/estado/modalidad) a <=250 m; (d) >=1 a <=100 m;")
    print("  (e) categoria de cuidado a <=100 m. Todo lo que no cumpla: faltante verdadero.")
    muestras = {}
    for estr, lab in [(True, "RURAL"), (False, "URBANO")]:
        pool = m[m.rural == estr]
        muestras[estr] = pool.sample(n=min(50, len(pool)), random_state=42)
    samp = pd.concat([muestras[True], muestras[False]])
    # Nombre casi identico FUERA de la ventana de 1 km: separa falso faltante por
    # desfase de coordenadas de falso faltante por ausencia real en Overture.
    con.register("q", samp[["id", "lat", "lon"]])
    far = con.execute("""SELECT q.id, p.pnom, p.pcat,
              2*6371008.8*ASIN(SQRT(POWER(SIN(RADIANS(p.plat-q.lat)/2),2)
                + COS(RADIANS(q.lat))*COS(RADIANS(p.plat))
                  *POWER(SIN(RADIANS(p.plon-q.lon)/2),2))) AS d
        FROM q JOIN pois p ON ABS(p.plat-q.lat) < 0.056 AND ABS(p.plon-q.lon) < 0.068""").df()
    far = far[far.d > 1000]
    kk = dict(zip(samp["id"], samp["nombre"].map(key_of)))
    far["sim"] = [sim(kk.get(i, ""), key_of(n) if isinstance(n, str) else "")
                  for i, n in zip(far["id"], far["pnom"])]
    far = far[far.sim >= 0.85].sort_values("d").groupby("id").first()
    for estr, lab in [(True, "RURAL"), (False, "URBANO")]:
        print(f"\n  --- {lab}: {len(muestras[estr])} de {len(pool)} faltantes ---")
        for i, (_, r_) in enumerate(muestras[estr].iterrows(), 1):
            c = pr[pr["id"] == r_["id"]].sort_values("d").head(3)
            vec = " | ".join(f"{x.pnom} [{x.pcat}] {x.d:.0f}m" for x in c.itertuples())
            lejos = ""
            if r_["id"] in far.index:
                f_ = far.loc[r_["id"]]
                lejos = f"  <<LEJOS {f_['pnom']} {f_['d']/1000:.1f}km sim={f_['sim']:.2f}>>"
            print(f"  {i:2d}. {r_['nombre']} ({r_['tipo']}, {r_['ciudad']} {r_['estado']}, "
                  f"tracto {r_['tracto']}, camas={r_['camas_nh']:.0f}, est={r_['estaciones_dx']:.0f})")
            print(f"      -> {vec if vec else 'SIN POI EN 1 KM'}{lejos}")
    n_far = {True: int(muestras[True]["id"].isin(far.index).sum()),
             False: int(muestras[False]["id"].isin(far.index).sum())}
    print(f"  nombre casi identico fuera de 1 km (desfase de coordenadas), sim>=0.85: "
          f"rural {n_far[True]}/{len(muestras[True])}, urbano {n_far[False]}/{len(muestras[False])}")

    # ---------------- seccion 7: correccion por falsos faltantes
    print("\n=== 7. CORRECCION POR TASA DIFERENCIAL DE FALSOS FALTANTES ===")
    rr, nr = VALIDACION["rural"]
    ru, nu = VALIDACION["urban"]
    fr, fu = rr / nr, ru / nu
    for lab, k_, n_ in [("rural", rr, nr), ("urbano", ru, nu)]:
        lo, hi = clopper(k_, n_)
        print(f"  falsos faltantes {lab:<7} {k_}/{n_} = {k_/n_*100:4.1f}%  IC95% exacto "
              f"[{lo*100:.1f},{hi*100:.1f}]")
    dep_r = (out.rural & out["faltante_c"]).sum()
    dep_u = ((~out.rural) & out["faltante_c"]).sum()
    n_r, n_u = out.rural.sum(), (~out.rural).sum()
    print(f"  tasa bruta  rural={dep_r/n_r*100:.1f}%  urbano={dep_u/n_u*100:.1f}%  "
          f"OR_crudo={(dep_r/n_r)/(dep_u/n_u):.2f}")
    cr_r, cr_u = dep_r * (1 - fr), dep_u * (1 - fu)
    print(f"  tasa corregida rural={cr_r/n_r*100:.1f}%  urbano={cr_u/n_u*100:.1f}%  "
          f"OR_crudo_corregido={(cr_r/n_r)/(cr_u/n_u):.2f}")
    print(f"  faltantes corregidos: {cr_r:.0f} rurales + {cr_u:.0f} urbanos = "
          f"{cr_r+cr_u:.0f} de {len(out)} ({(cr_r+cr_u)/len(out)*100:.1f}%)")
    # MH con correccion proporcional dentro de cada condado
    rows = []
    for s, g in out.groupby("estrato"):
        a = int((g.rural & g["faltante_c"]).sum())
        b = int((g.rural & ~g["faltante_c"]).sum())
        c = int((~g.rural & g["faltante_c"]).sum())
        e = int((~g.rural & ~g["faltante_c"]).sum())
        rows.append(dict(a=a * (1 - fr), b=b + a * fr, c=c * (1 - fu), e=e + c * fu))
    rc = mh_stats(pd.DataFrame(rows), seed=42)
    res["rural_corr"] = rc
    fmt_mh("rural, corregido por falsos falt.", rc)

    # ---------------- seccion 8: tabla de 15
    print("\n=== 8. TABLA: 15 CENTROS VERIFICADOS COMO FALTANTES ===")
    pool = pd.concat([muestras[True], muestras[False]])
    ver = pool[(~pool["any50"]) & (pool["sim_max"] < 0.5)].copy()
    ver["orden"] = (~ver["rural"]).astype(int) * 1e9 - ver["camas_nh"].fillna(0) \
        - ver["estaciones_dx"].fillna(0)
    ver = ver.sort_values("orden").head(15)
    cols15 = ["nombre", "tracto", "tracto_nombre", "condado_nombre", "estado", "tipo",
              "camas_nh", "estaciones_dx", "residentes_dia_nh", "pacientes_dx",
              "poi_cercano", "poi_cercano_dist", "ciudad"]
    for i, (_, r_) in enumerate(ver.iterrows(), 1):
        carga = (f"{r_['camas_nh']:.0f} beds" if r_["tipo"] == "nursing_home"
                 else f"{r_['estaciones_dx']:.0f} stations" if r_["tipo"] == "dialysis" else "clinic")
        print(f"  {i:2d}. {r_['nombre'][:40]:<42}{r_['ciudad'][:14]:<15}{r_['estado']} "
              f"{r_['tipo']:<13} {carga:<13} tract {r_['tracto']} ({r_['tracto_nombre']}) "
              f"{r_['condado_nombre'][:14]:<15} nearest={r_['poi_cercano']} "
              f"[{r_['poi_cat']}] {r_['poi_cercano_dist']:.0f}m")

    # ---------------- seccion 9: top por carga
    print("\n=== 9. TOP 20 FALTANTES POR CARGA (unidades separadas) ===")
    m2 = m.copy()
    m2["carga"] = m2["camas_nh"].fillna(0) + m2["estaciones_dx"].fillna(0)
    for _, r_ in m2.sort_values("carga", ascending=False).head(20).iterrows():
        u = (f"{r_['camas_nh']:.0f} beds" if r_["tipo"] == "nursing_home"
             else f"{r_['estaciones_dx']:.0f} stations" if r_["tipo"] == "dialysis" else "clinic")
        print(f"  {r_['nombre'][:44]:<46}{r_['ciudad'][:16]:<17}{r_['estado']} {r_['tipo']:<13} "
              f"{u:<13} tract {r_['tracto']} urb={r_['pct_urban']:.2f} svi={r_['svi_overall']:.2f}")

    # ---------------- seccion 10: veredicto
    print("\n=== 10. VEREDICTO ===")
    def sig(k):
        r_ = res.get(k)
        return bool(r_ and np.isfinite(r_.get("or_mh", np.nan)) and (r_["lo"] > 1 or r_["hi"] < 1))
    print(f"  rural sin ajuste        : {'SIRVE' if sig('rural') else 'no sostiene'}")
    print(f"  rural ajustado densidad : {'SIRVE' if sig('rural_dens') else 'no sostiene'}")
    print(f"  rural corregido falsos  : {'SIRVE' if sig('rural_corr') else 'no sostiene'}")
    print(f"  tribal                  : {'SIRVE' if sig('tribal') else 'no sostiene'}")
    print(f"  SVI alto vs bajo        : {'SIRVE' if sig('svi') else 'no sostiene'}")
    for k in ["nursing_home", "dialysis", "health_center"]:
        print(f"  rural | {k:<14} {'SIRVE' if sig('rural_'+k) else 'no sostiene'}")

    # ---------------- escrituras
    out["tamano"] = out["residentes_dia_nh"].fillna(out["pacientes_dx"])
    cols = ["id", "nombre", "direccion", "ciudad", "estado", "condado_nombre", "region",
            "tracto", "tracto_nombre", "condado", "tipo", "encontrado", "encontrado_estricto",
            "encontrado_laxo", "faltante", "faltante_c", "nombre_lejos", "dist_lejos",
            "poi_cercano", "poi_cat", "poi_cercano_dist",
            "poi_cuidado", "poi_cuidado_dist", "any50", "dens_1km", "camas_nh", "estaciones_dx",
            "residentes_dia_nh", "pacientes_dx", "fecha", "pct_urban", "svi_overall",
            "tribal_pct", "aiannh_name", "rural", "tribal", "svi_alto", "fuente"]
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    out[cols].sort_values(["tipo", "faltante", "nombre"], ascending=[True, False, True]) \
        .round({"poi_cercano_dist": 1, "poi_cuidado_dist": 1, "pct_urban": 3,
                "svi_overall": 3, "tribal_pct": 3}) \
        .to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"\n[out] {CSV_OUT} filas={len(out)}")

    write_tex_missing(ver)
    write_tex_rates(out, res)
    write_fig(out)
    print(f"[out] {os.path.join(TEX_TAB, 'care_missing.tex')}")
    print(f"[out] {os.path.join(TEX_TAB, 'care_rates.tex')}")
    print(f"[out] {os.path.join(TEX_FIG, 'care_rates.pdf')}")


# ---------------------------------------------------------------- salidas
def _esc(s):
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"),
                 ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def write_tex_missing(ver):
    lines = [
        "% Generated by procesos/bias_discovery/l7_cuidado.py",
        "\\begin{tabular}{@{}p{4.1cm}p{3.0cm}p{1.9cm}lp{3.4cm}rrrr@{}}",
        "\\toprule",
        "Center & Census tract & County & Type & Nearest Overture POI & Dist. & Beds & Stat. & Res. \\\\",
        " & & & & & (m) & & & /day \\\\",
        "\\midrule",
    ]
    for _, r in ver.iterrows():
        beds = f"{r['camas_nh']:.0f}" if pd.notna(r["camas_nh"]) else "--"
        stat = f"{r['estaciones_dx']:.0f}" if pd.notna(r["estaciones_dx"]) else "--"
        pts = (f"{r['residentes_dia_nh']:.0f}" if pd.notna(r["residentes_dia_nh"])
               else f"{r['pacientes_dx']:.0f}" if pd.notna(r["pacientes_dx"]) else "--")
        tract = f"{r['tracto']} ({r['tracto_nombre']})"
        poi = f"{r['poi_cercano']} [{str(r['poi_cat']).replace('_', ' ')}]"
        lines.append(
            f"{_esc(r['nombre'])} & {_esc(tract)} & {_esc(str(r['condado_nombre']).replace(' County', ''))} "
            f"({r['estado']}) & {_esc(r['tipo'].replace('_', ' '))} & {_esc(poi)} & "
            f"{r['poi_cercano_dist']:.0f} & {beds} & {stat} & {pts} \\\\")
    lines += [
        "\\bottomrule",
        "\\multicolumn{9}{@{}p{13.5cm}@{}}{\\footnotesize Beds: Medicare/Medicaid-certified "
        "nursing-home beds (CMS 4pq5-n9py). Stat.: dialysis stations (CMS 23ew-n7w9). "
        "Res./day: average residents per day for nursing homes, one-year Medicare dialysis "
        "patients for dialysis units. Units are not summed.} \\\\",
        "\\end{tabular}", ""]
    with open(os.path.join(TEX_TAB, "care_missing.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_tex_rates(out, res):
    def cell(g, col):
        k, n = int(g[col].sum()), len(g)
        if n == 0:
            return None
        lo, hi = wilson(k, n)
        return k / n * 100, lo * 100, hi * 100, n

    def row(lab, g, indent=True):
        pre = "\\quad " if indent else ""
        a, c = cell(g, "faltante"), cell(g, "faltante_c")
        if a is None:
            return f"{pre}{_esc(lab)} & 0 & -- & -- & -- \\\\"
        return (f"{pre}{_esc(lab)} & {a[3]} & {a[0]:.1f} & {c[0]:.1f} & "
                f"[{c[1]:.1f}, {c[2]:.1f}] \\\\")

    lines = ["% Generated by procesos/bias_discovery/l7_cuidado.py",
             "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
             "Stratum & $n$ & 1-km rule & Name rule & 95\\% CI (name rule) \\\\", "\\midrule"]
    lines.append("\\multicolumn{5}{@{}l}{\\emph{By centre type}} \\\\")
    for k, g in out.groupby("tipo"):
        lines.append(row(k.replace("_", " "), g))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{By region}} \\\\")
    for r_ in REGIONS:
        lines.append(row(r_, out[out["region"] == r_]))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Rural vs urban, overall and by type}} \\\\")
    for lab, g in [("all centers", out)] + [(k.replace("_", " "), g) for k, g in out.groupby("tipo")]:
        for ru in [True, False]:
            lines.append(row(f"{lab} --- {'rural' if ru else 'urban'}", g[g["rural"] == ru]))
    lines.append("\\midrule")
    lines.append("\\multicolumn{5}{@{}l}{\\emph{Rural vs urban Mantel--Haenszel odds ratio}} \\\\")
    lines.append("\quad \emph{Stratum} & $n$ & \multicolumn{2}{r}{\emph{Bootstrap CI}} & \emph{OR [MH 95\% CI]} \\\\")
    for lab, key in [("overall", "rural"), ("overall, name rule", "rural_c"),
                     ("nursing home", "rural_nursing_home"),
                     ("dialysis", "rural_dialysis"), ("health center", "rural_health_center"),
                     ("nursing home, name rule", "ruralc_nursing_home"),
                     ("dialysis, name rule", "ruralc_dialysis"),
                     ("health center, name rule", "ruralc_health_center"),
                     ("eastern-ok, name rule", "ruralc_eastern-ok"),
                     ("south-central-tx, name rule", "ruralc_south-central-tx"),
                     ("maricopa-az, name rule", "ruralc_maricopa-az"),
                     ("northern-ca, name rule", "ruralc_northern-ca"),
                     ("eastern-wa, name rule", "ruralc_eastern-wa"),
                     ("density-adjusted", "rural_dens"),
                     ("false-negative corrected", "rural_corr")]:
        r = res.get(key)
        if not r or not np.isfinite(r.get("or_mh", np.nan)):
            lines.append(f"\\quad {_esc(lab)} & -- & \\multicolumn{{2}}{{r}}{{not identified}} & -- \\\\")
            continue
        boot = f"[{r['boot'][0]:.2f}, {r['boot'][1]:.2f}]" if r.get("boot") else "--"
        lines.append(f"\\quad {_esc(lab)} & {r['n']} & \\multicolumn{{2}}{{r}}{{{boot}}} & "
                     f"{r['or_mh']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] \\\\")
    lines += ["\\bottomrule",
              "\\multicolumn{5}{@{}p{13.5cm}@{}}{\\footnotesize Dialysis rests on very few rural cases, so its intervals are unstable and are reported as computed.} \\\\",
              "\\end{tabular}", ""]
    with open(os.path.join(TEX_TAB, "care_rates.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    labs, vals, errs, ns = [], [], [], []
    for r_ in REGIONS:
        g = out[out["region"] == r_]
        k, n = int(g["faltante_c"].sum()), len(g)
        lo, hi = wilson(k, n)
        labs.append(r_.replace("-", "-\n"))
        vals.append(k / n * 100)
        errs.append([(k / n - lo) * 100, (hi - k / n) * 100])
        ns.append(n)
    ax.bar(range(len(labs)), vals, yerr=np.array(errs).T, capsize=3, color="#4477aa",
           error_kw={"lw": 1})
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels([f"{a}\n(n={b})" for a, b in zip(labs, ns)], fontsize=7)
    ax.set_ylabel("Care facilities absent from Overture (%)")
    ax.set_title("By region", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.set_xlabel("")

    ax = axes[1]
    labs, vals, errs, ns = [], [], [], []
    for k_, g in [("All", out)] + [(t, out[out["tipo"] == t])
                                   for t in ["nursing_home", "dialysis", "health_center"]]:
        for ru in [True, False]:
            sub = g[g["rural"] == ru]
            kk, n = int(sub["faltante_c"].sum()), len(sub)
            if n == 0:
                continue
            lo, hi = wilson(kk, n)
            labs.append(f"{k_.replace('_', ' ')}\n{'rural' if ru else 'urban'} (n={n})")
            vals.append(kk / n * 100)
            errs.append([(kk / n - lo) * 100, (hi - kk / n) * 100])
    ax.bar(range(len(labs)), vals, yerr=np.array(errs).T, capsize=2, lw=1,
           color=["#cc6677" if i % 2 == 0 else "#88aacc" for i in range(len(vals))],
           error_kw={"lw": 0.8})
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=5.5, rotation=90)
    ax.set_title("Rural vs urban, by type", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.45)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(TEX_FIG, "care_rates.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
