"""Referencia externa del diagnostico de hospitales (l9_hospitales.py): CMS, VA, IHS y
las salas de urgencias autonomas, geocodificados y clasificados como hospital
con o sin urgencias.

Se separo de l9_hospitales.py para respetar el limite de 1,000 lineas por archivo. No hay
logica de analisis aca, solo construccion y cache de la referencia.
"""

import json
import os
import re
import subprocess

import numpy as np
import pandas as pd

import l7_cuidado as L7

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

EXT_CMS = os.path.join(ROOT, "datos", "externos", "cms")            # consultado 2026-10-30
EXT_VA = os.path.join(ROOT, "datos", "externos", "va")              # consultado 2026-10-30
EXT_IHS = os.path.join(ROOT, "datos", "externos", "ihs")            # consultado 2026-10-30
EXT_FSED = os.path.join(ROOT, "datos", "externos", "fsed")          # consultado 2026-10-31
EXT_CEN = os.path.join(ROOT, "datos", "externos", "census_geo")     # consultado 2026-10-30

STATES = {"AZ", "CA", "OK", "TX", "WA"}
# La referencia mira tambien los estados vecinos: un hospital de NM, NV, UT, CO, OR, ID, AR,
# MO, KS o LA puede ser el mas cercano de un tracto del borde. Los POIs siguen siendo solo
# los de las 5 regiones.
STATES_REF = STATES | {"NM", "NV", "UT", "CO", "OR", "ID", "AR", "MO", "KS", "LA"}
HOSP_CATS = ("hospital",)          # categoria evaluada
POOL_CATS = ("hospital", "medical_center", "childrens_hospital")  # sensibilidad: el pool que un usuario saca

GEO_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
COUNTY_URL = "https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"


# Un hospital de agudos cuyo campo dice "No" puede ser una especialidad sin urgencias
# (cirugia, corazon, ortopedia, rehab) o un error de reporte. Se separa por nombre.
_ESPECIALIDAD = re.compile(
    r"SURGICAL|SURGERY|HEART|CARDIOVASC|ORTHOPED|\bORTHO\b|SPINE|REHAB|PSYCH|BEHAVIOR|"
    r"LONG TERM|LTAC|\bEYE\b|CANCER|WOMEN|WOMAN|SPECIALTY|SPECIALITIES|PHYSICIAN|SELECT |"
    r"KINDRED|ENCOMPASS|\bPAIN\b|SPORTS|SKIN|DIGESTIVE|KIDNEY|DIALYSIS|WEIGHT|FERTILITY|"
    r"VEIN|SLEEP|WOUND|AMBULAT|OUTPATIENT|INFUSION|ENDOSCOP|\bNURSING\b|HOSPICE")
_TIPOS_AGUDOS = ("Acute Care Hospitals", "Critical Access Hospitals", "Rural Emergency Hospital",
                 "Acute Care - Veterans Administration", "Acute Care - Department of Defense")


def er_de_cms(hospital_type, flag, nombre):
    """(real con urgencias, cota inferior, motivo) para una fila de CMS.

    Base: flag Yes; o CAH (42 CFR 485.618 exige urgencias 24 h), REH (42 CFR 485.502 da
    servicios de urgencias) o agudo de VA/DoD. Para 'Acute Care' con flag No se mira el
    nombre: especialidad -> sin urgencias; resto -> con urgencias.
    Cota inferior: cualquier agudo, CAH, REH, VA o DoD, sin mirar el flag.
    """
    ac = hospital_type in _TIPOS_AGUDOS
    lb = ac
    if flag:
        return True, lb, "flag Yes"
    if hospital_type in ("Critical Access Hospitals", "Rural Emergency Hospital"):
        return True, lb, "normativa (CAH/REH)"
    if hospital_type in ("Acute Care - Veterans Administration",
                         "Acute Care - Department of Defense"):
        return True, lb, "agudo VA/DoD"
    if hospital_type == "Acute Care Hospitals":
        if _ESPECIALIDAD.search(str(nombre).upper()):
            return False, lb, "agudo especialidad, flag No"
        return True, lb, "agudo general con flag No"
    return False, lb, "otro tipo, flag No"

# ---------------------------------------------------------------- fuentes externas
def load_cms():
    rows = []
    for f in sorted(os.listdir(EXT_CMS)):
        if f.startswith("hosp_") and f.endswith(".json"):
            with open(os.path.join(EXT_CMS, f), encoding="utf-8") as fh:
                rows += json.load(fh)["results"]
    d = pd.DataFrame(rows).drop_duplicates("facility_id")
    d = d[d["state"].isin(STATES_REF)].copy()
    d["zip5"] = d["zip_code"].astype(str).str[:5]
    d["ruc"] = d["countyparish"].str.upper()
    d["er_flag"] = d["emergency_services"].eq("Yes")
    return d


def ref_all(cms):
    """Todas las instituciones de referencia, con o sin coordenada (nombre + estado)."""
    a = pd.DataFrame({"nombre": cms["facility_name"].astype(str), "estado": cms["state"],
                      "er": cms["er"], "geo": cms["lat"].notna()})
    others = []
    for fn in (load_va, load_ihs, load_fser):
        v = fn()
        others.append(pd.DataFrame({"nombre": v["nombre"], "estado": v["estado"],
                                    "er": v["er"], "geo": True}))
    a = pd.concat([a] + others, ignore_index=True)
    a["key"] = a["nombre"].map(L7.key_of)
    return a


def place_centroids():
    """Centroide del lugar censal (Census Gazetteer 2023), para hospitales CMS sin geocodificar."""
    p = os.path.join(EXT_CEN, "2023_Gaz_place_national.txt")
    if not os.path.exists(p):
        return {}
    g = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, encoding="latin-1")
    g.columns = [c.strip() for c in g.columns]
    g = g[g["USPS"].isin(STATES)].copy()
    suf = (r" (CITY|TOWN|VILLAGE|CDP|BOROUGH|TOWNSHIP|COUNTY|MUNICIPALITY|URBAN COUNTY|"
           r"CITY AND BOROUGH|UNIFIED GOVERNMENT|METROPOLITAN GOVERNMENT|COMUNIDAD|ZONA URBANA)$")
    g["k"] = g["NAME"].str.upper().str.replace(suf, "", regex=True).str.strip()
    g["lat"] = pd.to_numeric(g["INTPTLAT"], errors="coerce")
    g["lon"] = pd.to_numeric(g["INTPTLONG"], errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).drop_duplicates(["USPS", "k"])
    return {(r.USPS, r.k): (r.lat, r.lon) for r in g.itertuples()}


def geocode_cms(d):
    """Census Geocoder batch (misma logica que l7_cuidado.py). Cachea la respuesta y solo
    pide al servicio los ids que no estan en la cache: la de 5 estados se conserva y los
    10 estados vecinos se agregan en un archivo aparte."""
    names = ["facility_id", "addr_in", "match", "type", "addr_out", "xy", "tiger", "side"]
    parts = []
    for f in ("hosp_geo_out.csv", "hosp_geo_out_extra.csv"):
        p = os.path.join(EXT_CMS, f)
        if os.path.exists(p):
            g = pd.read_csv(p, header=None, dtype=str, keep_default_na=False, names=names)
            if len(g):
                parts.append(g)
    got = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)
    got = got.drop_duplicates("facility_id")
    falta = d[~d["facility_id"].isin(got["facility_id"])]
    if len(falta):
        inp = os.path.join(EXT_CMS, "hosp_geo_in_extra.csv")
        falta[["facility_id", "address", "citytown", "state", "zip5"]].to_csv(
            inp, index=False, header=False, encoding="utf-8")
        r = subprocess.run(["curl", "-s", "--max-time", "600", "-X", "POST",
                            "-F", f"addressFile=@{inp}", "-F", "benchmark=Public_AR_Current",
                            "-F", "vintage=Current_Current", GEO_URL],
                           capture_output=True, text=True)
        if r.returncode != 0 or len(r.stdout) < 100:
            raise RuntimeError(f"Census Geocoder fallo: {r.returncode} {r.stderr[:300]}")
        with open(os.path.join(EXT_CMS, "hosp_geo_out_extra.csv"), "w", encoding="utf-8") as fh:
            fh.write(r.stdout)
        print(f"[geo] geocodificados {len(falta)} hospitales nuevos de los estados vecinos")
        got = pd.concat([got, pd.read_csv(os.path.join(EXT_CMS, "hosp_geo_out_extra.csv"),
                                          header=None, dtype=str, keep_default_na=False,
                                          names=names)], ignore_index=True)
    g = got[got["match"].eq("Match")].copy()
    g["lon"] = pd.to_numeric(g["xy"].str.split(",").str[0], errors="coerce")
    g["lat"] = pd.to_numeric(g["xy"].str.split(",").str[1], errors="coerce")
    g = g.dropna(subset=["lat", "lon"]).drop_duplicates("facility_id")
    c = d.merge(g[["facility_id", "lat", "lon", "type"]], on="facility_id", how="left")
    c["geo_prec"] = np.where(c["lat"].notna(), "direccion", "")
    print(f"[geo] CMS: {c['lat'].notna().sum()}/{len(c)} geocodificados "
          f"({100*c['lat'].notna().mean():.1f}%); {c['lat'].isna().sum()} sin direccion")
    # Respaldo: dirección de CMS en Nominatim (l9_geo_respaldo.py). Lo que no resuelve cae al centroide de la ciudad.
    pn = os.path.join(EXT_CMS, "hosp_geo_nominatim.csv")
    if os.path.exists(pn):
        n = pd.read_csv(pn, dtype={"facility_id": str}).dropna(subset=["lat", "lon"]).drop_duplicates("facility_id").set_index("facility_id")
        ok = c["facility_id"].isin(n.index) & c["lat"].isna()
        c.loc[ok, "lat"] = c.loc[ok, "facility_id"].map(n["lat"])
        c.loc[ok, "lon"] = c.loc[ok, "facility_id"].map(n["lon"])
        c.loc[ok, "geo_prec"] = "nominatim"
        print(f"[geo]   {int(ok.sum())} resueltos con la dirección de CMS en Nominatim")
    pc = place_centroids()
    fill = 0
    for i in c.index[c["lat"].isna()]:
        k = (c.at[i, "state"], str(c.at[i, "citytown"]).upper().strip())
        if k in pc:
            c.at[i, "lat"], c.at[i, "lon"] = pc[k]
            c.at[i, "geo_prec"] = "centroide_lugar"
            fill += 1
    print(f"[geo]   {fill} de {c['lat'].isna().sum() + fill} colocados en el centroide de su ciudad "
          f"(error tipico 1-3 km); quedan sin coordenada: {c['lat'].isna().sum()}")
    return c


def load_va():
    r = []
    for f in ("vha_5states.geojson", "vha_extra.geojson"):
        p = os.path.join(EXT_VA, f)
        if os.path.exists(p):
            r += json.load(open(p, encoding="utf-8"))["features"]
    r = pd.DataFrame([f["properties"] for f in r])
    r = r[r["S_STATE"].isin(STATES_REF)].drop_duplicates("STA_NO")
    r["nombre"] = r["STA_NAME"].astype(str)
    # Solo el VAMC tiene sala de urgencias; los Vet Center y los CBOC no.
    r["er"] = r["nombre"].str.upper().str.contains("MEDICAL CENTER")
    r["lat"] = pd.to_numeric(r["LAT"], errors="coerce")
    r["lon"] = pd.to_numeric(r["LON"], errors="coerce")
    r["tipo"] = "VA"
    r["ciudad"] = r["S_CITY"]
    r["estado"] = r["S_STATE"]
    return r[["nombre", "lat", "lon", "er", "tipo", "ciudad", "estado"]].dropna(subset=["lat", "lon"])


def load_ihs():
    p = os.path.join(EXT_IHS, "ihs_all.geojson")
    d = json.load(open(p, encoding="utf-8"))["features"]
    r = pd.DataFrame([f["properties"] for f in d])
    r = r[r["STATE"].isin(STATES_REF)].copy()
    r["nombre"] = r["FACILITY_NAME"].astype(str)
    # ER_FLAG=1 sin camas es un centro ambulatorio mal marcado (varias clinicas de Alaska y
    # Utah): solo cuenta si tiene camas o si el nombre dice emergencia.
    camas = pd.to_numeric(r["BEDS"], errors="coerce").fillna(0)
    r["er"] = r["ER_FLAG"].astype(str).eq("1") & (
        (camas > 0) | r["FACILITY_NAME"].str.upper().str.contains("EMERGENCY", na=False))
    r["lat"] = pd.to_numeric(r["LATITUDE"], errors="coerce")
    r["lon"] = pd.to_numeric(r["LONGITUDE"], errors="coerce")
    r["tipo"] = "IHS"
    r["ciudad"] = r["CITY"]
    r["estado"] = r["STATE"]
    return r[["nombre", "lat", "lon", "er", "tipo", "ciudad", "estado"]].dropna(subset=["lat", "lon"])


def load_fser():
    """Salas de urgencias autónomas y hospitales con urgencias que CMS no lista (consulta 2026-10-31):
    roster de licencias de Texas HHSC (corte 2026-10-21), informe AEMS de Arizona central, operadores
    y directorio OSDH. Fuente y dirección por fila en datos/externos/fsed/fsed_referencia.csv."""
    r = pd.read_csv(os.path.join(EXT_FSED, "fsed_referencia.csv"), dtype={"zip": str})
    r["er"] = True
    r["tipo"] = "FSED:" + r["tipo_fuente"].astype(str)
    return r[["nombre", "lat", "lon", "er", "tipo", "ciudad", "estado"]].dropna(subset=["lat", "lon"])


# Un hospital de agudos cuyo campo dice "No" puede ser una especialidad sin urgencias
# (cirugia, corazon, ortopedia, rehab) o un error de reporte. Se separa por nombre.
_ESPECIALIDAD = re.compile(
    r"SURGICAL|SURGERY|HEART|CARDIOVASC|ORTHOPED|\bORTHO\b|SPINE|REHAB|PSYCH|BEHAVIOR|"
    r"LONG TERM|LTAC|\bEYE\b|CANCER|WOMEN|WOMAN|SPECIALTY|SPECIALITIES|PHYSICIAN|SELECT |"
    r"KINDRED|ENCOMPASS|\bPAIN\b|SPORTS|SKIN|DIGESTIVE|KIDNEY|DIALYSIS|WEIGHT|FERTILITY|"
    r"VEIN|SLEEP|WOUND|AMBULAT|OUTPATIENT|INFUSION|ENDOSCOP|\bNURSING\b|HOSPICE")
_TIPOS_AGUDOS = ("Acute Care Hospitals", "Critical Access Hospitals", "Rural Emergency Hospital",
                 "Acute Care - Veterans Administration", "Acute Care - Department of Defense")


def er_de_cms(hospital_type, flag, nombre):
    """(real con urgencias, cota inferior, motivo) para una fila de CMS.

    Base: flag Yes; o CAH (42 CFR 485.618 exige urgencias 24 h), REH (42 CFR 485.502 da
    servicios de urgencias) o agudo de VA/DoD. Para 'Acute Care' con flag No se mira el
    nombre: especialidad -> sin urgencias; resto -> con urgencias.
    Cota inferior: cualquier agudo, CAH, REH, VA o DoD, sin mirar el flag.
    """
    ac = hospital_type in _TIPOS_AGUDOS
    lb = ac
    if flag:
        return True, lb, "flag Yes"
    if hospital_type in ("Critical Access Hospitals", "Rural Emergency Hospital"):
        return True, lb, "normativa (CAH/REH)"
    if hospital_type in ("Acute Care - Veterans Administration",
                         "Acute Care - Department of Defense"):
        return True, lb, "agudo VA/DoD"
    if hospital_type == "Acute Care Hospitals":
        if _ESPECIALIDAD.search(str(nombre).upper()):
            return False, lb, "agudo especialidad, flag No"
        return True, lb, "agudo general con flag No"
    return False, lb, "otro tipo, flag No"


def build_ref():
    cms = load_cms()
    c = geocode_cms(cms)
    c["nombre"] = c["facility_name"].astype(str)
    tri = [er_de_cms(t, f, n) for t, f, n in zip(c["hospital_type"], c["er_flag"], c["nombre"])]
    c["er"] = [t[0] for t in tri]
    c["er_lb"] = [t[1] for t in tri]
    c["motivo"] = [t[2] for t in tri]
    miss = c[c["lat"].isna()]
    print(f"[geo] CMS sin geocodificar: {len(miss)} de {len(c)}; de ellos con urgencias (base) "
          f"{int(miss['er'].sum())} de {int(c['er'].sum())}")
    an = c[(c["hospital_type"] == "Acute Care Hospitals") & (~c["er_flag"])]
    esp = an[an["er"] == False]  # noqa: E712
    print(f"[er] Acute Care con flag No: {len(an)}; por nombre pasan a urgencias "
          f"{int(an['er'].sum())}, quedan sin urgencias {len(esp)} "
          f"(especialidad: {', '.join(sorted(esp['nombre'].str[:22].head(6)))})")
    c = c.dropna(subset=["lat", "lon"]).copy()
    c["tipo"] = "CMS:" + c["hospital_type"].str.replace(" - ", "/", regex=False)
    c["ciudad"] = c["citytown"]
    c["cms_id"] = c["facility_id"]
    c["condado_nombre"] = c["countyparish"]
    c["estado"] = c["state"]
    fser = load_fser().assign(cms_id="", condado_nombre="", er_lb=True, motivo="sala autonoma o hospital no CMS")
    va = load_va().assign(cms_id="", condado_nombre="", er_lb=load_va()["er"].to_numpy(),
                          motivo="VAMC")
    ihs = load_ihs().assign(cms_id="", condado_nombre="",
                            er_lb=load_ihs()["er"].to_numpy(), motivo="IHS ER")
    cols = ["nombre", "lat", "lon", "er", "er_lb", "tipo", "ciudad", "cms_id",
            "condado_nombre", "estado", "motivo"]
    ref = pd.concat([c[cols], va[cols], ihs[cols], fser[cols]], ignore_index=True)
    ref["key"] = ref["nombre"].map(L7.key_of)
    ref["ini"] = ref["nombre"].map(_initials)
    return ref, ref_all(c)


def _initials(name):
    toks = [t for t in L7.key_of(name).split() if len(t) > 2]
    return "".join(t[0] for t in toks)


def load_county_names():
    p = os.path.join(EXT_CEN, "national_county.txt")
    if not os.path.exists(p):
        os.makedirs(EXT_CEN, exist_ok=True)
        r = subprocess.run(["curl", "-s", "--max-time", "60", COUNTY_URL, "-o", p])
        if r.returncode != 0:
            return {}
    d = pd.read_csv(p, header=None, dtype=str, keep_default_na=False,
                    names=["USPS", "STATEFP", "COUNTYFP", "NAME", "CLASS"])
    return dict(zip(d.STATEFP + d.COUNTYFP, d.NAME.str.replace(" County", "", regex=False).str.title()))


