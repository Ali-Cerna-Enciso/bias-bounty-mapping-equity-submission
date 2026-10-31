"""Best Bias Discovery: el POI 'hospital' de Overture no es un hospital con urgencias.

Un score de brecha solo ve faltantes; es ciego a los falsos positivos. Este script mide
cuantos POIs de la categoria `hospital` de Overture NO son hospitales con sala de
urgencias, cuanto mas lejos queda el hospital real mas cercano y si eso pesa mas en
tractos rurales, tribales o de SVI alto dentro del mismo condado.

Referencias externas (solo diagnostico del premio, nunca el CSV puntuado):
  CMS Hospital General Information  data.cms.gov/provider-data/dataset/xubh-q36u
      (5,419 filas; tipo de hospital y emergency_services Si/No; sin lat/lon: se geocodifica
       con el Census Geocoder, misma logica que l7_cuidado.py)
  VA  VHA Facilities                services1.arcgis.com/smmmD7AGkh7eJR2a (servicio oficial VA)
      -> se toma como hospital con urgencias el VAMC ("... VA MEDICAL CENTER")
  IHS ITU Health Facilities         services2.arcgis.com/VFLAJVozK0rtzQmT (servicio oficial IHS)
      -> ER_FLAG = 1 y camas > 0
  TX DSHS Freestanding Emergency    services.arcgis.com/su8ic9KbA7PYVxPS (corte 2020-07-01, 210)
      -> por licencia operan como urgencias 24/7: cuentan como reales con urgencias
  Estados vecinos: la referencia suma NM, NV, UT, CO, OR, ID, AR, MO, KS y LA; los POIs no.

Un hospital es "real con urgencias" si el flag de CMS dice Yes, o si es Critical Access
(42 CFR 485.618 exige urgencias 24 h) o Rural Emergency (42 CFR 485.502), o si es agudo de
VA/DoD. Un Acute Care con flag No entra por nombre: especialidad (cirugia, corazon, ortopedia,
rehab) queda sin urgencias, el resto cuenta con urgencias. La cota inferior no mira el flag:
todo agudo, CAH, REH, VA, DoD y FSER con licencia.

El calculo esta aca; la referencia en l9_referencia.py y las tablas y la figura en
l9_salidas.py (limite de 1,000 lineas por archivo).
  Census county FIPS names          www2.census.gov/geo/docs/reference/codes/files/national_county.txt

Entradas internas: Overture POIs y census-tracts de las 5 regiones + strata nacional.
Salidas: l9_pois.csv, l9_tractos.csv y, si el hallazgo sirve, tablas/figura del writeup.

Distancias en linea recta (centroide del tracto al POI), NO tiempo de viaje.

Corre:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe procesos/bias_discovery/l9_hospitales.py
"""

import json
import math
import os
import re
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l7_cuidado as L7  # noqa: E402  (reuso: normalizacion de nombres, Wilson, MH, bootstrap)
import l9_referencia as REF  # noqa: E402  (CMS/VA/IHS/FSER y geocodificacion)
import l9_salidas as SAL  # noqa: E402  (tablas LaTeX y figura)
from l9_referencia import (HOSP_CATS, POOL_CATS, build_ref, er_de_cms,  # noqa: E402,F401
                           load_county_names, load_fser, place_centroids, ref_all)

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

EXT_CEN = os.path.join(ROOT, "datos", "externos", "census_geo")   # consultado 2026-10-30
CSV_POI = os.path.join(ROOT, "datos", "bias_discovery", "l9_pois.csv")
CSV_TRACT = os.path.join(ROOT, "datos", "bias_discovery", "l9_tractos.csv")
TEX_TAB = os.path.join(ROOT, "docs", "2_premios", "writeup", "tablas")
TEX_FIG = os.path.join(ROOT, "docs", "2_premios", "writeup", "figuras")

REGIONS = ["maricopa-az", "northern-ca", "eastern-ok", "south-central-tx", "eastern-wa"]

MATCH_KM = 2.0     # ventana de nombre para llamar "mismo hospital" (campus grande)
NAME_SIM = 0.6     # umbral difflib, igual que l7
REF_DATE = "2026-10-30"

# Errores de clasificacion medidos sobre las muestras de la seccion 7
# (100 falsos + 100 reales + 100 nombre+estado, semilla 42). (falso_positivo, falso_negativo)
# por definicion A/B/C: fp = marcados reales que no cumplen; fn = marcados falsos que si.
# A: etiquetas guardadas en datos/bias_discovery/l9_etiquetas_validacion.csv (30/100 y 8/100). El grupo
# emparejado solo por nombre y estado dio fp(A) = 61/100: esa regla sobreestima los "reales", así que el
# 81.1% de tractos con error es conservador. B y C no se reetiquetaron y no se citan en el paper.
VERIF = {"A": (0.30, 0.08), "B": (0.18, 0.14), "C": (0.07, 0.03)}

# Verificacion web (2026-10-31) de las dos tablas de lugares: (a) que es el POI; (b) que no hay un hospital o sala
# autonoma mas cerca que el de la referencia. Detalle y paginas leidas: VERIFICACION en el historial del repo.
VERIF_URB = {
    "48201556000": dict(
        lugar='Hockley area', condado='Harris, TX',
        nota='an HVAC factory (the Daikin Texas Technology Park) in unincorporated Harris County near Waller, https://daikintexastechnologypark.com/page/contact ; ER: Fairfield Emergency Room, Cypress'),
    "48029161902": dict(
        lugar='Luckey Ranch', condado='Bexar, TX',
        nota='a walk-in primary-care clinic in the Luckey Ranch subdivision off US 90, https://www.universityhealth.com/locations/luckey-ranch ; ER: Physicians Premier, Potranco Road'),
    "48329010123": dict(
        lugar='East Midland', condado='Midland, TX',
        nota="an on-site occupational-health clinic at Pioneer Natural Resources' Midland offices, https://medcor.com/clinics/ ; ER: Midland Memorial Hospital"),
    "04023966108": dict(
        lugar='Rio Rico', condado='Santa Cruz, AZ',
        nota='a directory listing in a commercial strip, 54 m from Mariposa Community Health Center, https://mariposachc.net/locations/ ; ER: Holy Cross Hospital, a critical access hospital (CMS Hospital General Information)'),
    "48201222900": dict(
        lugar='Aldine', condado='Harris, TX',
        nota='a Harris Health primary-care clinic on Aldine Mail Route, closed at weekends, https://www.harrishealth.org/locations-hh/Pages/aldine-health-center.aspx ; ER: Memorial Hermann Northeast Hospital'),
    "06067007106": dict(
        lugar='Natomas', condado='Sacramento, CA',
        nota='a medical-device manufacturer in the Natomas industrial area, https://www.vortran.com/ ; ER: Sutter Medical Center'),
}


VERIF_RURAL = {
    "48505950402": dict(
        lugar="Zapata", condado="Zapata, TX",
        nota="Gateway Community Health Center, whose page states ``This is not an emergency clinic. For emergencies, please call 911'', https://gatewaychc.com/zapata-clinic ; ER: STAT Specialty Hospital - Laredo South, https://statspecialtyhospital.com/locations/laredo-south . Zapata County has no CMS hospital and no HHSC-licensed freestanding ER; the commercial Zapata Mercy Minor Emergency Room is a walk-in clinic open limited hours"),
    "48131950200": dict(
        lugar="Freer", condado="Duval, TX",
        nota="an outpatient rural health clinic of CHRISTUS Spohn Alice, closed at weekends, https://www.christushealth.org/locations/freer ; ER: https://www.christushealth.org/locations/spohn-hospital-alice"),
    "48047950100": dict(
        lugar="Falfurrias", condado="Brooks, TX",
        nota="a private clinic with no official page we could verify; Brooks County has no CMS hospital and no HHSC-licensed freestanding ER; ER: https://www.christushealth.org/locations/spohn-hospital-kleberg"),
    "48169950102": dict(
        lugar="Post", condado="Garza, TX",
        nota="a scout camp of the South Plains Council, https://www.spccamps.org/camp-post ; ER: Lynn County Hospital District, a critical access hospital (CMS). Garza County has no hospital and no licensed freestanding ER"),
    "04013723305": dict(
        lugar="Gila Bend", condado="Maricopa, AZ",
        nota="a generic directory listing located in the Love's Travel Stop plaza on I-8 (Overture maps Love's, McDonald's and the Greyhound stop within 25--61 m); the town's real facility is a community health centre without emergency care, https://adelantehealthcare.com/locations/gila-bend/ ; ER: standalone Abrazo Buckeye Emergency Center, https://aems.org/central-region-map/abrazo-buckeye-emergency-center"),
    "04019005200": dict(
        lugar="Ajo", condado="Pima, AZ",
        nota="a dental clinic of the Tohono O'odham health system at a highway junction, outside this tract: the tract itself holds no hospital POI (IHS facility registry); ER: TONHC Sells Hospital, https://tonhc.org/"),
    "04012020102": dict(
        lugar="Salome", condado="La Paz, AZ",
        nota="a family-health clinic run by La Paz Regional Hospital, https://crh.arizona.edu/la-paz-regional-hospital ; ER: Wickenburg Community Hospital, https://wickhosp.com/emergency/ ; the critical access hospitals in Parker are farther"),
    "04017960200": dict(
        lugar="Holbrook", condado="Navajo, AZ",
        nota="a medical office at the address of an internal-medicine group practice (directory listings only); local care is outpatient, https://elkridge.org/location/holbrook ; ER: Little Colorado Medical Center, Winslow, https://lcmcwmh.com/emergency/"),
}


_LABELS = [
    ("veterinario", r"\bANIMAL\b|\bVETERINAR|\bVET\b|\bPET\b|\bEQUINE\b|\bLIVESTOCK\b"),
    ("urgent_care", r"\bURGENT\b|\bMINOR EMERG|\bXPRESS\b|\bMEDEXPRESS\b|\bCARE ?NOW\b|"
                    r"\bAFTER HOURS\b|\bREDICLINIC\b|\bMED ?STOP\b|\bNEXTCARE\b|\bEXPRESS CARE\b"),
    ("imagen_lab", r"\bIMAGING\b|\bRADIOLOG|\bXRAY\b|\bX RAY\b|\bMRI\b|\bCT SCAN\b|\bLABORATOR|\bLAB\b|"
                   r"\bDIAGNOSTIC\b|\bPATHOLOG|\bULTRASOUND\b|\bMAMMOGRAM\b|\bSCREENING\b"),
    ("hospice_home_health", r"\bHOSPICE\b|\bHOME HEALTH\b|\bHOMECARE\b|\bHOME CARE\b|\bPALLIATIVE\b"),
    ("farmacia", r"\bPHARMACY\b|\bDRUG\b|\bPRESCRIPTION\b"),
    ("dental", r"\bDENTAL\b|\bDENTIST|\bORTHODONT|\bSMILE\b"),
    ("conducta_salud_mental", r"\bBEHAVIORAL\b|\bMENTAL\b|\bCOUNSEL|\bPSYCH|\bADDICTION\b|"
                               r"\bRECOVERY\b|\bSOBRIETY\b|\bAUTISM\b|\bTHERAPY\b"),
    ("terapia_fisica", r"\bPHYSICAL THERAP|\bOCCUPATIONAL THERAP|\bREHAB\b|\bCHIROPRACT|\bSPINE\b"),
    ("enfermeria_ltc", r"\bNURSING\b|\bSKILLED\b|\bASSISTED LIVING\b|\bLONG TERM\b|"
                       r"\bTRANSITIONAL\b|\bSENIOR\b|\bRETIREMENT\b|\bGROUP HOME\b|\bMEMORY CARE\b"),
    ("equipo_medico", r"\bSUPPLY\b|\bSUPPLIES\b|\bDURABLE\b|\bEQUIPMENT\b|\bBRACE\b|\bORTHOTIC\b|"
                      r"\bPROSTHET|\bMOBILITY\b|\bSCOOTER\b"),
    ("bienestar_estetica", r"\bWELLNESS\b|\bFITNESS\b|\bGYM\b|\bYOGA\b|\bMASSAGE\b|\bSPA\b|\bSALON\b|"
                           r"\bWEIGHT LOSS\b|\bNUTRITION\b|\bACUPUNCT|\bMED SPA\b"),
    ("administrativo", r"\bBILLING\b|\bINSURANCE\b|\bADMINISTRAT|\bMANAGEMENT\b|\bSTAFFING\b|"
                       r"\bCONSULT|\bRECRUIT|\bFOUNDATION\b|\bAUXILIARY\b|\bTHRIFT\b|\bGIFT SHOP\b"),
    ("educacion", r"\bUNIVERSITY\b|\bCOLLEGE\b|\bSCHOOL\b|\bACADEMY\b|\bINSTITUTE OF\b"),
    ("militar_dod", r"\bARMY\b|\bNAVY\b|\bAIR FORCE\b|\bMILITARY\b|\bBARRACKS\b|\bVETERANS AFFAIRS\b|"
                    r"\bVAMC\b|\bVA \b|\bBASE\b|\bRESERVE\b"),
    ("clinica_consultorio", r"\bCLINIC\b|\bCLÍNICA\b|\bHEALTH CENTER\b|\bHEALTHCARE CENTER\b|"
                            r"\bCOMMUNITY HEALTH\b|\bMEDICAL GROUP\b|\bPHYSICIANS?\b|\bASSOCIATES\b|"
                            r"\bFAMILY (MEDICINE|PRACTICE|CARE)\b|\bINTERNAL MEDICINE\b|\bPEDIATRIC\b|"
                            r"\bOBSTETRIC|\bGYNECOLOG|\bCARDIOLOG|\bONCOLOG|\bORTHOPED|\bSURGERY\b|"
                            r"\bSURGICAL\b|\bUROLOG|\bDERMATOLOG|\bNEUROLOG|\bGASTRO|\bALLERGY\b|"
                            r"\bARTHRITIS\b|\bPAIN\b|\bWOUND CARE\b|\bVASCULAR\b|\bOCULUS\b|"
                            r"\bOPTOMETR|\bOPHTHALM|\bHEARING\b|\bAUDIOLOG|\bPODIATR|\bENDOSCOP|"
                            r"\bINFUSION\b|\bDIALYSIS\b|\bKIDNEY\b|\bFERTILITY\b|\bMEDICAL\b|"
                            r"\bHEALTH\b|\bOUTPATIENT\b|\bSPECIALTY\b|\bSPECIALIST\b|\bCARE\b"),
]
_LABELS = [(k, re.compile(v)) for k, v in _LABELS]

_CAT_EN = {
    "clinica_consultorio": "clinic or practice", "otro_sin_etiqueta": "other, no keyword",
    "urgent_care": "urgent care", "conducta_salud_mental": "behavioural health",
    "imagen_lab": "imaging or lab", "administrativo": "administrative or vendor",
    "bienestar_estetica": "wellness or spa", "educacion": "school or training",
    "hospice_home_health": "hospice or home health", "militar_dod": "military or VA office",
    "terapia_fisica": "physical therapy", "enfermeria_ltc": "nursing or long-term care",
    "farmacia": "pharmacy", "veterinario": "veterinary", "dental": "dental",
    "equipo_medico": "medical equipment",
}


# Anexo: unidad, servicio o consulta dentro de un hospital, no el hospital. Es la causa
# medida del 38% de "reales" que no lo eran en la primera lectura.
_ANEXO = re.compile(
    r"\bCLINIC|\bCENTRO? FOR\b|\bCENTER FOR\b|\bIMAGING|\bRADIOLOG|\bLAB\b|\bLABORATOR|\bOUTPATIENT|"
    r"\bPAVILION|\bPROFESSIONAL|\bPLAZA|\bOFFICE|\bWOUND|\bUNIT\b|\bWARD\b|\bROOM\b|\bPHARMACY|"
    r"\bHOSPICE|\bHOME HEALTH|\bHOMECARE|\bSLEEP|\bURGENT|\bDENTAL|\bVETERINAR|\bGROUP\b|"
    r"\bNETWORK|\bPHYSICIANS?\b|\bASSOCIATES|\bDEPT\b|\bDEPARTMENT|\bCANCER CENTER|\bDIAGNOSTIC|"
    r"\bFOUNDATION|\bBILLING|\bADMINISTRAT|\bSUPPLY|\bSUPPLIES|\bMOBILE\b|\bEXPRESS\b|\bMINOR\b")
_MODALITY = re.compile(
    r"\bREHAB\w*|\bDIALYSIS|\bPSYCHIATRIC|\bBEHAVIORAL|\bSKILLED|\bLONG TERM|\bSUBSTANCE|"
    r"\bSOBRIETY|\bHOSPICE|\bASSISTED")
_LOOKS = re.compile(r"\bHOSPITAL|\bMEDICAL CENTER|\bMED CTR|\bMEDCENTER|\bHEALTH SYSTEM|"
                    r"\bMEDICAL CITY|\bINFIRMARY|\bSANATORIUM")
_MILITARY = re.compile(r"\bARMY\b|\bAIR FORCE\b|\bNAVAL\b|\bNAVY\b|\bMILITARY\b|\bDARNALL\b|"
                       r"\bCAVAZOS\b|\bBARRACKS\b|\bBASE\b|\bAFB\b|\bAMC\b|\bVA MEDICAL\b")


def is_anexo(name):
    """Un hospital de verdad puede llamarse 'Rehabilitation Hospital'; un anexo no."""
    s = re.sub(r"[^A-Z0-9 ]", " ", L7.norm_name(name))
    return bool(_ANEXO.search(s)) and not bool(_LOOKS.search(s))


def looks_hospital(name):
    s = re.sub(r"[^A-Z0-9 ]", " ", L7.norm_name(name))
    return bool(_LOOKS.search(s)) or bool(_MILITARY.search(s))


def modality_conflict(pnom, ref_name):
    """El POI es un hospital de otro tipo (rehab, dialisis, psiquiatrico) que el del match."""
    p = re.sub(r"[^A-Z0-9 ]", " ", L7.norm_name(pnom))
    r = re.sub(r"[^A-Z0-9 ]", " ", L7.norm_name(ref_name))
    return bool(_MODALITY.search(p)) and not bool(_MODALITY.search(r))


def label_false(name):
    s = re.sub(r"[^A-Z0-9& ]", " ", L7.norm_name(name))
    for k, pat in _LABELS:
        if pat.search(s):
            return k
    return "otro_sin_etiqueta"


# ---------------------------------------------------------------- Overture + tractos
def load_tracts(con):
    off = os.path.join(ROOT, "datos", "features_base", "features_buildings.parquet").replace(os.sep, "/")
    con.execute(f"CREATE TABLE official AS SELECT DISTINCT GEOID FROM read_parquet('{off}')")
    parts = []
    for r in REGIONS:
        p = os.path.join(ROOT, "CONCURSO", "data", "strata", r, f"{r}-census-tracts.parquet")
        parts.append(
            f"SELECT GEOID, '{r}' AS region, STATEFP, COUNTYFP, NAMELSAD, pop_total, pct_urban, "
            f"bbox.xmin AS x0, bbox.xmax AS x1, bbox.ymin AS y0, bbox.ymax AS y1, geometry "
            f"FROM read_parquet('{p.replace(os.sep, '/')}')")
    con.execute("CREATE TABLE tracts AS " + " UNION ALL ".join(parts))
    con.execute("""CREATE TABLE tract_pt AS
        SELECT GEOID,
               ST_X(ST_Centroid(ST_Transform(geometry,'EPSG:4326','EPSG:5070',
                     always_xy := true))) AS cx,
               ST_Y(ST_Centroid(ST_Transform(geometry,'EPSG:4326','EPSG:5070',
                     always_xy := true))) AS cy,
               ST_Y(ST_Centroid(geometry)) AS clat, ST_X(ST_Centroid(geometry)) AS clon
        FROM tracts""")
    con.execute("""CREATE TABLE tract_c AS SELECT t.*, p.cx, p.cy, p.clat, p.clon
                FROM tracts t JOIN tract_pt p USING (GEOID)
                WHERE t.GEOID IN (SELECT GEOID FROM official)""")
    nat = os.path.join(ROOT, "CONCURSO", "data", "strata", "national",
                       "national-strata-tract-table.parquet").replace(os.sep, "/")
    con.execute(f"""CREATE TABLE strata AS
        SELECT GEOID, svi_overall, tribal_pct, aiannh_name, TRY_CAST(pop_total AS BIGINT) pop_nat
        FROM read_parquet('{nat}')""")


def load_pois(con):
    files = []
    for r in REGIONS:
        p = os.path.join(ROOT, "CONCURSO", "data", "reference", r, f"{r}-overture-pois.parquet")
        files.append(f"'{p.replace(os.sep, '/')}'")
    con.execute(f"""CREATE TABLE poi_all AS
        SELECT id, names.primary AS pnom, categories.primary AS pcat,
               taxonomy.primary AS tax, confidence,
               ST_X(ST_Transform(geometry,'EPSG:4326','EPSG:5070', always_xy := true)) AS px,
               ST_Y(ST_Transform(geometry,'EPSG:4326','EPSG:5070', always_xy := true)) AS py,
               ST_Y(geometry) AS plat, ST_X(geometry) AS plon
        FROM read_parquet([{",".join(files)}])
        WHERE geometry IS NOT NULL""")
    cats = "','".join(POOL_CATS)
    con.execute(f"CREATE TABLE poi AS SELECT * FROM poi_all WHERE pcat IN ('{cats}')")
    con.execute("""CREATE TABLE poi_t AS
        SELECT p.*, t.GEOID AS tracto, t.region, t.NAMELSAD AS tracto_nombre,
               ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY t.GEOID) rn
        FROM poi p JOIN tract_c t
          ON p.plon BETWEEN t.x0 AND t.x1 AND p.plat BETWEEN t.y0 AND t.y1
         AND ST_Contains(t.geometry, ST_Point(p.plon, p.plat))""")
    con.execute("""CREATE TABLE poi_g AS
        SELECT p.*, x.tracto, x.region, x.tracto_nombre FROM poi p
        LEFT JOIN (SELECT id, tracto, region, tracto_nombre FROM poi_t WHERE rn = 1) x USING (id)""")


def project_ref(con, ref):
    con.register("ref_df", ref)
    con.execute("""CREATE TABLE refp AS
        SELECT nombre, er, er_lb, tipo, ciudad, cms_id, condado_nombre, key, ini,
               ST_X(ST_Transform(ST_Point(lon, lat),'EPSG:4326','EPSG:5070', always_xy := true)) AS rx,
               ST_Y(ST_Transform(ST_Point(lon, lat),'EPSG:4326','EPSG:5070', always_xy := true)) AS ry
        FROM ref_df""")


# ---------------------------------------------------------------- clasificacion
def classify(pois, ref, refall, state_of_region, km=MATCH_KM):
    """Un POI 'hospital' es real si coincide por nombre con una institucion de referencia
    a <= km. La distancia sola no basta: el anexo de un hospital es otro POI.
    Si no hay referencia con coordenada pero si un hospital del mismo estado con nombre
    casi identico (geocodificacion fallida en CMS), se marca 'real_sin_coord' y no se
    cuenta como falso positivo."""
    rk = ref["key"].to_numpy()
    ri = ref["ini"].to_numpy()
    rer = ref["er"].to_numpy()
    rx, ry = ref["rx"].to_numpy(), ref["ry"].to_numpy()
    ak, ae, ar = refall["key"].to_numpy(), refall["estado"].to_numpy(), refall["er"].to_numpy()
    # indice invertido token -> indices, para no comparar contra toda la referencia
    idx_tok = {}
    for j, s in enumerate(ak):
        for t in str(s).split():
            if len(t) >= 4:
                idx_tok.setdefault(t, []).append(j)
    idx_ref_tok = {}
    for j, s in enumerate(rk):
        for t in str(s).split():
            if len(t) >= 4:
                idx_ref_tok.setdefault(t, []).append(j)
    out = []
    lim = km * 1000.0
    pts = pois[["px", "py", "pnom", "region"]].to_numpy(object)
    for px, py, nom, region in pts:
        d = np.hypot(rx - px, ry - py)
        near = np.where(d <= lim)[0]
        best, best_s, best_d = None, -1.0, np.inf
        k = L7.key_of(nom) if isinstance(nom, str) else ""
        for j in near:
            if not k or not rk[j]:
                continue
            s = L7.sim(k, rk[j])
            cont = L7.contained(k, rk[j])
            ini_ok = len(k.replace(" ", "")) >= 3 and k.replace(" ", "") == ri[j]
            sc = max(s, 0.99 if (cont or ini_ok) else 0.0)
            # se prefiere el mejor nombre, no el POI mas cercano
            if sc >= NAME_SIM and (sc > best_s or (sc == best_s and d[j] < best_d)):
                best, best_s, best_d = j, sc, d[j]
        if best is not None:
            er = bool(rer[best])
            if er and modality_conflict(nom, ref["nombre"].iloc[best]):
                er = False
            out.append(("real", ref["nombre"].iloc[best], float(best_d) / 1000.0, er))
            continue
        st = state_of_region.get(region, "")
        cj = set()
        for t in k.split():
            if len(t) >= 4:
                cj.update(idx_tok.get(t, ()))
        cand = [j for j in cj if ae[j] == st and k and
                (L7.sim(k, ak[j]) >= 0.85 or L7.contained(k, ak[j]))]
        best_er = [j for j in cand if ar[j]]
        if best_er:
            # el hospital con urgencias existe en el registro nominal aunque no tenga coordenada
            j = max(best_er, key=lambda j: L7.sim(k, ak[j]))
            er = not modality_conflict(nom, refall["nombre"].iloc[j])
            out.append(("real_sin_coord", refall["nombre"].iloc[j], np.nan, er))
            continue
        if cand:
            j = max(cand, key=lambda j: L7.sim(k, ak[j]))
            out.append(("real_sin_coord", refall["nombre"].iloc[j], np.nan, False))
            continue
        if len(k.split()) == 1 and len(k) >= 5:
            # nombre generico ("Memorial", "Community"): vale solo si hay un hospital con ese
            # token a <=5 km, y ahi si se conoce la distancia
            for j in idx_ref_tok.get(k, ()):
                if d[j] <= 5000:
                    out.append(("real", ref["nombre"].iloc[j], float(d[j]) / 1000.0, bool(rer[j])))
                    break
            else:
                out.append(("falso", "", np.nan, False))
            continue
        out.append(("falso", "", np.nan, False))
    o = pois.copy()
    o["clase"] = [a[0] for a in out]
    o["ref_nombre"] = [a[1] for a in out]
    o["ref_km"] = [a[2] for a in out]
    o["match_er"] = [bool(a[3]) if a[0] in ("real", "real_sin_coord") else False for a in out]
    o["anexo"] = [is_anexo(n) for n in o["pnom"]]
    o["looks"] = [looks_hospital(n) for n in o["pnom"]]
    # un anexo (consulta, imagen, heridas, departamento) no es el hospital aunque comparta nombre
    o["tiene_urgencias"] = o["match_er"] & ~o["anexo"]
    o["defB"] = o["clase"].ne("falso") & ~o["anexo"]
    o["defC"] = o["defB"] | o["looks"] | o["pnom"].map(label_false).isin(
        ["imagen_lab", "enfermeria_ltc", "hospice_home_health", "urgent_care"])
    o["etiqueta"] = [("" if c != "falso" else label_false(n))
                     for c, n in zip(o["clase"], o["pnom"])]
    return o


# ---------------------------------------------------------------- estadistica
def wilson(k, n):
    return L7.wilson(k, n)


def pct_ci(k, n):
    if n == 0:
        return "   n/a"
    lo, hi = wilson(k, n)
    return f"{100*k/n:5.1f} [{100*lo:4.1f},{100*hi:4.1f}]"


def mh_or(df, expo, case):
    return L7.mh_or(df, expo, case=case)


# ---------------------------------------------------------------- main
def main():
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='8GB'")
    con.execute("INSTALL spatial; LOAD spatial;")

    ref, refall = build_ref()
    print(f"[src] referencia: CMS {int((ref.tipo.str.startswith('CMS')).sum())} "
          f"(urgencias {int((ref.er & ref.tipo.str.startswith('CMS')).sum())}) | "
          f"VA {int((ref.tipo=='VA').sum())} (VAMC {int(((ref.tipo=='VA') & ref.er).sum())}) | "
          f"IHS {int((ref.tipo=='IHS').sum())} (ER {int(((ref.tipo=='IHS') & ref.er).sum())})")
    print(f"[src] hospitales con urgencias (base) en los 15 estados: {int(ref.er.sum())}; "
          f"cota inferior (todo agudo/CAH/REH/VA/DoD): {int(ref.er_lb.sum())}")
    for lab, m in [("CMS flag Yes", ref.tipo.str.startswith("CMS") & ref.er &
                    ref.motivo.eq("flag Yes")),
                   ("CMS CAH/REH por normativa", ref.motivo.eq("normativa (CAH/REH)")),
                   ("CMS agudo general con flag No", ref.motivo.eq("agudo general con flag No")),
                   ("CMS agudo de especialidad (No)", ref.motivo.eq("agudo especialidad, flag No")),
                   ("VA (VAMC)", ref.tipo.eq("VA") & ref.er),
                   ("IHS con ER", ref.tipo.eq("IHS") & ref.er),
                   ("FSER con licencia TX", ref.tipo.eq("TX-FSER"))]:
        print(f"    {lab:<32} {int(m.sum()):>5}")
    print(f"[src] registro nominal completo (con y sin coordenada): {len(refall)} instituciones")

    load_tracts(con)
    load_pois(con)
    project_ref(con, ref)
    ref = con.execute("SELECT * FROM refp").df()
    print(f"[src] Overture: {con.execute('SELECT count(*) FROM poi_all').fetchone()[0]} POIs; "
          f"hospital/medical_center/childrens={con.execute('SELECT count(*) FROM poi').fetchone()[0]}")

    pois = con.execute("SELECT * FROM poi_g").df()
    pois = classify(pois, ref, refall, dict(zip(REGIONS, [r[-2:].upper() for r in REGIONS])))
    h = pois[pois.pcat.isin(HOSP_CATS)].copy()
    print(f"[poi] categoria 'hospital': {len(h)} POIs  "
          f"real_con_urgencias={int(h.tiene_urgencias.sum())} "
          f"real_sin_urgencias={int(((h.clase=='real') & ~h.tiene_urgencias).sum())} "
          f"real_sin_coordenada={int((h.clase=='real_sin_coord').sum())} "
          f"falso={int((h.clase=='falso').sum())}")

    nat = con.execute("""SELECT t.*, s.svi_overall, s.tribal_pct, s.aiannh_name
                         FROM tract_c t LEFT JOIN strata s USING (GEOID)""").df()
    cnames = load_county_names()
    nat["condado"] = nat.STATEFP + nat.COUNTYFP
    nat["condado_nombre"] = nat["condado"].map(cnames).fillna(nat["condado"])
    nat["tracto_nombre"] = nat["NAMELSAD"]
    nat["rural"] = nat["pct_urban"] < 0.5
    nat["tribal"] = nat["tribal_pct"].fillna(0) > 0
    nat["svi_alto"] = nat["svi_overall"] >= 0.75
    nat["svi_bajo"] = nat["svi_overall"] <= 0.25

    # ---------------- 1. precision por region y rural/urbano
    er_pts = ref[ref.er].reset_index(drop=True)
    _, _i0 = cKDTree(er_pts[["rx", "ry"]].to_numpy()).query(
        np.c_[nat.cx.to_numpy(), nat.cy.to_numpy()], k=1)
    nat["real_nombre"] = er_pts["nombre"].to_numpy()[_i0]
    h = h.dropna(subset=["tracto"]).merge(
        nat[["GEOID", "condado", "condado_nombre", "rural", "tribal", "svi_alto", "svi_bajo"]],
        left_on="tracto", right_on="GEOID", how="left")
    print("\n=== 1. PRECISION DE LA CATEGORIA 'hospital' EN TRES DEFINICIONES ===")
    print("  A = hospital con sala de urgencias | B = hospital de cualquier tipo |")
    print("  C = no es clinica ni consulta (hospital, urgencias, imagen/lab, enfermeria, hospice)")
    print("  region              n     A [IC95]              B [IC95]              C [IC95]     infl.")
    for r in list(REGIONS) + ["TOTAL"]:
        g = h if r == "TOTAL" else h[h.region == r]
        n_ref = nat[nat.region == r]["real_nombre"].nunique() if r != "TOTAL" else \
            nat["real_nombre"].nunique()
        print(f"    {r:<17} {len(g):>5}  {pct_ci(int(g.tiene_urgencias.sum()), len(g))}  "
              f"{pct_ci(int(g.defB.sum()), len(g))}  {pct_ci(int(g.defC.sum()), len(g))}  "
              f"{len(g)/max(n_ref,1):5.1f}x")
    print("  rural vs urbano:")
    for lab, g in [("rural", h[h.rural == True]), ("urbano", h[h.rural == False])]:  # noqa: E712
        print(f"    {lab:<17} {len(g):>5}  {pct_ci(int(g.tiene_urgencias.sum()), len(g))}  "
              f"{pct_ci(int(g.defB.sum()), len(g))}  {pct_ci(int(g.defC.sum()), len(g))}")
    print(f"  anexos de hospital con nombre parecido (consulta, imagen, heridas, unidad): "
          f"{int(h.anexo.sum())} POIs ({100*h.anexo.mean():.1f}%)")
    print("  etiqueta de los POIs 'hospital' que no son hospitales:")
    lab = h[h.clase == "falso"]["etiqueta"].value_counts()
    for k, v in lab.items():
        print(f"    {k:<26} {v:>5} ({100*v/len(h):4.1f}% del total)")
    print("  de los POIs que empatan con un hospital, cuantos son de un tipo sin urgencias:")
    rs = h[h.clase.ne("falso") & (~h.match_er)]
    print(f"    {len(rs)} de {int(h.clase.ne('falso').sum())} "
          f"({100*len(rs)/max(int(h.clase.ne('falso').sum()),1):.1f}%)")

    # ---------------- 2. por tracto
    hs = h[["id", "pnom", "px", "py", "clase", "tiene_urgencias", "etiqueta", "confidence",
            "anexo", "looks", "defB", "defC"]].copy()
    re_ = ref[ref.er][["rx", "ry", "nombre", "tipo", "ciudad"]].copy()
    re_lb = ref[ref.er_lb][["rx", "ry", "nombre", "tipo", "ciudad"]].copy()

    hx, hy = hs.px.to_numpy(), hs.py.to_numpy()
    tx, ty = nat.cx.to_numpy(), nat.cy.to_numpy()
    tree_h = cKDTree(np.c_[hx, hy])
    dh, ih = tree_h.query(np.c_[tx, ty], k=1)
    tree_r = cKDTree(re_[["rx", "ry"]].to_numpy())
    dr, ir = tree_r.query(np.c_[tx, ty], k=1)
    # cota inferior: todo agudo, CAH, REH, VA y DoD, sin mirar el flag emergency_services
    dl, il = cKDTree(re_lb[["rx", "ry"]].to_numpy()).query(np.c_[tx, ty], k=1)
    nat["real_km_lb"] = dl / 1000.0
    nat["real_nombre_lb"] = re_lb["nombre"].to_numpy()[il]
    # cualquier hospital de la referencia, separa el error del POI del error del campo
    da, _ = cKDTree(ref[["rx", "ry"]].to_numpy()).query(np.c_[tx, ty], k=1)
    nat["any_km"] = da / 1000.0

    nat["poi_id"] = hs["id"].to_numpy()[ih]
    nat["poi_nombre"] = hs["pnom"].to_numpy()[ih]
    nat["poi_clase"] = hs["clase"].to_numpy()[ih]
    nat["poi_urgencias"] = hs["tiene_urgencias"].to_numpy()[ih]
    nat["poi_etiqueta"] = hs["etiqueta"].to_numpy()[ih]
    nat["poi_conf"] = hs["confidence"].to_numpy()[ih]
    nat["poi_anexo"] = hs["anexo"].to_numpy()[ih]
    nat["poi_looks"] = hs["looks"].to_numpy()[ih]
    nat["poi_defB"] = hs["defB"].to_numpy()[ih]
    nat["poi_defC"] = hs["defC"].to_numpy()[ih]
    nat["poi_km"] = dh / 1000.0
    nat["real_nombre"] = re_["nombre"].to_numpy()[ir]
    nat["real_tipo"] = re_["tipo"].to_numpy()[ir]
    nat["real_ciudad"] = re_["ciudad"].to_numpy()[ir]
    nat["real_km"] = dr / 1000.0
    nat["error"] = (~nat.poi_urgencias) & nat.poi_km.notna()
    nat["artefacto_er"] = (nat.real_km - nat.any_km) > 3.0
    nat["km_extra"] = (nat.real_km - nat.poi_km).clip(lower=0)
    # el hospital existe pero CMS no lo pudo geocodificar: el POI esta donde debe estar
    nat.loc[nat.poi_urgencias & nat.poi_clase.eq("real_sin_coord"), "km_extra"] = 0.0
    nat["error_lb"] = (~nat.poi_urgencias) & nat.poi_km.notna()
    nat["km_extra_lb"] = (nat.real_km_lb - nat.poi_km).clip(lower=0)
    nat.loc[nat.poi_urgencias & nat.poi_clase.eq("real_sin_coord"), "km_extra_lb"] = 0.0

    print("\n=== 2. POR TRACTO: distancia al 'hospital' de Overture vs al hospital real con urgencias ===")
    print("  (linea recta centroide-POI, no tiempo de viaje)")
    for lab, g in [("todos", nat), ("rural", nat[nat.rural == True]), ("urbano", nat[nat.rural == False])]:  # noqa: E712
        print(f"    {lab:<8} tractos={len(g):>5}  POI mas cercano NO tiene urgencias: "
              f"{pct_ci(int(g.error.sum()), len(g))}  "
              f"km POI mediana={g.poi_km.median():5.2f}  km real mediana={g.real_km.median():5.2f}  "
              f"extra medio={g.km_extra.mean():5.2f}")
    print("  cota inferior de la referencia (todo agudo/CAH/REH/VA/DoD/FSER, sin mirar el flag):")
    for lab, g in [("todos", nat), ("rural", nat[nat.rural == True]),  # noqa: E712
                   ("urbano", nat[nat.rural == False])]:  # noqa: E712
        print(f"    {lab:<8} POI mas cercano NO tiene urgencias: "
              f"{pct_ci(int(g.error_lb.sum()), len(g))}  "
              f"km real mediana={g.real_km_lb.median():5.2f}  "
              f"extra medio={g.km_extra_lb.mean():5.2f}  "
              f"extra_pond={np.average(g.km_extra_lb, weights=np.maximum(g.pop_total.fillna(0),1)):5.2f}")
    print("  por region (tasa de tractos mal orientados / km extra medio ponderado por poblacion):")
    for r in REGIONS:
        g = nat[nat.region == r]
        w = g.pop_total.fillna(0)
        print(f"    {r:<17} n={len(g):>5} error={100*g.error.mean():5.1f}%  "
              f"extra_pond={np.average(g.km_extra, weights=np.maximum(w,1)):5.2f} km  "
              f"pob_afectada={int(w[g.error].sum()):>7,} ({100*w[g.error].sum()/max(w.sum(),1):4.1f}%)")

    print("  inflacion de la categoria: POIs 'hospital' por hospital con urgencias que sirve la region")
    for r in REGIONS:
        g = nat[nat.region == r]
        n_ref = g["real_nombre"].nunique()
        n_poi = int((h.region == r).sum())
        print(f"    {r:<17} POIs={n_poi:>5}  hospitales con urgencias={n_ref:>4}  "
              f"ratio={n_poi/max(n_ref,1):5.1f}x")

    pop = nat.pop_total.fillna(0)
    print(f"\n  POBLACION: {int(pop.sum()):,} hab en {len(nat):,} tractos; "
          f"{int(pop[nat.error].sum()):,} ({100*pop[nat.error].sum()/pop.sum():.1f}%) en tractos cuyo "
          f"'hospital' Overture mas cercano no tiene urgencias")
    print(f"  km extra ponderado por poblacion: "
          f"{np.average(nat.km_extra, weights=np.maximum(pop,1)):.2f} km; "
          f"sobre los tractos afectados: {np.average(nat.km_extra[nat.error], weights=np.maximum(pop[nat.error],1)):.2f} km")

    # ---------------- 3. disparidad binaria y de gravedad
    smap = dict(zip(REGIONS, [r[-2:].upper() for r in REGIONS]))
    pop = nat.pop_total.fillna(0)
    print("\n=== 3. DISPARIDAD BINARIA DENTRO DEL CONDADO (todos los errores) ===")
    res = {}
    nat["estrato"] = nat["condado"]
    for key, sub, lab in [
        ("rural", nat, "rural vs urbano"),
        ("tribal", nat, "tribal vs no tribal"),
        ("svi", nat[nat.svi_alto | nat.svi_bajo], "SVI alto vs bajo"),
    ]:
        expo = "svi_alto" if key == "svi" else key
        r = mh_or(sub, expo, "error")
        res[key] = r
        if r is None or not np.isfinite(r.get("or_mh", np.nan)):
            print(f"  {lab:<22} no identificado")
            continue
        boot = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
        print(f"  {lab:<22} OR_MH={r['or_mh']:.2f} IC95[{r['lo']:.2f},{r['hi']:.2f}]{boot} "
              f"condados={r['k']} (inf.={r['k_inf']}) n={r['n']}"
              f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")

    # ---------------- 3b. gravedad: la cola, no la media
    print("\n=== 3b. GRAVEDAD: TRACTOS CON ERROR Y KM EXTRA EN LA COLA ===")
    print("  (una cosa es que el POI mas cercano no sea el hospital; otra, que el hospital")
    print("   real quede a 10, 20 o 30 km mas lejos. Misma tasa de error, dano distinto)")
    for T in (10, 20, 30):
        nat[f"e{T}"] = nat.error & (nat.km_extra >= T)
    def strata_list():
        return [("todos", nat), ("rural", nat[nat.rural == True]),  # noqa: E712
                ("urbano", nat[nat.rural == False]),  # noqa: E712
                ("tribal", nat[nat.tribal == True]),  # noqa: E712
                ("no tribal", nat[nat.tribal == False]),  # noqa: E712
                ("SVI alto", nat[nat.svi_alto == True]),  # noqa: E712
                ("SVI bajo", nat[nat.svi_bajo == True])]  # noqa: E712

    ESTR = strata_list()
    print(f"    {'estrato':<14}{'n':>6}  {'error%':>7} {'>=10km%':>8} {'>=20km%':>8} {'>=30km%':>8}"
          f"   {'pob>=10km%':>10} {'pob>=20km%':>10} {'pob>=30km%':>10}")
    for lab, g in ESTR:
        w = g.pop_total.fillna(0)
        ws = max(w.sum(), 1)
        print(f"    {lab:<14}{len(g):>6}  {100*g.error.mean():>7.1f} "
              f"{100*g.e10.mean():>8.1f} {100*g.e20.mean():>8.1f} {100*g.e30.mean():>8.1f}"
              f"   {100*w[g.e10].sum()/ws:>10.1f} {100*w[g.e20].sum()/ws:>10.1f} "
              f"{100*w[g.e30].sum()/ws:>10.1f}")
    print("  por region:")
    for r in REGIONS:
        g = nat[nat.region == r]
        w = g.pop_total.fillna(0)
        ws = max(w.sum(), 1)
        print(f"    {r:<17}{len(g):>5}  {100*g.error.mean():>7.1f} "
              f"{100*g.e10.mean():>8.1f} {100*g.e20.mean():>8.1f} {100*g.e30.mean():>8.1f}"
              f"   {100*w[g.e10].sum()/ws:>10.1f} {100*w[g.e20].sum()/ws:>10.1f} "
              f"{100*w[g.e30].sum()/ws:>10.1f}")
    print("  km extra (media y p90) por estrato, solo tractos con error:")
    for lab, g in ESTR:
        ge = g[g.error]
        print(f"    {lab:<14} n={len(ge):>5} media={ge.km_extra.mean():5.2f} p50={ge.km_extra.median():5.2f} "
              f"p90={ge.km_extra.quantile(.9):6.2f} max={ge.km_extra.max():6.1f}")
    print("  OR_MH de la cola (condado como estrato; 'caso' = error y km extra >= umbral):")
    for T in (10, 20, 30):
        for key, expo in [("rural", "rural"), ("tribal", "tribal"), ("svi", "svi_alto")]:
            sub = nat[nat.svi_alto | nat.svi_bajo] if key == "svi" else nat
            r = mh_or(sub, expo, f"e{T}")
            res[f"{key}_e{T}"] = r
            if r is None or not np.isfinite(r.get("or_mh", np.nan)):
                print(f"    >= {T} km {key:<8} no identificado")
                continue
            boot = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
            print(f"    >= {T} km {key:<8} OR_MH={r['or_mh']:.2f} IC95[{r['lo']:.2f},{r['hi']:.2f}]{boot}"
                  f" condados={r['k']} n={r['n']}"
                  f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")
    for T in (10, 20):
        nat[f"e{T}_lb"] = nat.error_lb & (nat.km_extra_lb >= T)
    print("  cota inferior (mismo calculo con la referencia mas generosa):")
    for lab, g in [("rural", nat[nat.rural == True]), ("urbano", nat[nat.rural == False])]:  # noqa: E712
        print(f"    {lab:<8} error={100*g.error_lb.mean():5.1f}%  "
              f">=10 km={100*g.e10_lb.mean():5.1f}%  >=20 km={100*g.e20_lb.mean():5.1f}%  "
              f"extra p90={g.km_extra_lb.quantile(.9):6.2f}")
    for T in ("e10_lb", "e20_lb"):
        r = mh_or(nat, "rural", T)
        res[f"rural_{T}"] = r
        if r and np.isfinite(r.get("or_mh", np.nan)):
            boot = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
            print(f"    {T} rural OR_MH={r['or_mh']:.2f} IC95[{r['lo']:.2f},{r['hi']:.2f}]{boot}"
                  f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")
    print("  el condado puede absorber la ruralidad: mismo OR con region x densidad de POIs")
    dens = pois.groupby("tracto").size().rename("n_pois")
    nat["n_pois"] = nat.GEOID.map(dens).fillna(0)
    # Desempate por GEOID: 5,522 tractos tienen 0 POIs y rank("first") dependería del orden de filas de DuckDB.
    rango = nat.sort_values("GEOID")["n_pois"].rank(method="first").reindex(nat.index)
    nat["dens_t"] = pd.qcut(rango, 3, labels=["baja", "media", "alta"])
    nat["estrato"] = nat.region.astype(str) + "|" + nat.dens_t.astype(str)
    for T in ("e10", "e20"):
        r = mh_or(nat.dropna(subset=["dens_t"]), "rural", T)
        res[f"rural_{T}_regdens"] = r
        if r is None or not np.isfinite(r.get("or_mh", np.nan)):
            print(f"    {T} region x densidad: no identificado")
            continue
        boot = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
        print(f"    {T} region x densidad OR_MH={r['or_mh']:.2f} IC95[{r['lo']:.2f},{r['hi']:.2f}]{boot}"
              f" estratos={r['k']} n={r['n']}"
              f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")
    nat["estrato"] = nat["condado"]

    # ---------------- 3c. proporcion: ¿es solo geografia?
    print("\n=== 3c. EXTRA-KM EN PROPORCION (km extra / km al hospital real) ===")
    print("  Si el error se concentra donde todo queda lejos, la proporcion no se mueve.")
    nat["ratio"] = np.where(nat.real_km > 0, nat.km_extra / nat.real_km, np.nan)
    for lab, g in strata_list():
        gr = g[g.error & g.ratio.notna()]
        print(f"    {lab:<14} n={len(gr):>5} p50={gr.ratio.median():5.2f} p90={gr.ratio.quantile(.9):5.2f} "
              f"ratio>=0.5: {100*gr.ratio.ge(0.5).mean():5.1f}%  ratio>=0.8: {100*gr.ratio.ge(0.8).mean():5.1f}%")
    nat["r50"] = nat.error & nat.ratio.ge(0.5)
    for key, expo in [("rural", "rural"), ("tribal", "tribal"), ("svi", "svi_alto")]:
        sub = nat[nat.svi_alto | nat.svi_bajo] if key == "svi" else nat
        r = mh_or(sub, expo, "r50")
        res[f"{key}_r50"] = r
        if r is None or not np.isfinite(r.get("or_mh", np.nan)):
            print(f"    OR_MH {key:<8} ratio>=0.5 no identificado")
            continue
        boot = f" boot[{r['boot'][0]:.2f},{r['boot'][1]:.2f}]" if r.get("boot") else ""
        print(f"    OR_MH {key:<8} ratio>=0.5 = {r['or_mh']:.2f} IC95[{r['lo']:.2f},{r['hi']:.2f}]{boot}"
              f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")

    # ---------------- 3d. variacion entre regiones
    print("\n=== 3d. LA MISMA ETIQUETA NO SIGNIFICA LO MISMO EN CADA REGION ===")
    from scipy.stats import chi2_contingency, fisher_exact
    tab = np.array([[int(h[h.region == r].tiene_urgencias.sum()),
                     int((~h[h.region == r].tiene_urgencias).sum())] for r in REGIONS])
    chi2, pchi, _, _ = chi2_contingency(tab)
    print(f"  chi2 de homogeneidad de la precision A entre las 5 regiones: chi2={chi2:.1f} p={pchi:.2e}")
    print("  pares con diferencia (Fisher, Bonferroni 10 comparaciones):")
    for i in range(len(REGIONS)):
        for j in range(i + 1, len(REGIONS)):
            a1, b1 = tab[i]
            a2, b2 = tab[j]
            _, pv = fisher_exact([[a1, b1], [a2, b2]])
            if pv * 10 < 0.05:
                print(f"    {REGIONS[i]:<17} {100*a1/(a1+b1):5.1f}%  vs  {REGIONS[j]:<17} "
                      f"{100*a2/(a2+b2):5.1f}%   p={pv:.1e} (p_adj={min(pv*10,1):.1e})")

    # ---------------- 4. confidence
    print("\n=== 4. QUE PASA SI EL USUARIO FILTRA POR confidence ALTA (>=0.90) ===")
    for thr in [None, 0.90]:
        g = h if thr is None else h[h.confidence >= thr]
        print(f"  {'sin filtro' if thr is None else 'conf>=%.2f' % thr:<12} n={len(g):>5} "
              f"A={pct_ci(int(g.tiene_urgencias.sum()), len(g))} "
              f"C={pct_ci(int(g.defC.sum()), len(g))}")
    hc = h[h.confidence >= 0.9]
    dhc, ihc = cKDTree(np.c_[hc.px.to_numpy(), hc.py.to_numpy()]).query(np.c_[tx, ty], k=1)
    err_hc = (~hc.tiene_urgencias.to_numpy()[ihc])
    print(f"  tractos con solo POIs de conf alta: {len(hc)} POIs -> mal orientados "
          f"{100*err_hc.mean():.1f}% (sin filtro {100*nat.error.mean():.1f}%)")
    nac = nat.copy()
    nac["error_hc"] = err_hc
    r = mh_or(nac, "rural", "error_hc")
    if r and np.isfinite(r.get("or_mh", np.nan)):
        print(f"  OR_MH rural con el filtro de confianza: {r['or_mh']:.2f} "
              f"[{r['lo']:.2f},{r['hi']:.2f}]"
              f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")

    # ---------------- 5. especificidad y sensibilidad
    print("\n=== 5. NO ES SOLO DENSIDAD, Y SENSIBILIDAD ===")
    tt = nat.pivot_table(index="dens_t", columns="rural", values="error", aggfunc="mean") * 100
    tn = nat.pivot_table(index="dens_t", columns="rural", values="error", aggfunc="size")
    print(pd.concat([tt.round(1).add_suffix("_err%"), tn.add_suffix("_n")], axis=1).to_string())
    nat["estrato"] = nat["dens_t"]
    rd = mh_or(nat.dropna(subset=["dens_t"]), "rural", "error")
    if rd and np.isfinite(rd.get("or_mh", np.nan)):
        print(f"  rural ajustado SOLO por densidad de POIs: OR_MH={rd['or_mh']:.2f} "
              f"[{rd['lo']:.2f},{rd['hi']:.2f}]"
              f"{'  SIG' if (rd['lo']>1 or rd['hi']<1) else '  no sig'}")
    nat["estrato"] = nat["condado"]
    print(f"  nulos de confidence entre los POIs 'hospital': {int(h.confidence.isna().sum())}; "
          f"<=0: {int((h.confidence <= 0).sum()) if h.confidence.notna().any() else 0}")
    base = dict(prec_a=100 * h.tiene_urgencias.mean(), prec_c=100 * h.defC.mean(),
                err=100 * nat.error.mean(), extra=float(np.average(nat.km_extra, weights=np.maximum(pop, 1))))
    print(f"  BASE: A={base['prec_a']:.1f}% C={base['prec_c']:.1f}% error={base['err']:.1f}% "
          f"extra_pond={base['extra']:.2f} km")
    for lab, km, drop_null in [("tolerancia de nombre 5 km", 5.0, False),
                               ("sin POIs de confidence nula", MATCH_KM, True)]:
        h2 = classify(pois, ref, refall, smap, km=km)
        h2 = h2[h2.pcat.isin(HOSP_CATS) & h2.tracto.isin(set(nat.GEOID))].copy()
        if drop_null:
            h2 = h2[h2.confidence.notna() & (h2.confidence > 0)].copy()
        dm, im = cKDTree(np.c_[h2.px.to_numpy(), h2.py.to_numpy()]).query(np.c_[tx, ty], k=1)
        e2 = (~h2.tiene_urgencias.to_numpy()[im]) & np.isfinite(dm)
        x2 = (dr / 1000.0 - dm / 1000.0)
        x2 = np.where(e2, np.clip(x2, 0, None), 0.0)
        print(f"  {lab:<30} n={len(h2):>5} A={100*h2.tiene_urgencias.mean():5.1f}% "
              f"C={100*h2.defC.mean():5.1f}% error={100*e2.mean():5.1f}% "
              f"extra_pond={np.average(x2, weights=np.maximum(pop, 1)):4.2f} km "
              f"(delta error {100*e2.mean()-base['err']:+.1f} pp)")
    print("  poblacion: ya se pondera por el total del tracto puesto en su centroide; el")
    print("  ponderado y el no ponderado se imprimen juntos en la seccion 2.")

    # ---------------- 6. piso conservador
    print("\n=== 6. PISO CONSERVADOR: TRATAR COMO HOSPITAL TODO NOMBRE QUE LO PAREZCA ===")
    nat["piso_real"] = nat.poi_defC | nat.poi_looks
    nat["error_piso"] = ~nat.piso_real
    nat["extra_piso"] = np.where(nat.error_piso, nat.km_extra, 0.0)
    print(f"  base:         error={100*nat.error.mean():5.1f}%  extra_pond="
          f"{np.average(nat.km_extra, weights=np.maximum(pop,1)):4.2f} km  "
          f"pob={100*pop[nat.error].sum()/max(pop.sum(),1):4.1f}%")
    print(f"  piso:         error={100*nat.error_piso.mean():5.1f}%  extra_pond="
          f"{np.average(nat.extra_piso, weights=np.maximum(pop,1)):4.2f} km  "
          f"pob={100*pop[nat.error_piso].sum()/max(pop.sum(),1):4.1f}%")
    for lab, g in [("rural", nat[nat.rural == True]), ("urbano", nat[nat.rural == False])]:  # noqa: E712
        w = np.maximum(g.pop_total.fillna(0), 1)
        print(f"  piso {lab:<7} error={100*g.error_piso.mean():5.1f}%  "
              f"extra_pond={np.average(g.extra_piso, weights=w):4.2f} km")
    for T in (10, 20, 30):
        k = int((nat.error_piso & (nat.extra_piso >= T)).sum())
        print(f"  piso >= {T} km: {k} tractos ({100*k/len(nat):4.1f}%), "
              f"{int(pop[nat.error_piso & (nat.extra_piso >= T)].sum()):>9,} hab")
    nat["estrato"] = nat["condado"]
    r = mh_or(nat, "rural", "error_piso")
    res["rural_piso"] = r
    if r and np.isfinite(r.get("or_mh", np.nan)):
        print(f"  OR_MH rural en el piso: {r['or_mh']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]"
              f"{'  SIG' if (r['lo']>1 or r['hi']<1) else '  no sig'}")
    print("  categorias exactas de Overture contadas como 'hospital': "
          + ", ".join(HOSP_CATS))
    extra_cats = [f"{c}={int(n)}" for c, n in
                  pois[~pois.pcat.isin(HOSP_CATS)].pcat.value_counts().items()]
    print("  afines NO contadas: " + ", ".join(extra_cats))
    print("  (si se sumaran, el denominador se multiplica y la precision cae aun mas)")

    # ---------------- 7. validacion por muestras
    print("\n=== 7. VALIDACION POR MUESTRAS: 100 'falsos' + 100 'reales' + 100 nombre+estado (semilla 42) ===")
    print("  Criterio, escrito antes de mirar la muestra:")
    print("   A: el POI es el hospital con sala de urgencias (no una consulta, unidad o anexo).")
    print("   B: el POI es un hospital de cualquier tipo (psiquiatrico, LTAC, rehab) o su anexo.")
    print("   C: el POI no es clinica ni consulta (hospital, urgencias, imagen/lab, enfermeria, hospice).")
    fa = h[h.clase == "falso"].sample(n=min(100, int((h.clase == "falso").sum())), random_state=42)
    re_s = h[h.clase.eq("real") & h.tiene_urgencias].sample(
        n=min(100, int((h.clase.eq("real") & h.tiene_urgencias).sum())), random_state=42)
    ne = h[h.clase.eq("real_sin_coord")].sample(
        n=min(100, int(h.clase.eq("real_sin_coord").sum())), random_state=42)
    print(f"\n  --- {len(fa)} marcados FALSOS ---")
    for i, (_, r) in enumerate(fa.iterrows(), 1):
        print(f"  {i:3d}. [{r.etiqueta:<22}] {str(r.pnom)[:50]:<52} {r.region:<16} {r.tracto}")
    print(f"\n  --- {len(re_s)} marcados REALES con urgencias ---")
    for i, (_, r) in enumerate(re_s.iterrows(), 1):
        print(f"  {i:3d}. {str(r.pnom)[:42]:<44} -> {str(r.ref_nombre)[:36]:<38} "
              f"{str(r.ref_km)[:5]:>5} km  {r.region:<15}")
    print(f"\n  --- {len(ne)} emparejados SOLO por nombre y estado ---")
    for i, (_, r) in enumerate(ne.iterrows(), 1):
        print(f"  {i:3d}. {str(r.pnom)[:42]:<44} -> {str(r.ref_nombre)[:36]:<38} "
              f"{'ER' if r.tiene_urgencias else 'no-ER':<5} {r.region:<15}")

    # Lectura de las muestras de arriba. (falso_positivo, falso_negativo) por definicion:
    # falso_positivo = marcados reales que en verdad no cumplen la definicion;
    # falso_negativo = marcados falsos que en verdad si la cumplen.
    print("\n  lectura -> precision corregida por definicion "
          "(VP=%.0f%%, FN=%.0f%% en A)" % (100 * VERIF["A"][0], 100 * VERIF["A"][1]))
    for D, col in [("A", "tiene_urgencias"), ("B", "defB"), ("C", "defC")]:
        fp, fn = VERIF[D]
        n1 = int(h[col].sum())
        n0 = len(h) - n1
        corr = (n1 * (1 - fp) + n0 * fn) / len(h)
        lo, hi = wilson(int(round(corr * len(h))), len(h))
        print(f"    {D}: crudo {100*n1/len(h):5.1f}% -> corregido {100*corr:5.1f}% "
              f"[{100*lo:.1f},{100*hi:.1f}]  (VP medido {100*fp:.0f}% en los reales, "
              f"FN medido {100*fn:.0f}% en los falsos)")
    nat["error_c"] = np.select([nat.poi_clase.eq("falso"), ~nat.poi_urgencias],
                               [1 - VERIF["A"][1], 1.0], default=VERIF["A"][0])
    nat["estrato"] = nat["condado"]
    rc = L7.mh_stats(pd.DataFrame([
        dict(a=(g.rural & (g.error_c > 0.5)).sum(), b=(g.rural & (g.error_c <= 0.5)).sum(),
             c=(~g.rural & (g.error_c > 0.5)).sum(), e=(~g.rural & (g.error_c <= 0.5)).sum())
        for _, g in nat.groupby("estrato")]), seed=42)
    res["rural_corr"] = rc
    print(f"    OR_MH rural con la correccion: {rc['or_mh']:.2f} [{rc['lo']:.2f},{rc['hi']:.2f}]"
          f"{'  SIG' if (rc['lo']>1 or rc['hi']<1) else '  no sig'}")

    # ---------------- 8. tabla de 15
    print("\n=== 8. 15 TRACTOS CON MAYOR POBLACION x km EXTRA (POI verificado como no-hospital) ===")
    ref_flaca = nat.any_km > 15.0   # la referencia no tiene ningun hospital cerca: no es el POI
    top = nat[nat.error & ~nat.artefacto_er & ~nat.piso_real & ~ref_flaca].copy()
    print(f"  (fuera {int((nat.error & nat.artefacto_er).sum())} tractos donde el hueco lo explica")
    print(f"   el campo emergency_services de CMS, {int((nat.error & nat.piso_real).sum())} donde")
    print(f"   el nombre del POI sugiere hospital -esos son el piso de la seccion 6- y")
    print(f"   {int((nat.error & ~nat.artefacto_er & ~nat.piso_real & ref_flaca).sum())} tractos sin")
    print(f"   ningun hospital de referencia a <=15 km: ahi el hueco es de la referencia, no del POI)")
    rem = nat[nat.error & ~nat.artefacto_er & ~nat.piso_real & ref_flaca]
    print(f"  casos remotos que el filtro deja fuera (no hay hospital de referencia a <=15 km;")
    print(f"  el POI tambien es falso, pero ahi el hueco es del desierto hospitalario):")
    for _, r in rem.sort_values("pop_total", ascending=False).head(3).iterrows():
        print(f"    tract {r.GEOID} {r.tracto_nombre} | {r.condado_nombre}, {r.region} | "
              f"pop={int(r.pop_total or 0):,} | POI '{r.poi_nombre}' [{r.poi_etiqueta}] "
              f"{r.poi_km:.1f} km -> '{r.real_nombre}' {r.real_km:.1f} km")
    top["peso"] = top.pop_total.fillna(0) * top.km_extra
    top = top.sort_values("peso", ascending=False).head(15)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i:2d}. tract {r.GEOID} {r.tracto_nombre} | {r.condado_nombre}, "
              f"{r.region} | pop={int(r.pop_total or 0):>6,} | "
              f"POI '{r.poi_nombre}' [{r.poi_etiqueta}] {r.poi_km:.1f} km -> "
              f"real '{r.real_nombre}' {r.real_km:.1f} km (+{r.km_extra:.1f} km)")
    n_top = int(top.GEOID.isin(VERIF_URB).sum())
    top = nat[nat.GEOID.isin(VERIF_URB)].copy()
    top["peso"] = top.pop_total.fillna(0) * top.km_extra
    top = top.sort_values("peso", ascending=False)
    print(f"  filas verificadas: {len(top)}, de ellas entre los 15 de esta corrida: {n_top}")
    print(f"  verificados en la web y sin ER autonomo mas cerca: {len(top)} tractos")
    for _, r in top.iterrows():
        v = VERIF_URB[r.GEOID]
        print(f"    {v['lugar']:<13} {v['condado']:<18} {v['nota']}")

    # ---------------- 9. veredicto
    print("\n=== 9. VEREDICTO ===")

    def sig(k):
        v = res.get(k)
        return bool(v and np.isfinite(v.get("or_mh", np.nan)) and (v["lo"] > 1 or v["hi"] < 1))
    print(f"  base: A={100*h.tiene_urgencias.mean():.1f}% B={100*h.defB.mean():.1f}% "
          f"C={100*h.defC.mean():.1f}%; error de tracto {100*nat.error.mean():.1f}%, "
          f"piso {100*nat.error_piso.mean():.1f}%")
    print(f"  OR_MH rural: binario {'SIG' if sig('rural') else 'no sig'}; "
          f">=10km {'SIG' if sig('rural_e10') else 'no sig'}; "
          f">=20km {'SIG' if sig('rural_e20') else 'no sig'}; "
          f"proporcion {'SIG' if sig('rural_r50') else 'no sig'}; "
          f"piso {'SIG' if sig('rural_piso') else 'no sig'}")
    print(f"  tribal: binario {'SIG' if sig('tribal') else 'no sig'}; "
          f">=10km {'SIG' if sig('tribal_e10') else 'no sig'} | "
          f"SVI alto: binario {'SIG' if sig('svi') else 'no sig'}; "
          f">=10km {'SIG' if sig('svi_e10') else 'no sig'}")
    print(f"  cota inferior: error={100*nat.error_lb.mean():5.1f}%  "
          f"extra_pond={np.average(nat.km_extra_lb, weights=np.maximum(pop,1)):4.2f} km  "
          f">=10km rural {100*nat[nat.rural].e10_lb.mean():4.1f}% vs urbano "
          f"{100*nat[~nat.rural].e10_lb.mean():4.1f}%")
    print("  la probabilidad de error NO varia por ruralidad; lo que varia es el dano cuando")
    print("  ocurre. Se reporta la precision y el error global como hallazgo, y la cola como")
    print("  agravante sin significancia estadistica propia.")

    # ---------------- 10. tabla rural verificada
    print("\n=== 10. TABLA RURAL: PEOR KM EXTRA EN CONDADOS DISTINTOS, VERIFICADO EN LA WEB ===")
    print("  Verificacion web del 2026-10-31. En estos tractos NO se excluye el desierto")
    print("  hospitalario: ahi el hueco es real y es el caso que ilustra la disparidad.")
    rural = []
    for geoid, v in VERIF_RURAL.items():
        row = nat[nat.GEOID == geoid]
        if row.empty or not bool(row.iloc[0].error):
            continue
        r = row.iloc[0]
        rural.append((r, v))
        print(f"  {v['lugar']:<11}{v['condado']:<18} tracto {r.GEOID} pop={int(r.pop_total or 0):>6,} "
              f"| POI '{str(r.poi_nombre)[:38]}' {r.poi_km:.1f} km -> '{str(r.real_nombre)[:34]}' "
              f"{r.real_km:.1f} km (+{r.km_extra:.1f})")
        print(f"      {v['nota']}")
    rural.sort(key=lambda t: -t[0].km_extra)
    rural = rural[:10]
    print(f"  lugares verificados: {len(rural)} en {len({v['condado'] for _, v in rural})} condados")

    # ---------------- escrituras
    hc_cols = ["id", "pnom", "pcat", "tax", "confidence", "region", "tracto", "tracto_nombre",
               "condado", "condado_nombre", "clase", "match_er", "anexo", "looks",
               "tiene_urgencias", "defB", "defC", "etiqueta", "ref_nombre", "ref_km",
               "rural", "tribal", "svi_alto", "plat", "plon"]
    h[hc_cols].sort_values(["region", "clase", "pnom"]).round(
        {"confidence": 4, "ref_km": 3, "plat": 6, "plon": 6}).to_csv(
        CSV_POI, index=False, encoding="utf-8-sig")
    tc = ["GEOID", "region", "tracto_nombre", "condado", "condado_nombre", "pop_total", "pct_urban",
          "svi_overall", "tribal_pct", "rural", "tribal", "svi_alto", "poi_nombre", "poi_clase",
          "poi_urgencias", "poi_etiqueta", "poi_km", "poi_conf", "real_nombre", "real_tipo",
          "real_km", "any_km", "km_extra", "error", "ratio", "e10", "e20", "e30",
          "error_piso", "extra_piso"]
    nat[tc].sort_values(["region", "GEOID"]).round(
        {"pct_urban": 3, "svi_overall": 3, "tribal_pct": 3, "poi_km": 3, "real_km": 3,
         "km_extra": 3, "poi_conf": 4, "ratio": 3}).to_csv(CSV_TRACT, index=False,
                                                           encoding="utf-8-sig")
    print(f"\n[out] {CSV_POI} filas={len(h)}")
    print(f"[out] {CSV_TRACT} filas={len(nat)}")

    SAL.write_tex_precision(h, nat)
    SAL.write_tex_severity(h, nat, res)
    SAL.write_tex_tracts(top, VERIF_URB)
    SAL.write_tex_rural(rural)
    SAL.write_fig(h, nat)
    print(f"[out] {os.path.join(TEX_TAB, 'hospital_precision.tex')}")
    print(f"[out] {os.path.join(TEX_TAB, 'hospital_severity.tex')}")
    print(f"[out] {os.path.join(TEX_TAB, 'hospital_tracts.tex')}")
    print(f"[out] {os.path.join(TEX_TAB, 'hospital_rural.tex')}")
    print(f"[out] {os.path.join(TEX_FIG, 'hospital_error.pdf')}")

    print("\n=== DUDAS (decisiones conservadoras tomadas sin preguntar) ===")
    print("  - VA: solo el VAMC ('... VA MEDICAL CENTER') cuenta como urgencias; CBOC y Vet Center no.")
    print("  - DoD/militar (Reynolds Army Hospital, Darnall): en el piso conservador cuentan como")
    print("    hospital por nombre; en la base solo si CMS los lista con urgencias.")
    print("  - Hospital real sin urgencias (psiquiatrico, LTAC, rehab) cuenta en B y no en A.")
    print("  - Un anexo (consulta, imagen, heridas, unidad, departamento) nunca cuenta como el")
    print("    hospital: es la causa medida de los 'reales' que no lo eran en la primera lectura.")
    print("  - El flag de CMS no manda solo: CAH y REH entran por normativa y los agudos con No")
    print("    entran por nombre. La cota inferior (todo agudo/CAH/REH/VA/DoD/FSER) se reporta aparte.")
    print("  - El listado de FSER de Texas es del 2020-07-01: un ER autonomo abierto despues no esta")
    print("    en la referencia, y por eso se descartaron los suburbanos con FSER nuevo.")


if __name__ == "__main__":
    main()
