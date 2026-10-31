"""Respaldo de geocodificación para los hospitales CMS que el Census Geocoder no resolvió.

Consulta Nominatim (OpenStreetMap, ODbL) con la dirección de CMS, una petición por segundo, y guarda el resultado en
datos/externos/cms/hosp_geo_nominatim.csv. l9_referencia.py lo usa antes de caer al centroide de la ciudad.
Con el archivo presente no se toca la red. Uso: python procesos/bias_discovery/l9_geo_respaldo.py
"""
import csv
import json
import os
import subprocess
import time
import urllib.parse

import l9_referencia as L

SAL = os.path.join(L.EXT_CMS, "hosp_geo_nominatim.csv")
UA = "bias-bounty-mapping-equity reproducibility (hospital reference geocoding)"


def consulta(q):
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        dict(q=q, format="jsonv2", limit=1, countrycodes="us"))
    r = subprocess.run(["curl", "-s", "--max-time", "40", "-A", UA, url], capture_output=True)
    try:
        return json.loads(r.stdout.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []


def main():
    c = L.geocode_cms(L.load_cms())
    falta = c[c.geo_prec.ne("direccion")]
    hecho = {}
    if os.path.exists(SAL):
        hecho = {r["facility_id"]: r for r in csv.DictReader(open(SAL, encoding="utf-8"))}
    with open(SAL, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["facility_id", "lat", "lon", "consulta"])
        if not hecho:
            w.writeheader()
        for r in falta.itertuples():
            if r.facility_id in hecho:
                continue
            q = f"{r.address}, {r.citytown}, {r.state} {r.zip5}"
            m = consulta(q) or consulta(f"{r.facility_name}, {r.citytown}, {r.state}")
            w.writerow(dict(facility_id=r.facility_id, lat=m[0]["lat"] if m else "", lon=m[0]["lon"] if m else "", consulta=q))
            f.flush()
            time.sleep(1.1)


if __name__ == "__main__":
    main()
