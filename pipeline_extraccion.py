"""
pipeline_extraccion.py

Versión del notebook lista para correr fuera de Colab (en un GitHub Action).
Reemplaza drive.mount() por autenticación con una cuenta de servicio de
Google, usando la API de Drive para navegar la carpeta por nombre y
descargar solo los archivos .nc de la ventana de días necesaria.

Variables de entorno esperadas:
  GCP_SERVICE_ACCOUNT_KEY   contenido completo del JSON de la cuenta de servicio
"""

import os
import io
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import gspread

# ---------------------------------------------------------------------------
# Configuración (ajusta esto a tu caso si algo no calza)
# ---------------------------------------------------------------------------

# Ruta de carpetas dentro de Drive, tal como la ves en la interfaz web,
# empezando por la carpeta compartida por GloH2O. Antes era
# "/content/drive/MyDrive/PMA_CIAT/..."; aquí son los mismos nombres de
# carpeta, solo que navegados por la API en vez de por sistema de archivos.
RUTA_BASE = ["PMA_CIAT"]  # ajusta si el nombre real de la carpeta es otro
CARPETA_MSWEP = RUTA_BASE + ["MSWEP_V280"]
CARPETA_MSWX = RUTA_BASE + ["MSWX_V100"]
CARPETA_GEOJSON = RUTA_BASE  # se asume que gadm41_COL_2.json vive junto a las otras dos

DIAS_DESCARGA = 30
LOCAL_CACHE = "cache_nc"
os.makedirs(LOCAL_CACHE, exist_ok=True)

MUNICIPIOS = {
    "Los Palmitos (Sucre)":          {"name_2": "LosPalmitos", "name_1": "Sucre"},
    "San José de Toluviejo (Sucre)": {"name_2": "Toluviejo",   "name_1": "Sucre"},
    "Icononzo (Tolima)":             {"name_2": "Icononzo",    "name_1": "Tolima"},
    "Planadas (Tolima)":             {"name_2": "Planadas",    "name_1": "Tolima"},
}

VARIABLES = {
    "precip": {"carpeta": CARPETA_MSWEP + ["NRT", "Daily"], "var_candidates": ["precipitation", "precip", "P", "pr"]},
    "tavg":   {"carpeta": CARPETA_MSWX + ["NRT", "Temp", "Daily"], "var_candidates": ["air_temperature", "temperature", "Temp", "tas", "T"]},
    "tmin":   {"carpeta": CARPETA_MSWX + ["NRT", "Tmin", "Daily"], "var_candidates": ["air_temperature", "temperature", "Tmin", "tasmin"]},
    "tmax":   {"carpeta": CARPETA_MSWX + ["NRT", "Tmax", "Daily"], "var_candidates": ["air_temperature", "temperature", "Tmax", "tasmax"]},
    "relhum": {"carpeta": CARPETA_MSWX + ["NRT", "RelHum", "Daily"], "var_candidates": ["relative_humidity", "RelHum", "rh", "hurs"]},
}


# ---------------------------------------------------------------------------
# Autenticación y navegación de Drive por nombre de carpeta
# ---------------------------------------------------------------------------

def autenticar_drive():
    info = json.loads(os.environ["GCP_SERVICE_ACCOUNT_KEY"])
    creds = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/spreadsheets.readonly",
        ],
    )
    return build("drive", "v3", credentials=creds), creds


def resolver_destino_si_es_atajo(servicio, archivo):
    """Si el archivo es un acceso directo (shortcut), sigue el enlace y
    devuelve el id de la carpeta/archivo real al que apunta."""
    if archivo.get("mimeType") == "application/vnd.google-apps.shortcut":
        detalle = servicio.files().get(
            fileId=archivo["id"], fields="shortcutDetails", supportsAllDrives=True
        ).execute()
        return detalle["shortcutDetails"]["targetId"]
    return archivo["id"]


def buscar_id_por_ruta(servicio, partes_ruta, id_padre="root"):
    """Camina la ruta de carpetas por nombre (como en el Explorador de Drive),
    siguiendo accesos directos (shortcuts) cuando los encuentra, y devuelve
    el id de la última carpeta. El primer nivel se busca sin restringir por
    carpeta padre, porque una carpeta compartida directamente con la cuenta
    de servicio no cuelga de su propio 'root'."""
    actual = id_padre
    for nombre in partes_ruta:
        if actual == "root":
            query = f"name = '{nombre}' and trashed = false"
        else:
            query = f"name = '{nombre}' and '{actual}' in parents and trashed = false"
        resultado = servicio.files().list(
            q=query, fields="files(id, name, mimeType)", supportsAllDrives=True, includeItemsFromAllDrives=True
        ).execute()
        candidatos = [
            f for f in resultado.get("files", [])
            if f.get("mimeType") in ("application/vnd.google-apps.folder", "application/vnd.google-apps.shortcut")
        ]
        if not candidatos:
            # Diagnóstico: qué hay REALMENTE ahí, para no tener que adivinar
            q_hijos = f"'{actual}' in parents and trashed = false" if actual != "root" else "trashed = false"
            hijos = servicio.files().list(
                q=q_hijos, fields="files(name, mimeType)", supportsAllDrives=True,
                includeItemsFromAllDrives=True, pageSize=50,
            ).execute().get("files", [])
            disponibles = [f"{h['name']} ({h['mimeType'].rsplit('.', 1)[-1]})" for h in hijos]
            raise FileNotFoundError(
                f"No se encontró '{nombre}' dentro de la ruta {partes_ruta}. "
                f"Lo que sí hay ahí: {disponibles}"
            )
        actual = resolver_destino_si_es_atajo(servicio, candidatos[0])
    return actual


def listar_archivos_en_carpeta(servicio, carpeta_id):
    """Devuelve {nombre_archivo: file_id_real} de una carpeta (una sola
    página; si tienes miles de archivos por carpeta, hay que paginar con
    pageToken). Si algún archivo es en realidad un acceso directo, sigue
    el enlace y guarda el id del archivo real, no el del acceso directo."""
    archivos = {}
    page_token = None
    while True:
        resultado = servicio.files().list(
            q=f"'{carpeta_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        for f in resultado.get("files", []):
            archivos[f["name"]] = resolver_destino_si_es_atajo(servicio, f)
        page_token = resultado.get("nextPageToken")
        if not page_token:
            break
    return archivos


def descargar_archivo(servicio, file_id, destino_local):
    if os.path.exists(destino_local) and os.path.getsize(destino_local) > 10_000:
        return destino_local
    request = servicio.files().get_media(fileId=file_id)
    with io.FileIO(destino_local, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        listo = False
        while not listo:
            _, listo = downloader.next_chunk()
    tamano = os.path.getsize(destino_local)
    if tamano < 10_000:  # un .nc diario real pesa MBs; esto casi seguro es un archivo dañado o vacío
        raise IOError(
            f"El archivo descargado '{destino_local}' pesa solo {tamano} bytes — "
            "probablemente Drive no entregó el contenido real (revisa si sigue siendo un acceso "
            "directo sin resolver, o si el archivo original está vacío)."
        )
    return destino_local


# ---------------------------------------------------------------------------
# Ventana de fechas y descarga de la variable
# ---------------------------------------------------------------------------

HOY = datetime.utcnow().date()


def fechas_ventana(dias=DIAS_DESCARGA):
    return [HOY - timedelta(days=i) for i in range(dias, -1, -1)]


def nombre_archivo_diario(fecha):
    doy = fecha.timetuple().tm_yday
    return f"{fecha.year}{doy:03d}.nc"


def descargar_ventana_variable(servicio, var_key, cfg):
    carpeta_id = buscar_id_por_ruta(servicio, cfg["carpeta"])
    archivos_drive = listar_archivos_en_carpeta(servicio, carpeta_id)

    rutas_locales, faltantes = [], []
    destino_dir = os.path.join(LOCAL_CACHE, var_key)
    os.makedirs(destino_dir, exist_ok=True)

    for fecha in fechas_ventana():
        nombre = nombre_archivo_diario(fecha)
        if nombre not in archivos_drive:
            faltantes.append(nombre)
            continue
        destino = os.path.join(destino_dir, nombre)
        descargar_archivo(servicio, archivos_drive[nombre], destino)
        rutas_locales.append(destino)

    if faltantes:
        print(f"⚠️  {var_key}: {len(faltantes)} archivo(s) no encontrados "
              f"(el NRT puede tener 1-3 días de rezago): {faltantes[:5]}")
    return sorted(rutas_locales)


def nombre_variable_real(ds, candidatos):
    for c in candidatos:
        if c in ds.data_vars:
            return c
    return list(ds.data_vars)[0]


def abrir_dataset(archivos):
    # parallel=True abre los .nc con varios hilos a la vez vía dask, pero la
    # librería HDF5 detrás de netCDF4 no siempre es segura para eso y puede
    # colgarse o reventar (segmentation fault) sin ni siquiera un error de
    # Python. En serie es un poco más lento, pero confiable.
    return xr.open_mfdataset(archivos, combine="nested", concat_dim="time", parallel=False)


def extraer_serie_punto(archivos, var_candidatos, lat, lon):
    valores, fechas, var_real = [], [], None
    for archivo in archivos:
        with xr.open_dataset(archivo) as ds:
            if var_real is None:
                var_real = nombre_variable_real(ds, var_candidatos)
            punto = ds[var_real].sel(lat=lat, lon=lon, method="nearest")
            valores.append(float(punto.values.squeeze()))
            fechas.append(pd.Timestamp(ds["time"].values[0]))
    return pd.Series(valores, index=pd.DatetimeIndex(fechas)).sort_index()



# ---------------------------------------------------------------------------
# Geometrías de los 4 municipios (archivo estático del repo, no cambia
# entre corridas — no hace falta bajarlo de Drive cada vez)
# ---------------------------------------------------------------------------

import geopandas as gpd
import rioxarray  # noqa: F401 - habilita el accessor .rio sobre xarray

RUTA_GEOJSON_MUNICIPIOS = "municipios_monitoreados.json"  # generado antes, vive en el repo


def cargar_geometrias():
    gdf = gpd.read_file(RUTA_GEOJSON_MUNICIPIOS).set_index("nombre")
    gdf_proj = gdf.to_crs(6933)  # área equivalente, para centroides correctos
    return gdf, gdf_proj


def centroide(gdf_proj, nombre):
    punto = gdf_proj.loc[nombre, "geometry"].centroid
    # el centroide se calculó en 6933 (metros); se reproyecta a lat/lon
    punto_geo = gpd.GeoSeries([punto], crs=6933).to_crs(4326).iloc[0]
    return punto_geo.y, punto_geo.x  # lat, lon


# ---------------------------------------------------------------------------
# Recorte raster (grilla espacial, solo para precipitación por ahora)
# ---------------------------------------------------------------------------

def construir_grid_precipitacion(archivos_precip, gdf, nombre):
    ds = abrir_dataset(archivos_precip)
    ds = ds.rio.write_crs("EPSG:4326")
    if "lat" in ds.dims and "lon" in ds.dims:
        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    geometria = gdf.loc[nombre, "geometry"]
    recorte = ds.rio.clip([geometria], all_touched=True, drop=True)
    var_real = nombre_variable_real(recorte, VARIABLES["precip"]["var_candidates"])

    df = recorte[var_real].to_dataframe(name="precip").reset_index().dropna(subset=["precip"])
    celdas = []
    for (lat, lon), grupo in df.groupby(["lat", "lon"]):
        grupo = grupo.sort_values("time")
        celdas.append({
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "valores": [round(float(v), 1) for v in grupo["precip"].tolist()],
        })
    return celdas


def serie_a_lista(serie):
    return [{"date": fecha.strftime("%Y-%m-%d"), "value": round(float(valor), 1)} for fecha, valor in serie.items()]


# ---------------------------------------------------------------------------
# Reportes de campo (Google Sheets que diligencian los productores)
# ---------------------------------------------------------------------------

SHEET_ID_REPORTES = "1XA1t4_6NZdrORj91GW7I0ZORgggrUNNu72kitz7pdtk"  # Hoja "Reportes de campo PMA"

VARIABLE_SHEET_A_CLAVE = {
    "Lluvia (mm)": "precip",
    "Temperatura (°C)": "tavg",
    "Humedad relativa (%)": "relhum",
}


def leer_reportes_de_campo(creds, sheet_id):
    """Lee la hoja de Google Sheets que diligencian los productores/técnicos
    cada semana y la convierte al mismo formato que ya usa el dashboard
    (datos.observaciones). Filas incompletas o con un municipio/variable que
    no calza exactamente con lo esperado se ignoran silenciosamente (se
    listan al final para que sea fácil detectar un typo en la hoja)."""
    gc = gspread.authorize(creds)
    hoja = gc.open_by_key(sheet_id).sheet1
    filas = hoja.get_all_records()

    observaciones, ignoradas = [], 0
    for fila in filas:
        variable = VARIABLE_SHEET_A_CLAVE.get(str(fila.get("variable", "")).strip())
        municipio = str(fila.get("municipio", "")).strip()
        fecha = str(fila.get("fecha", "")).strip()
        try:
            valor = float(fila.get("valor"))
        except (TypeError, ValueError):
            valor = None

        if not variable or municipio not in MUNICIPIOS or not fecha or valor is None:
            ignoradas += 1
            continue

        observaciones.append({
            "municipio": municipio,
            "fecha": fecha,
            "variable": variable,
            "valor": round(valor, 1),
            "comentario": str(fila.get("comentario", "")).strip(),
        })

    if ignoradas:
        print(f"⚠️  {ignoradas} fila(s) del Sheet de reportes se ignoraron por estar incompletas o con un valor inesperado.")
    return observaciones


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main():
    servicio, creds = autenticar_drive()
    gdf, gdf_proj = cargar_geometrias()

    # Descarga cada variable una sola vez (no una vez por municipio)
    archivos_por_variable = {}
    for var_key, cfg in VARIABLES.items():
        print(f"Descargando ventana de '{var_key}'...")
        archivos_por_variable[var_key] = descargar_ventana_variable(servicio, var_key, cfg)

    municipios_json = {}
    for nombre in MUNICIPIOS:
        lat, lon = centroide(gdf_proj, nombre)
        print(f"{nombre}: centroide lat={lat:.4f}, lon={lon:.4f}")

        series = {}
        for var_key, cfg in VARIABLES.items():
            archivos = archivos_por_variable[var_key]
            if not archivos:
                print(f"⚠️  Sin archivos para '{var_key}', se omite en {nombre}.")
                continue
            series[var_key] = serie_a_lista(
                extraer_serie_punto(archivos, cfg["var_candidates"], lat, lon)
            )

        municipios_json[nombre] = {
            **series,
            "grid": construir_grid_precipitacion(archivos_por_variable["precip"], gdf, nombre)
                    if archivos_por_variable.get("precip") else [],
        }
        print(f"{nombre}: grilla con {len(municipios_json[nombre]['grid'])} celdas")

    print("Leyendo reportes de campo...")
    try:
        observaciones = leer_reportes_de_campo(creds, SHEET_ID_REPORTES)
        print(f"{len(observaciones)} reporte(s) de campo cargados.")
    except Exception as e:
        print(f"⚠️  No se pudieron leer los reportes de campo (¿ya compartiste la hoja con la cuenta de servicio?): {e}")
        observaciones = []

    data_json = {
        "generado_en": datetime.utcnow().isoformat(),
        "municipios": municipios_json,
        "observaciones": observaciones,
        "enso": {
            "valor": 1.9,  # TODO: automatizar con la tabla ONI del NOAA CPC más adelante
            "categoria": "El Niño fuerte",
            "actualizado": datetime.utcnow().strftime("%d %b %Y"),
            "fuente_url": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml",
        },
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False)
    print("\ndata.json listo:", os.path.getsize("data.json"), "bytes")


if __name__ == "__main__":
    main()

