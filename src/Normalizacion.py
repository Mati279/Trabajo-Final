"""
Módulo de normalización de ortomosaicos
=======================================

Incluye funciones para la normalización **espacial** y **radiométrica** de ortomosaicos UAV.

Etapas:
--------
1. Normalización espacial (alineación)
   - Usa como referencia el ortomosaico multiespectral (MS) del 17/ene.
   - Ajusta todas las imágenes (RGB, RED, NIR, MS) a la misma grilla (CRS, resolución, tamaño).
   - Implementa la función `align_to_reference()` basada en rasterio.WarpedVRT.

2. Normalización radiométrica
   - Escala los valores digitales de cada banda al rango [0, 1].
   - Ajusta automáticamente según la profundidad de bits (8, 16 o float).
   - Implementa la función `normalize_radiometric()`.

El módulo asume que las rutas de los ortomosaicos se obtienen con
`load_orthomosaics(date, load_arrays=False)` del módulo Ortomosaicos.py.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from Ortomosaicos import load_orthomosaics

# =============================================================================
# NORMALIZACIÓN ESPACIAL
# =============================================================================

def get_reference_profile(date: str = "17ene") -> dict:
    """
    Obtiene la grilla de referencia desde el ortomosaico multiespectral (MS) del 17/ene.

    Retorna:
    --------
    dict : perfil de rasterio con CRS, transform, ancho, alto, dtype, etc.
    """
    paths = load_orthomosaics(date, load_arrays=False)
    ref_path = paths["ms"]

    assert isinstance(ref_path, str) and os.path.exists(ref_path), \
        "No se encontró la ruta del MS 17/ene."

    with rasterio.open(ref_path) as ref_src:
        ref_profile = ref_src.profile.copy()
        print("Referencia espacial:", os.path.basename(ref_path))
        print(f"CRS: {ref_src.crs}\nDims: {ref_src.height} x {ref_src.width}")

    return ref_profile


def align_to_reference(path_tif: str, ref_profile: dict,
                       resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Re-muestrea un ortomosaico para alinearlo a la grilla de referencia.

    Parámetros
    ----------
    path_tif : str
        Ruta al archivo .tif que se quiere alinear.
    ref_profile : dict
        Perfil de referencia (REF_PROFILE) obtenido del MS 17/ene.
    resampling : rasterio.enums.Resampling
        Método de re-muestreo. 'bilinear' es apropiado para datos continuos (imágenes).

    Retorna
    -------
    np.ndarray
        Array (bandas, alto, ancho) alineado al sistema de referencia.
    """
    if path_tif is None or not os.path.exists(path_tif):
        return None

    with rasterio.open(path_tif) as src:
        vrt_opts = dict(
            crs=ref_profile["crs"],
            transform=ref_profile["transform"],
            width=ref_profile["width"],
            height=ref_profile["height"],
            resampling=resampling
        )
        with WarpedVRT(src, **vrt_opts) as vrt:
            data = vrt.read(out_dtype="float32")
    return data


def align_all_campaigns(datasets: dict, ref_profile: dict) -> dict:
    """
    Alinea todos los ortomosaicos de las distintas fechas a la grilla del MS 17/ene.

    Parámetros
    ----------
    datasets : dict
        Diccionario con las fechas como claves ('10ene', '17ene', '24ene', ...)
        y subdiccionarios con rutas a los ortomosaicos.
    ref_profile : dict
        Perfil de referencia (del MS 17/ene).

    Retorna
    -------
    dict : diccionario 'aligned' con arrays float32 alineados.
    """
    aligned = {fecha: {} for fecha in datasets.keys()}

    for fecha in datasets.keys():
        paths = load_orthomosaics(fecha, load_arrays=False)
        for tipo in ["rgb", "red", "nir", "ms"]:
            aligned[fecha][tipo] = align_to_reference(
                paths.get(tipo),
                ref_profile,
                resampling=Resampling.bilinear
            )

    return aligned

# =============================================================================
# NORMALIZACIÓN RADIOMÉTRICA
# =============================================================================

def normalize_radiometric(image: np.ndarray) -> np.ndarray:
    """
    Escala los valores de un ortomosaico al rango [0, 1] según su tipo de dato.

    Retorna un array float32 listo para análisis o visualización coherente.
    """
    if image is None:
        return None

    arr = image.astype("float32")

    if np.issubdtype(image.dtype, np.uint8):
        arr /= 255.0
    elif np.issubdtype(image.dtype, np.uint16):
        arr /= 65535.0
    else:
        max_val = np.nanmax(arr)
        if max_val > 0:
            arr /= max_val

    return arr


def normalize_all(aligned: dict) -> dict:
    """
    Aplica la normalización radiométrica a todos los ortomosaicos de 'aligned'.

    Retorna:
    --------
    dict : nuevo diccionario 'normalized' con valores en [0, 1].
    """
    normalized = {fecha: {} for fecha in aligned.keys()}

    for fecha in aligned.keys():
        for tipo in ["rgb", "red", "nir", "ms"]:
            img = aligned[fecha][tipo]
            normalized[fecha][tipo] = normalize_radiometric(img)

    return normalized
