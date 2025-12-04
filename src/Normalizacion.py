"""
Módulo de normalización de ortomosaicos
=======================================

Incluye funciones para la normalización espacial y radiométrica para un par MS/RGB.

Principios de Diseño:
--------------------
1. Alineación Intra-sesión: El RGB se alinea al molde espacial del MS de su mismo vuelo.
2. Modularidad: Las funciones de alineación y normalización radiométrica son genéricas.
3. El módulo solo procesa arrays y perfiles de Rasterio ya cargados.
"""

# Imports
import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

# Normalización espacial
def align_to_reference(target_array: np.ndarray, ref_profile: dict,
                       src_profile: dict, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """
    Re-muestrea un ortomosaico (target_array) para alinearlo a la grilla de referencia (ref_profile).

    Parámetros
    ----------
    target_array : np.ndarray
        Array que se quiere alinear.
    ref_profile : dict
        Perfil de referencia obtenido del archivo MS.
    src_profile : dict
        Perfil espacial original del target_array.
    resampling : rasterio.enums.Resampling
        Método de re-muestreo. 'bilinear' es apropiado.

    Retorna
    -------
    np.ndarray
        Array alineado al sistema de referencia, o None si hay problemas.
    """
    if target_array is None:
        return None

    # Usamos MemoryFile para crear un dataset temporal en RAM, evitando archivos en disco (más rápido).
    try:
        with rasterio.io.MemoryFile() as memfile:
            # Definimos el perfil temporal con los datos del array de origen
            temp_profile = src_profile.copy()
            temp_profile.update(dtype=target_array.dtype, count=target_array.shape[0], 
                                height=target_array.shape[1], width=target_array.shape[2])
            
            # Escribimos el array de origen (RGB) al dataset en memoria
            with memfile.open(**temp_profile) as dst:
                dst.write(target_array)
                
            # Abrimos el dataset en memoria como origen para WarpedVRT
            with memfile.open(**temp_profile) as src:
                # Definimos el "molde" de destino (el MS profile)
                vrt_opts = dict(
                    crs=ref_profile["crs"],
                    transform=ref_profile["transform"],
                    width=ref_profile["width"],
                    height=ref_profile["height"],
                    resampling=resampling
                )
                
                # Aplicamos la alineación y leemos el resultado
                with WarpedVRT(src, **vrt_opts) as vrt:
                    data = vrt.read(out_dtype="float32")
                return data
                
    except Exception as e:
        print(f"Error durante la alineación espacial: {e}")
        return None


# Normalización radiométrica
def normalize_radiometric(image: np.ndarray) -> np.ndarray:
    """
    Escala los valores de un ortomosaico al rango [0, 1] según su profundidad de bits.

    Retorna un array float32 listo para análisis.
    """
    if image is None:
        return None

    arr = image.astype("float32")

    if np.issubdtype(image.dtype, np.uint8):
        arr /= 255.0
    elif np.issubdtype(image.dtype, np.uint16): 
        arr /= 65535.0
    else:
        # Para floats ya cargados, normalizamos por el máximo.
        max_val = np.nanmax(arr)
        if max_val > 0:
            arr /= max_val

    return arr


def normalize_all(aligned_data: dict) -> dict:
    """
    Aplica la normalización radiométrica ([0, 1]) a los arrays MS y RGB alineados.
    """
    normalized = {}
    
    # Normalizamos el MS
    normalized["ms"] = normalize_radiometric(aligned_data["ms_data"])
    
    # Normalizamos el RGB (si existe)
    if aligned_data.get("rgb_aligned") is not None:
        normalized["rgb"] = normalize_radiometric(aligned_data["rgb_aligned"])
    else:
        normalized["rgb"] = None

    return normalized

# =============================================================================
# FLUJO PRINCIPAL DE SESIÓN ÚNICA
# =============================================================================

def process_session(ms_data: np.ndarray, ms_profile: dict, 
                    rgb_data: np.ndarray, rgb_profile: dict) -> dict:
    """
    Función maestra que aplica toda la normalización (Espacial y Radiométrica) 
    para una única sesión (par MS/RGB).

    Parámetros:
    ----------
    ms_data, rgb_data : np.ndarray
        Arrays de la imagen Multiespectral y RGB.
    ms_profile, rgb_profile : dict
        Perfiles de rasterio de ambas imágenes.

    Retorna:
    --------
    dict: Diccionario con claves 'ms' y 'rgb' con arrays normalizados y alineados.
    """
    
    # 1. Normalización Espacial (Alineación RGB -> MS)
    # -------------------------------------
    print("Iniciando Normalización Espacial (Alineación RGB -> MS)...")
    
    # Alineamos el RGB (target) al perfil del MS (referencia/molde).
    rgb_aligned = align_to_reference(
        target_array=rgb_data, 
        ref_profile=ms_profile,
        src_profile=rgb_profile, # Perfil original del RGB
        resampling=Resampling.bilinear
    )

    if rgb_aligned is None:
        print("❌ Alineación espacial fallida.")
        return {"ms": None, "rgb": None}
    
    print("✅ Alineación espacial completada.")

    # 2. Normalización Radiométrica
    # -----------------------------
    print("Iniciando Normalización Radiométrica ([0, 1])...")
    
    aligned_data = {
        # El MS ya está en el sistema de coordenadas de referencia, solo se normaliza radiométricamente.
        "ms_data": ms_data,       
        "rgb_aligned": rgb_aligned
    }
    
    # Aplicamos la normalización [0, 1] a ambos
    normalized = normalize_all(aligned_data)
    
    print("✅ Normalización radiométrica completada.")

    return normalized