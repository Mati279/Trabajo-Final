"""
Módulo de normalización de ortomosaicos

Incluye funciones para la normalización espacial y radiométrica para un par MS/RGB.
"""

# Imports
import os
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

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
        Método de re-muestreo. 'bilinear' es apropiado porque es continuo.

    Retorna
    np.ndarray
        Array alineado al sistema de referencia, o None si hay problemas.
    """
    if target_array is None:
        return None

    # Usamos MemoryFile para crear un dataset temporal en RAM.
    try:
        with MemoryFile() as memfile:
            # Definimos el perfil temporal con los datos del array de origen
            temp_profile = src_profile.copy()
            temp_profile.update(dtype=target_array.dtype, count=target_array.shape[0], # Modifica las keys.
                                height=target_array.shape[1], width=target_array.shape[2])
            
            # Escribimos el array de origen al dataset en memoria
            with memfile.open(**temp_profile) as dst:
                dst.write(target_array)
                
            # Abrimos el dataset en memoria como origen para WarpedVRT
            with memfile.open(**temp_profile) as src:
                # Definimos el molde
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
        return None # type: ignore


# Normalización radiométrica
def normalize_radiometric(image: np.ndarray) -> np.ndarray:
    """
    Escala los valores de un ortomosaico al rango [0, 1].

    Retorna un array float32.
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
    Aplica la normalización radiométrica a los arrays MS y RGB alineados.
    """
    normalized = {}
    
    # Normaliza el MS
    normalized["ms"] = normalize_radiometric(aligned_data["ms_data"])
    
    # Normaliza el RGB
    if aligned_data.get("rgb_aligned") is not None:
        normalized["rgb"] = normalize_radiometric(aligned_data["rgb_aligned"])
    else:
        normalized["rgb"] = None

    return normalized


# Función maestra
def process_session(ms_data: np.ndarray, ms_profile: dict, 
                    rgb_data: np.ndarray, rgb_profile: dict) -> dict:
    """
    Función maestra que aplica toda la normalización (Espacial y Radiométrica) 
    para una única sesión (par MS/RGB).

    Parámetros:
    ms_data, rgb_data : np.ndarray
        Arrays de la imagen Multiespectral y RGB.
    ms_profile, rgb_profile : dict
        Perfiles de rasterio de ambas imágenes.

    Retorna:
    dict: Diccionario con claves 'ms' y 'rgb' con arrays normalizados y alineados.
    """
    
    # Normalización Espacial
    rgb_aligned = align_to_reference(
        rgb_data, 
        ms_profile,
        rgb_profile, 
        Resampling.bilinear
    )

    if rgb_aligned is None:
        print("Alineación espacial fallida.")
        return {"ms": None, "rgb": None}
    
    print("Alineación espacial completada.")

    # Normalización Radiométrica
    print("Iniciando Normalización Radiométrica")
    
    aligned_data = {
        "ms_data": ms_data,       
        "rgb_aligned": rgb_aligned
    }
    
    # Aplicamos la normalización [0, 1] a ambos
    normalized = normalize_all(aligned_data)
    
    print("Normalización radiométrica completada.")

    return normalized