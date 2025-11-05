"""
Módulo de carga y visualización de ortomosaicos
===============================================

Permite cargar automáticamente todos los ortomosaicos
de una fecha dada ('10ene', '17ene', '24ene') y visualizarlos.

Uso:
    from Ortomosaicos import load_orthomosaics, show_orthomosaic

    ortho = load_orthomosaics("17ene")
    show_orthomosaic(ortho["rgb"], title="RGB - 17ene")
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


def load_orthomosaics(date: str, load_arrays: bool = True) -> dict:
    """
    Carga todos los ortomosaicos disponibles para una fecha.

    Parámetros:
    -----------
    date : str
        Fecha/campaña (ej.: '10ene', '17ene', '24ene').
    load_arrays : bool
        Si True, abre cada ortomosaico como array (con rasterio).

    Retorna:
    --------
    dict
        Diccionario con claves ['rgb', 'ms', 'red', 'nir', 'dsm', 'cloud']
        y valores con arrays o rutas según load_arrays.
    """
    base_dir = rf"D:\Programas Python\Trabajo Final\Trabajo-Final\data\Material\Ortomosaicos\{date}"

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"No se encontró la carpeta: {base_dir}")

    file_map = {
        "rgb": f"estanzuela_{date}_rgb_orthophoto.tif",
        "ms":  f"estanzuela_{date}_MS_orthophoto.tif",
        "red": f"estanzuela_{date}_RED_orthophoto.tif",
        "nir": f"estanzuela_{date}_NIR_orthophoto.tif",
        "dsm": f"Octavio-Vergara-1-{date}-dsm.tif",
        "cloud": f"Octavio-Vergara-1-{date}-georeferenced_model.laz"
    }

    orthos = {}

    for key, filename in file_map.items():
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            if load_arrays and path.endswith(".tif"):
                with rasterio.open(path) as src:
                    orthos[key] = src.read()
            else:
                orthos[key] = path
        else:
            orthos[key] = None

    return orthos


def show_orthomosaic(orthomosaic, title: str = None):
    """
    Visualiza un ortomosaico (array o ruta .tif).

    Parámetros:
    -----------
    orthomosaic : np.ndarray | str
        Array de Rasterio o ruta a un archivo .tif.
    title : str, opcional
        Título del gráfico.
    """
    # Cargar datos
    if isinstance(orthomosaic, str) and os.path.exists(orthomosaic):
        with rasterio.open(orthomosaic) as src:
            data = src.read()
    elif isinstance(orthomosaic, np.ndarray):
        data = orthomosaic
    else:
        raise ValueError("El argumento debe ser un array o una ruta válida a un .tif")

    plt.figure(figsize=(8, 8))

    # --- Mostrar RGB o monocanal ---
    if data.shape[0] >= 3:
        if data.shape[0] == 4:
            data = data[:3]
        rgb_img = np.transpose(data, (1, 2, 0)).astype(np.float32) / np.max(data)
        plt.imshow(rgb_img)
    elif data.shape[0] == 1:
        plt.imshow(data[0], cmap="gray")
    else:
        plt.imshow(data[0], cmap="viridis")

    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def list_orthomosaics(date: str) -> None:
    """
    Lista los ortomosaicos disponibles en la fecha indicada.

    Ejemplo:
        >>> list_orthomosaics("17ene")
    """
    base_dir = rf"D:\Programas Python\Trabajo Final\Trabajo-Final\data\Material\Ortomosaicos\{date}"

    if not os.path.exists(base_dir):
        print(f"No existe la carpeta para {date}")
        return

    print(f"\n📂 Ortomosaicos disponibles para {date}:\n")
    for f in sorted(os.listdir(base_dir)):
        if f.lower().endswith((".tif", ".laz")):
            print("  -", f)
