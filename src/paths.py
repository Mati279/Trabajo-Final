import os
from typing import List, Tuple

# ==================================================
# Rutas base del proyecto
# ==================================================

BASE_DIR = r"D:\Programas Python\Trabajo Final\Trabajo-Final"

# Carpetas principales
DATA_DIR = os.path.join(BASE_DIR, "data")
ESTANZUELA_DIR = os.path.join(DATA_DIR, "estanzuela")

# Carpeta Material dentro de data/
MATERIAL_DIR = os.path.join(DATA_DIR, "Material")
ORTOMOSAICOS_DIR = os.path.join(MATERIAL_DIR, "Ortomosaicos")

# Subcarpetas de ortomosaicos por fecha
ORTO_10ENE = os.path.join(ORTOMOSAICOS_DIR, "10ene")
ORTO_17ENE = os.path.join(ORTOMOSAICOS_DIR, "17ene")
ORTO_24ENE = os.path.join(ORTOMOSAICOS_DIR, "24ene")

# ==================================================
# Funciones utilitarias
# ==================================================

def ensure_exists(path: str, what: str = "ruta") -> str:
    """Verifica que una ruta exista y la devuelve si es válida."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe la {what}: {path}")
    return path

# ==================================================
# Rutas de data/estanzuela
# ==================================================

def get_estanzuela_path(nombre_carpeta: str) -> str:
    """Devuelve la ruta completa a una carpeta dentro de data/estanzuela."""
    ruta = os.path.join(ESTANZUELA_DIR, nombre_carpeta)
    return ensure_exists(ruta, "carpeta de campaña Estanzuela")

# ==================================================
# Rutas de data/Material/Ortomosaicos
# ==================================================

def get_ortomosaicos_dir(fecha: str = None) -> str:
    """
    Devuelve la carpeta general de ortomosaicos o la de una campaña específica.
    Ejemplo:
        get_ortomosaicos_dir()        → ...\\data\\Material\\Ortomosaicos
        get_ortomosaicos_dir('17ene') → ...\\data\\Material\\Ortomosaicos\\17ene
    """
    ruta = os.path.join(ORTOMOSAICOS_DIR, fecha) if fecha else ORTOMOSAICOS_DIR
    return ensure_exists(ruta, "carpeta de ortomosaicos")

def list_ortomosaicos(
    fecha: str = None,
    extensiones: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
) -> List[str]:
    """Lista los ortomosaicos disponibles."""
    carpeta = get_ortomosaicos_dir(fecha)
    return sorted(
        f for f in os.listdir(carpeta)
        if f.lower().endswith(extensiones)
        and os.path.isfile(os.path.join(carpeta, f))
    )

def get_ortomosaico_path(fecha: str, nombre: str) -> str:
    """Devuelve la ruta completa a un ortomosaico específico."""
    ruta = os.path.join(get_ortomosaicos_dir(fecha), nombre)
    return ensure_exists(ruta, "archivo de ortomosaico")

def print_ortomosaicos() -> None:
    """Imprime los ortomosaicos detectados agrupados por fecha."""
    print("=== Ortomosaicos detectados ===")
    for fecha in ("10ene", "17ene", "24ene"):
        try:
            archivos = list_ortomosaicos(fecha)
            if archivos:
                print(f"\nCampaña {fecha.upper()}:")
                for f in archivos:
                    print(f" - {os.path.join(get_ortomosaicos_dir(fecha), f)}")
            else:
                print(f"\nCampaña {fecha.upper()}: (sin archivos .tif)")
        except FileNotFoundError:
            print(f"\nCampaña {fecha.upper()}: carpeta no encontrada")
