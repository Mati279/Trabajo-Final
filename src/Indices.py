import numpy as np

class VegetationIndices:
    """
    Calculadora de índices vegetativos.
    
    Trabaja con los outputs normalizados de "Normalizacion.process_session".
    
    Mapeo de Bandas MS:
    0: RED
    1: GREEN
    2: NIR
    3: RED EDGE
    4: ALPHA
    """
    
    # Bandas MS
    B_MS_RED = 0
    B_MS_GREEN = 1
    B_MS_NIR = 2
    B_MS_RED_EDGE = 3
    
    # Bandas RGB
    B_RGB_RED = 0
    B_RGB_GREEN = 1
    B_RGB_BLUE = 2

    def __init__(self, ms_norm_array, rgb_norm_array=None): # RGB opcional por ahora.
        """
        Inicializa la calculadora con arrays ya normalizados.

        Args:
            ms_norm_array (numpy.ndarray).
            rgb_norm_array (numpy.ndarray, opcional).
        """
        self.ms_array = ms_norm_array
        self.rgb_array = rgb_norm_array
        
        # Extracción de bandas del MS
        self.red = self.ms_array[self.B_MS_RED, :, :]
        self.green = self.ms_array[self.B_MS_GREEN, :, :]
        self.nir = self.ms_array[self.B_MS_NIR, :, :]
        self.red_edge = self.ms_array[self.B_MS_RED_EDGE, :, :]
        
        # Extracción de bandas del RGB (si disponible)
        if self.rgb_array is not None:
            self.rgb_red = self.rgb_array[self.B_RGB_RED, :, :]
            self.rgb_green = self.rgb_array[self.B_RGB_GREEN, :, :]
            self.rgb_blue = self.rgb_array[self.B_RGB_BLUE, :, :]

    def _safe_divide(self, numerator, denominator):
        """
        Maneja división por cero y NaNs.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(numerator, denominator)
            result[result == np.inf] = 0.0
            result[result == -np.inf] = 0.0
            result = np.nan_to_num(result, nan=0.0)
        return result

    # Índices principales (usando el MS)   
    def calculate_ndvi(self):
        """NDVI = (NIR - RED) / (NIR + RED)"""
        return self._safe_divide(self.nir - self.red, self.nir + self.red)

    def calculate_ndre(self):
        """NDRE = (NIR - RedEdge) / (NIR + RedEdge)."""
        return self._safe_divide(self.nir - self.red_edge, self.nir + self.red_edge)

    def calculate_savi(self, L=0.5): # L sería para el suelo.
        """SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)"""
        numerator = self.nir - self.red
        denominator = self.nir + self.red + L
        return self._safe_divide(numerator, denominator) * (1 + L)
    
    def calculate_gndvi(self):
        """GNDVI = (NIR - Green) / (NIR + Green)"""
        return self._safe_divide(self.nir - self.green, self.nir + self.green)

    # Índices adicionales (requiere el blue del RGB)
    def calculate_vari(self):
        """VARI = (Green - Red) / (Green + Red - Blue)."""
        numerator = self.rgb_green - self.rgb_red
        denominator = self.rgb_green + self.rgb_red - self.rgb_blue
        return self._safe_divide(numerator, denominator)

    def calculate_exg(self):
        """ExG = 2*Green - Red - Blue."""
        if self.rgb_blue is None: return None
        return 2 * self.rgb_green - self.rgb_red - self.rgb_blue

    def calculate_evi_hybrid(self):
        """
        EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))
        """
        if self.rgb_blue is None: return None
        numerator = self.nir - self.red
        denominator = self.nir + 6 * self.red - 7.5 * self.rgb_blue + 1
        return 2.5 * self._safe_divide(numerator, denominator)

    def calculate_main_indices(self):
        """Calcula y devuelve todos los índices principales."""
        indices = {
            "ndvi": self.calculate_ndvi(),
            "ndre": self.calculate_ndre(),
            "savi": self.calculate_savi(),
            "gndvi": self.calculate_gndvi()
        }
        
        if self.rgb_array is not None:
            indices["vari"] = self.calculate_vari()
            indices["exg"] = self.calculate_exg()
            indices["evi"] = self.calculate_evi_hybrid()
            
        return indices