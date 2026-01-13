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

    def __init__(self, ms_norm_array, rgb_norm_array=None): 
        """
        Inicializa la calculadora con arrays ya normalizados.

        Args:
            ms_norm_array (numpy.ndarray).
            rgb_norm_array (numpy.ndarray, opcional).
        """
        self.ms_array = ms_norm_array
        self.rgb_array = rgb_norm_array
        
        # Extrae las bandas del MS
        self.red = self.ms_array[self.B_MS_RED, :, :]
        self.green = self.ms_array[self.B_MS_GREEN, :, :]
        self.nir = self.ms_array[self.B_MS_NIR, :, :]
        self.red_edge = self.ms_array[self.B_MS_RED_EDGE, :, :]
        
        # Extrae las bandas del RGB (si disponible)
        if self.rgb_array is not None:
            self.rgb_red = self.rgb_array[self.B_RGB_RED, :, :]
            self.rgb_green = self.rgb_array[self.B_RGB_GREEN, :, :]
            self.rgb_blue = self.rgb_array[self.B_RGB_BLUE, :, :]

    def _safe_divide(self, numerator, denominator):
        """
        Realiza la división manejando denominadores cero, valores muy pequeños y NaNs.
        """
        # Crea una máscara para identificar divisiones por cero o números extremadamente pequeños
        # que disparan los valores a infinito.
        mask = (denominator == 0) | (np.abs(denominator) < 1e-10)
        
        # Prepara un array de salida lleno de NaNs para los casos inválidos
        result = np.full(numerator.shape, np.nan, dtype=np.float32)
        
        # Realiza la división solo donde el denominador es seguro
        result[~mask] = numerator[~mask] / denominator[~mask]
        
        return result

    # Índices principales (usando el MS)   
    def calculate_ndvi(self):
        """NDVI = (NIR - RED) / (NIR + RED)"""
        res = self._safe_divide(self.nir - self.red, self.nir + self.red)
        # Asegura que el resultado esté en el rango físico real [-1, 1]
        return np.clip(res, -1, 1)

    def calculate_ndre(self):
        """NDRE = (NIR - RedEdge) / (NIR + RedEdge)."""
        res = self._safe_divide(self.nir - self.red_edge, self.nir + self.red_edge)
        # Asegura que el resultado esté en el rango físico real [-1, 1]
        return np.clip(res, -1, 1)

    def calculate_savi(self, L=0.5): 
        """SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)"""
        numerator = self.nir - self.red
        denominator = self.nir + self.red + L
        return self._safe_divide(numerator, denominator) * (1 + L)
    
    def calculate_gndvi(self):
        """GNDVI = (NIR - Green) / (NIR + Green)"""
        res = self._safe_divide(self.nir - self.green, self.nir + self.green)
        return np.clip(res, -1, 1)

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
    
    def plot_index(self, index_array, title="Mapa de Índice", cmap='RdYlGn', vmin=None, vmax=None):
        """
        Muestra un mapa de calor de un índice y su histograma (relación 2:1).
        Permite definir vmin y vmax. Si no se definen y el cmap es RdYlGn, usa [-1, 1].
        """
        import matplotlib.pyplot as plt

        # Establece los límites visuales para índices normalizados
        if vmin is None and vmax is None and cmap == 'RdYlGn':
            vmin, vmax = -1, 1

        # Configuración de la figura: 2 columnas, relación de ancho 2:1
        fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
        
        # Mapa
        im = ax_map.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_map.set_title(title)
        ax_map.axis('off')
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04, label='Valor')

        # Histograma
        # Aplanar y filtrar NaNs
        valid_data = index_array.flatten()
        valid_data = valid_data[~np.isnan(valid_data)]
        
        ax_hist.hist(valid_data, bins=50, color='gray', alpha=0.7, edgecolor='white')
        ax_hist.set_title("Distribución de Valores")
        ax_hist.set_xlabel("Valor")
        ax_hist.set_ylabel("Frecuencia")
        ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()