import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import style
style.use('ggplot') or plt.style.use('ggplot')

# Preprocesado y modelado
# ==============================================================================
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')



# Escalado de las variables
# ==============================================================================

def distancias(x_n):
    
    x_n =  pd.DataFrame(x_n)  

    datos_scaled = scale(X=x_n, axis=0, with_mean=True, with_std=True) 
    datos_scaled = pd.DataFrame(datos_scaled, columns=x_n.columns, index=x_n.index)
    print(datos_scaled.head(4))

    # Cálculo de distancias
    # ==============================================================================
    print('------------------')
    print('Distancia')
    print('------------------')
    distancias = pairwise_distances(
                X      = datos_scaled,
                metric ='euclidean'
             )

    # Se descarta la diagonal superior de la matriz
    distancias[np.triu_indices(n=distancias.shape[0])] = np.nan

    distancias = pd.DataFrame(
                distancias,
                columns=datos_scaled.index,
                index = datos_scaled.index
            )

    distancias.iloc[:4,:4]
    distancias[distancias.isna()] = 0 
    print(distancias)

    return distancias

