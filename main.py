# Importar librerias necesarias
# ==============================================================================
from re import X
from turtle import distance
from typing import Any
import pandas as pd
import numpy as np
import os 

# Leer los datos completos 
# ==============================================================================

def preprocesar(file_path):

    """
    Preprocesamiento

    :parametro de entrada file_path
    :retorno x_p datos procesados

    """
   
    x = pd.read_csv(file_path)
    x_p = x.drop(['sg_document_type','sg_create_at','sg_update_at','form_date','legal_representatives','commercial_referrals','format_info.code','format_info.version','basic_info.entity.id_type','basic_info.entity.id','basic_info.legal_representative.name','basic_info.legal_representative.id_type',
                'basic_info.legal_representative.id','basic_info.address','basic_info.address','basic_info.phone','basic_info.contact_info','business_info.joint-document','business_info.commercial_registration','business_info.registered_shared_capital','business_info.constitution_date','business_info.good_or_service',
                'certificates.list_of_certificates','certificates.in_progress.process','certificates.in_progress.percentage_progress','certificates.in_progress.init_date','accounting_and_taxes.isRegimenComun','accounting_and_taxes.isRegimenSimplificado','accounting_and_taxes.isDeclaraRenta','accounting_and_taxes.isAutoRetenedor',
                'accounting_and_taxes.payment_terms','accounting_and_taxes.payment_terms_other','sg_additional_info'], axis=1) # Borrar columnas de etiquetas 
    
    x_p = pd.DataFrame(x_p)
    x_p =  pd.get_dummies(x_p, columms = ['user_type'])

    print(x_p)

    #x_p[x_p.isna()] = 0  
       
    return (x_p) # datos preprocesados 

# Normalization of data
# ==============================================================================
    
def normalizado(x_p):

    """
    normalización

    :parametro de entrada x_p datos procesados
    :retorno x_n datos normalizados

    """
    
    global x_n

    # Min de los datos por columna 
    min = x_p.min(0)

    # Min de los datos por filas 
    max = x_p.max(0)

    # Normalization de datos 
    x_n = (x_p - min)/(max - min)
    x_n =  np.array(x_n) 
    print(x_n)


    return (x_n) # datos normalizados y preprocesados 


if __name__ == '__main__':

    path = "/home/user/Documentos/2022_Segmentacion/paquetes/segmentacion/utilidades_formulario"
    dirs = os.listdir(path)


    for file in dirs:

        file_name = file
        print(file_name)
        file_path = '{}/{}'.format(path, file_name)
        print(file_path)

        x_p = preprocesar(file_path)
        # x_n = normalizado(x_p)  
        # print(x_n.shape)  


from algoritmos import algoritmo_1, algoritmo_2, algoritmo_3, algoritmo_4, algoritmo_5, algoritmo_6, evaluacion
from distancias import distancias

# DA1,MA1 = algoritmo_1(x_n) # (Dataframe de puntos, Matriz de densidad)
# #print('Dataframe de puntos')
# #print(DA1)

# DA2,CA2,MA2 = algoritmo_2(x_n) # (Dataframe de puntos, Matriz de Centros, Matriz de densidad)
# print('Dataframe de puntos')
# print(DA2)
# print('Matriz de centros')
# print(CA2)

# DA3,CA3,MA3 = algoritmo_3(x_n) # (Dataframe de puntos, Matriz de Centros, Matriz de densidad)
# print('Dataframe de puntos')
# print(DA3)
# print('Matriz de centros')
# print(CA3)

# DA4,CA4,MA4 = algoritmo_4(x_n) # (Dataframe de puntos, Matriz de Centros, Matriz de densidad)
# print('Dataframe de puntos')
# print(DA4)
# print('Matriz de centros')
# print(CA4)

# DA5,MA5 = algoritmo_5(x_n) # (Dataframe de puntos, Matriz de densidad)
# print('Dataframe de puntos')
# #print(DA5)

# DA6,MA6 = algoritmo_6(x_n) # (Dataframe de puntos, Matriz de densidad)
# print('Dataframe de puntos')
# print(DA6)

# D = distancias(x_n)
# #x = evaluacion(x_n)