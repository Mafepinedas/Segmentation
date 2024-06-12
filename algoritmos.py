# -*- coding: utf8 -*-
import warnings
from matplotlib.pyplot import step
from sklearn.cluster import AgglomerativeClustering, MeanShift, DBSCAN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import matrix, unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

def algoritmo_1(x_n): 

    """
    Algoritmo número I DBSCAN

    :parametro de entrada x_n
    :retorno matriz DA1

    """
      
    x_n =  np.array(x_n)  
    iteraciones = [7,7.5,10] 

    for i in iteraciones:

        clustering = DBSCAN(eps=0.25, min_samples=i).fit(x_n)
        D = print('Found clusters UMAP with DBSCAN', len(np.unique(clustering.labels_))) 
        D = clustering.labels_ 
        D = D.tolist()                  
        C_K = list(dict.fromkeys(D))   
        posicion = C_K
        C_K = len(C_K)            
      
        D_A1 = np.zeros((len(D), len(posicion)))
        posicion = np.array(posicion)
    
        for i in range(len(D)): 
        
            if D[i] in posicion:
                D_A1[i, np.where(posicion == D[i])[0][0]] = 1

        D_A1 = pd.DataFrame(D_A1, columns = posicion)
        matrix = D_A1.to_numpy()   
        print(matrix)         
    
    return D_A1,matrix


def algoritmo_2(x_n):

    """
    Algoritmo número II MeanShift

    :parametro de entrada x_n
    :retorno matriz DA2

    """

    x_n =  np.array(x_n)  
    iteraciones = [0.3,0.35,0.38,0.45,0.48]   

    for  bandwidth in iteraciones:
        ms = MeanShift(bandwidth=bandwidth, cluster_all=True)
        ms.fit(x_n)
        print('Found clusters meanshift', len(np.unique(ms.labels_)))
        C = ms.cluster_centers_
        D = ms.labels_
        D = D.tolist()                  
        C_K = list(dict.fromkeys(D))   
        posicion = C_K
        C_K = len(C_K)        
                 
      
        D_A2 = np.zeros((len(D), len(posicion)))
        posicion = np.array(posicion)
    
        for i in range(len(D)): 
        
            if D[i] in posicion:
                D_A2[i, np.where(posicion == D[i])[0][0]] = 1

        D_A2 = pd.DataFrame(D_A2, columns = posicion)        
        matrix = D_A2.to_numpy()  
        print(matrix)      

    return D_A2,C,matrix


def algoritmo_3(x_n):

    """
    Algoritmo número III KMeans

    :parametro de entrada x_n
    :retorno matriz DA3

    """
	
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]    
    score = [kmeans[i].fit(x_n).score(x_n) for i in range(len(kmeans))]
   
    
    plt.plot(Nc,score)
    plt.xlabel('Numero de Score')
    plt.ylabel('Score') 
    plt.title('Curva')
    #plt.show()

    kmeans = KMeans(n_clusters=4).fit(x_n)
    labels = kmeans.predict(x_n)   
    C = kmeans.cluster_centers_
    colores=['red','green','blue','cyan']
    asignar=[]

    for row in labels:
        asignar.append(colores[row])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_n[:, 0], x_n[:, 1], x_n[:, 2], c=asignar,s=60)
    ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000) 
    #plt.show()
      
    D = print('Found clusters UMAP with DBSCAN', len(np.unique(kmeans.predict(x_n)))) 
    D = kmeans.predict(x_n) 
    D = D.tolist()               
    C_K = list(dict.fromkeys(D))   
    posicion = C_K
    C_K = len(C_K)      
                  
    D_A3 = np.zeros((len(D), len(posicion)))
    posicion = np.array(posicion)
    
    for i in range(len(D)): 
        
        if D[i] in posicion:
                D_A3[i, np.where(posicion == D[i])[0][0]] = 1

    D_A3 = pd.DataFrame(D_A3, columns = posicion)
    matrix = D_A3.to_numpy()
    print(matrix)
    
    #plt.show()
    
    return D_A3,C,matrix


def algoritmo_4(x_n):  

    """
    Algoritmo número IV affinity_model

    :parametro de entrada x_n
    :retorno matriz DA4

    """  
    
    affinity_model = AffinityPropagation(damping=0.5)
    affinity_model.fit(x_n)
    affinity_result = affinity_model.predict(x_n)
    C = affinity_model.cluster_centers_    
    D = print('Found clusters affinity_model', len(np.unique(affinity_model.predict(x_n)))) 
    D = affinity_model.predict(x_n)   
    D = D.tolist()                 
    C_K = list(dict.fromkeys(D)) 
    posicion = C_K
    C_K = len(C_K)   
                  
    D_A4 = np.zeros((len(D), len(posicion)))
    posicion = np.array(posicion)
    
    for i in range(len(D)): 
        
        if D[i] in posicion:
                D_A4[i, np.where(posicion == D[i])[0][0]] = 1

    D_A4 = pd.DataFrame(D_A4, columns = posicion)    
    matrix = D_A4.to_numpy()   

    affinity_clusters = unique(affinity_result)
    # for affinity_cluster in affinity_clusters:
    #     index = where(affinity_result == affinity_cluster)
    #     pyplot.scatter(x_n[index, 0], x_n[index, 1])
    
    return D_A4,C,matrix

    
def algoritmo_5(x_n):


    """
    Algoritmo número V AgglomerativeClustering

    :parametro de entrada x_n
    :retorno matriz DA5

    """
    
    agglomerative_model = AgglomerativeClustering().fit(x_n)
    agglomerative_result = agglomerative_model.fit_predict(x_n)
    D = print('Found clusters agglomerative_result', len(np.unique(agglomerative_model.fit_predict(x_n)))) 
    D = agglomerative_model.fit_predict(x_n)
    D = D.tolist()                
    C_K = list(dict.fromkeys(D))  
    posicion = C_K
    C_K = len(C_K)      
               
    D_A5 = np.zeros((len(D), len(posicion)))
    posicion = np.array(posicion)

    for i in range(len(D)): 
        
        if D[i] in posicion:
                D_A5[i, np.where(posicion == D[i])[0][0]] = 1

    D_A5 = pd.DataFrame(D_A5, columns = posicion)
    matrix = D_A5.to_numpy()
        
    agglomerative_clusters = unique(agglomerative_result)    
    # for agglomerative_clusters in agglomerative_clusters:    
    #     index = where(agglomerative_result == agglomerative_clusters)        
    #     pyplot.scatter(x_n[index, 0], x_n[index, 1])   
    # pyplot.show()  

    return D_A5,matrix


def algoritmo_6(x_n):


    """
    Algoritmo número VI optics_modes

    :parametro de entrada x_n
    :retorno matriz DA6

    """

    x_n =  np.array(x_n)  
            
    for i in [10,11,13,15,21]:

        optics_model = OPTICS(eps=0.25, min_samples=i)
        optics_result = optics_model.fit_predict(x_n)
        D = print('Found clusters optics_model', len(np.unique(optics_model.fit_predict(x_n))))
        optics_clusters = unique(optics_result)       
        D = optics_model.fit_predict(x_n)
        D = D.tolist()                 
        C_K = list(dict.fromkeys(D)) 
        posicion = C_K
        C_K = len(C_K)        
                       
        D_A6 = np.zeros((len(D), len(posicion)))
        posicion = np.array(posicion)

        for i in range(len(D)): 
        
            if D[i] in posicion:
                D_A6[i, np.where(posicion == D[i])[0][0]] = 1

        D_A6 = pd.DataFrame(D_A6, columns = posicion)
        print(D_A6)
        matrix = D_A6.to_numpy()
        print(matrix)

       
        # for optics_clusters in optics_clusters:            
        #     index = where(optics_result == optics_clusters)   
        #     pyplot.scatter(x_n[index, 0], x_n[index, 1])
        # pyplot.show()
    
    return D_A6,matrix



def evaluacion(x_n):   
    
    
    agglomerative_model = AgglomerativeClustering(n_clusters=4)
    birch_model = Birch(threshold=0.03, n_clusters=4)
    dbscan_model = DBSCAN(eps=0.25, min_samples=9)
    kmeans_model = KMeans(n_clusters=4)
    mean_model = MeanShift()
    optics_model = OPTICS(eps=0.5, min_samples=10)
    gaussian_model = GaussianMixture(n_components=4)  
    
    birch_model.fit(x_n)
    kmeans_model.fit(x_n)
    gaussian_model.fit(x_n)
    
    agglomerative_result = agglomerative_model.fit_predict(x_n)
    print('Found clusters agglomerative_result', len(np.unique(agglomerative_model.fit_predict(x_n)))) # Vector
    
    birch_result = birch_model.predict(x_n)
    print('Found clusters birch_model', len(np.unique(birch_model.predict(x_n)))) # Vector

    dbscan_result = dbscan_model.fit_predict(x_n)
    print('Found clusters dbscan_model', len(np.unique(dbscan_model.fit_predict(x_n)))) # Vector

    kmeans_result = kmeans_model.predict(x_n)
    print('Found clusters kmeans_mode', len(np.unique(kmeans_model.predict(x_n)))) # Vector

    mean_result = mean_model.fit_predict(x_n)
    print('Found clusters mean_model', len(np.unique(mean_model.fit_predict(x_n)))) # Vector

    optics_result = optics_model.fit_predict(x_n)
    print('Found clusters optics_model', len(np.unique(optics_model.fit_predict(x_n)))) # Vector

    gaussian_result = gaussian_model.predict(x_n)
    print('Found clusters gaussian_model', len(np.unique(gaussian_model.predict(x_n)))) # Vector

     
    agglomerative_clusters = unique(agglomerative_result)
    birch_clusters = unique(birch_result)
    dbscan_clusters = unique(dbscan_result)
    kmeans_clusters = unique(kmeans_result)
    mean_clusters = unique(mean_result)
    optics_clusters = unique(optics_result)
    gaussian_clusters = unique(gaussian_result)

    

    
    for agglomerative_clusters in agglomerative_clusters:
        index = where(agglomerative_result == agglomerative_clusters)
        pyplot.scatter(x_n[index, 0], x_n[index, 1])   
    pyplot.show()   

    
    for birch_clusters in birch_clusters:
        
        index = where(birch_result == birch_clusters)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()

    
    for dbscan_clusters in dbscan_clusters:
       
        index = where(dbscan_result == dbscan_clusters)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()


    
    for kmeans_clusters in kmeans_clusters:
        
        index = where(kmeans_result == kmeans_clusters)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()

    
    for mean_cluster in mean_clusters:
        
        index = where(mean_result == mean_cluster)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()

    
    for optics_clusters in optics_clusters:
        
        index = where(optics_result == optics_clusters)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()

    
    for gaussian_clusters in gaussian_clusters:
       
        index = where(gaussian_result == gaussian_clusters)        
        pyplot.scatter(x_n[index, 0], x_n[index, 1])    
    pyplot.show()







