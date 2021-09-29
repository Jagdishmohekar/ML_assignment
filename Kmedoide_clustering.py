## kmedoids clustering implementation

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice

## here we are choosing 4 random medoids initially
def i_medoids(X, k):
           
    samples = choice(len(X), size=k, replace=False)
    return X[samples, :]

##Calculating of distances for data points form medoids
def calculate_dist(X, med, p):
    length = len(X)
    shape = med.shape
    
    
    if len(shape) == 1: 
        med = med.reshape((1,len(med)))
    k = len(med)
    
    S = np.empty((length, k))
    
    for i in range(length):
        d = np.linalg.norm(X[i, :] - med, ord=p, axis=1) ## calculating euclidean distance p=2 represent eucledien distance
        S[i, :] = d**p
    
    return S

##assigning labels to datapoints
def label_assigning(S_matrix):
    return np.argmin(S_matrix, axis=1)
  
##finding new medoids
def finding_new_medoid(X, medoids, p):
    
    S = calculate_dist(X, medoids, p)
    labels = label_assigning(S)
    
    out_medoids = medoids
                
    for i in set(labels):
        
        dissimilarity_avg = np.sum(calculate_dist(X, medoids[i], p))
        
        cluster_points = X[labels == i]
        
        for datap in cluster_points:
            
            new_medoid = datap
            dissimilarity_new= np.sum(calculate_dist(X, datap, p))
            
            if dissimilarity_new < dissimilarity_avg :
                dissimilarity_avg=dissimilarity_new
                
                out_medoids[i] = datap
                
    return out_medoids

##checking old medoids and new medoids are same or not if same then algorithm will stop no need to run further
def converge(old_medoids, new_medoids):
    return set([tuple(i) for i in old_medoids]) == set([tuple(i) for i in new_medoids])


## calling all methods from bellow methods here X is a dataset, k for number of clusters and 2 for eucledian distance
def Kmedoid_Clust(X, k, p, s_medoids=None, max_steps=200):
    if s_medoids is None:
        medoids = i_medoids(X, k)
    else:
        medoids = s_medoids
        
    conv = False
    labels = np.zeros(len(X))
    
    i = 1
    while (not conv) and (i <= max_steps):
        old_medoids = medoids.copy()
        
        S_matrix = calculate_dist(X, medoids, p)
        
        labels = label_assigning(S_matrix)
        
        new_medoids = finding_new_medoid(X, medoids, p)
        conv = converge(old_medoids, new_medoids)
        i += 1
    
    return (new_medoids,labels)



## main method
if __name__ == "__main__":
    import pandas as pd
    
# Importing the dataset
    dataset = pd.read_csv('C:/Users/com/Desktop/BR_mod.csv')
    
    #filling missing data in datatset
    data=dataset.fillna(dataset.median())
    
    #taking all rows and all fetures from dataset
    X = data.iloc[:, :].values
    print("Dataset:",X.shape)
    
    #in below method 4 is represent number of clusters 
    results = Kmedoid_Clust(X, 4, 2) ## here 2 is passing for eucledean distance
    final_medoids = results[0]
    data['clusters'] = results[1]
    
    print("\nFinal 4 medoids are as follows:\n") 
    print(final_medoids)
    ########################################
    
    print("\ndata points with respective their assigned cluster number as follows:\n")
    print(data['clusters'])
    #########################################################
    
    #printing number of data points assigned to per clusters
    print("\nNumber of data points per cluster as follows:\n")
    for i in set(data['clusters']):
        j=len(X[data['clusters'] == i])
        print("Cluster {}: {} data points".format(i,j))