## implementing Hierarchical agglomerative Clustering

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class aglomerative_dist:
    
    def __init__(self): 
        pass
    
    # calculating distance matrix 
    def compute_dist(self,n_samples):
        dist_matrix = np.zeros((len(n_samples),len(n_samples)))
        
            
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                if i!=j:
                    dist_matrix[i,j] = float(self.dist_cal(n_samples[i],n_samples[j]))
                else:
                    dist_matrix[i,j] = 0
        #print(dist_matrix)    
        return dist_matrix
        
    #calculating distance between two sample or two cluster or one sample and one cluster    
    def dist_cal(self,sample1,sample2):
        
        distance = []
        for i in range(len(sample1)):
            for j in range(len(sample2)):
                try:
                    distance.append(np.linalg.norm(np.array(sample1[i])-np.array(sample2[j])))
                except:
                    distance.append(self.inter_sample_dist(sample1[i],sample2[j]))
        return min(distance)
    
    # for calculating distances calling below method 
    def inter_sample_dist(self,sample1,sample2):
        
        if str(type(sample2[0]))!='<class \'list\'>':
            sample2=[sample2]
        if str(type(sample1[0]))!='<class \'list\'>':
            sample1=[sample1]
        a = len(sample1)
        b = len(sample2)
        
        distance = []
        if b>=a:
            for i in range(b):
                for j in range(a):
                    if (len(sample2[i])>=len(sample1[j])) and str(type(sample2[i][0])!='<class \'list\'>'):
                        distance.append(self.interclusterdist(sample1[i],sample2[j]))
                    else:
                        distance.append(np.linalg.norm(np.array(s2[i])-np.array(s1[j])))
        else:
            for i in range(a):
                for j in range(b):
                    if (len(sample1[i])>=len(sample2[j])) and str(type(sample1[i][0])!='<class \'list\'>'):
                        distance.append(self.inter_cluster_dist(sample1[i],sample2[j]))
                    else:
                        distance.append(np.array(sample1[i])-np.array(sample2[j]))
        return min(distance)
    
    # calculating distances between one cluster and one sample
    def interclusterdist(self,clustl,sample):
        if sample[0]!='<class \'list\'>':
            sample = [sample]
        distance   = []
        for i in range(len(clustl)):
            for j in range(len(sample)):
                distance.append(np.linalg.norm(np.array(clustl[i])-np.array(sample[j])))
        return min(distance)


# main method

if __name__ == "__main__":
    import pandas as pd
    
# Importing the dataset
    dataset = pd.read_csv('C:/Users/com/Desktop/BR_mod.csv')
    
    #filling missing data in datatset
    data=dataset.fillna(dataset.median())
    
    #taking all rows and all fetures from dataset
    X = data.iloc[:, :].values
    print("Dataset :",X.shape)
    
    #finding number of cluster using dendogram
    import scipy.cluster.hierarchy as sch
    dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
    plt.title('Dendrogram')
    plt.xlabel('BR_mod')
    plt.ylabel('Euclidean distances')
    plt.show()
    
    ## by dendogram we can see 2 is optimal number of clusters
    ####################################################################
    #assigning sequences to data points
    sequences = [[i] for i in range(X.shape[0])]
    #print(sequences)
    n_samples = [[list(X[i])] for i in range(X.shape[0])]
    #print(n_samples)
    length = len(n_samples)
    
    # using dendogram we can see 2 is optimal number of cluster 
    k = aglomerative_dist() 
    
    
    # calculating distance matrix and forming clusters 
    
    while length>1:    
        dist_matrix= k.compute_dist(n_samples)
        sample = np.where(dist_matrix==dist_matrix.min())[0]
        value= n_samples.pop(sample[1])
        n_samples[sample[0]].append(value)
        print("cluster 1'",sequences[sample[0]])
        print("Cluster 2",sequences[sample[1]])
        sequences[sample[0]].append(sequences[sample[1]])
        sequences[sample[0]] = [sequences[sample[0]]]
        v = sequences.pop(sample[1])
        length = len(n_samples)
        
    ### printing final clusters
    print("Cluster:",sequences[sample[0]])
                                                                                                                                          
    