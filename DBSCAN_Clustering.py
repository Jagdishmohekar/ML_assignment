## implementing DBSCAN Clustering Algorithm
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

core_point=[]
border_point=[]
# finding neighbours for each data points
def finding_neighnour(data,pt,eps,minpoint,labels,cluster):
    neighbour=[]
    label_no=[]
    
    # here I am calculating euclidean distance and comparing with epsilon
    for i in range(len(data)):
        if np.linalg.norm(data[pt]-data[i])<eps:
            neighbour.append(data[i])
            label_no.append(i)
    
    if len(neighbour) >= minpoint:
        core_point.append(pt)
        for i in range(len(labels)):
            if i in label_no:
                labels[i]=cluster
    else:
        border_point.append(pt)
        for i in range(len(labels)):
            if i in label_no:
                labels[i]=-1
    
    return labels


#counting number of clusters and ploting graph

def DBSCAN_Clust(data,epsilon,minpoint):
  
    labels=[0] * len(data)

    cluster=1

    for i in range(len(data)):
        if(labels[i]==0):
            labels = finding_neighnour(data,i,epsilon,minpoint,labels,cluster)        
    
    clusters,point_count = np.unique(labels,return_counts=True)
    
    #printing number of clusters formed
    print("Number of Clusters formed :{}".format(len(clusters)))
    
    #printing core points and border points of cluster 1
    print("Cluster No. {}:".format(cluster))
    print("Core Points: {}\n".format(core_point))
    print("Boarder Points: {}\n".format(border_point))
    
    
    #Plotting using 2 features   
    plt.title(f"DBSCAN Clustering for (epsilon={epsilon} , MinPoint={minpoint})")
    plt.scatter(data[:,1],data[:,2],c=labels,cmap='seismic')
    plt.show()
    
        
if __name__ == '__main__':
    data = arff.loadarff('C:/Users/com/Desktop/diabetes1.arff')
    dataset = pd.DataFrame(data[0])   
    #taking all rows and all fetures from dataset
    d = dataset.iloc[:, 0:8]
    
    #print(dataset)
    #data preprocessing : tranforming data
    X = StandardScaler().fit_transform(d.values)
    print("Dataset: {}\n".format(X.shape))
    #Executing the  Algorithm by passing epsilon=2 and minimum points are 5 on all features of dataset
    DBSCAN_Clust(X,2,5)
  