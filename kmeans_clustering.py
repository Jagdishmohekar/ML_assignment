## implementation of Kmeans Clustering algorithm 

import numpy as np
import matplotlib.pyplot as plt

# defining euclidean distance 

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


class KMeans_Clust:
    
    # using elbow method we can see 4 is optimal number of cluster
    def __init__(self, K=4, max_iteration=200): 
        self.max_iteration = max_iteration
        self.K = K
        self.centroids = []
        self.clusters = [[] for item in range(self.K)]
        
        
        
    def predict(self, X):
        self.X = X
        self.samples, self.features = X.shape

        random_sample_id = np.random.choice(self.samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_id]
        
        
        for i in range(self.max_iteration):
            
            # Assigning samples to the centroids 
            self.clusters = self.create_cluster(self.centroids)

            # Calculating new centroids 
            centroids_old = self.centroids
            self.centroids = self.get_centroid(self.clusters)
            
            # checking if clusters is final or not
            if self.converged(centroids_old, self.centroids):
                break
        return self.get_cluster_label(self.clusters)
    
    
    def create_cluster(self, centroids):
         
        cluster = [[] for item in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_id = self.closest_centroid(sample, centroids)
            cluster[centroid_id].append(idx)
        return cluster
    
    def get_cluster_label(self, clusters):
      
        label = np.empty(self.samples)
        for cluster_id, cluster in enumerate(clusters):
            for sample_index in cluster:
                label[sample_index] = cluster_id
        return label
    
    
    def get_centroid(self, clusters):
        # assigning mean value of clusters to centroids
        centroid = np.zeros((self.K, self.features))
        for cluster_id, cluster in enumerate(clusters):
            c_mean = np.mean(self.X[cluster], axis=0)
            centroid[cluster_id] = c_mean
            
            
        return centroid
    
    def closest_centroid(self, sample, centroids):
        # distance of the current sample from each centroid
        distance = [euclidean_dist(sample, point) for point in centroids]
        closest_index = np.argmin(distance)
        return closest_index
    
    
    def converged(self, centroids_old, centroids):
        
        dist = [
            euclidean_dist(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(dist) == 0
    
    #calulating number of data points assigned to per cluster
    
    def datapoints_assigned_to_cluster(self):
        j=0
        print("using elbow method I found that 4 number of cluster are optimal clusters and data points count per cluster as follows:\n")
        for i in self.clusters:
            print("cluster number {}: {} data points".format(j,len(i)))
            j=j+1
          
        
          
        
            
# main method

if __name__ == "__main__":
    import pandas as pd
    
# Importing the dataset
    dataset = pd.read_csv('C:/Users/com/Desktop/BR_mod.csv')
    
    #filling missing data in datatset
    data=dataset.fillna(dataset.median())
    
    #taking all rows and all fetures from dataset
    X = data.iloc[:, :].values
    print("Dataset:",X.shape)
    
    # using elbow method to find optimal number of cluster here I am using sklern 
    from sklearn.cluster import KMeans
    wcss = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 15), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    #################################################################################
    #creating object for KMeans_Clust class 
    
    # using elbow method we can see 4 is optimal number of cluster 
    k = KMeans_Clust(K=4, max_iteration=200) 
    
    #calling predict method
    y_pred = k.predict(X)
    
    #calling datapoints_assigned_to_cluster method to print actual how many data point finally assigned to per cluster 
    k.datapoints_assigned_to_cluster()
