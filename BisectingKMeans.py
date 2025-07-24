#Sam Nuttall - 201608203 - Data Mining and Visualisation Assignment 2

##For this code I reused my code for the created functions for Question 1 KMeans.py

import numpy as np
import random
import matplotlib.pyplot as plt

#Fix the seed value in order to not get new graphs every run
random.seed(42)
np.random.seed(42)

def ComputeDistance(x, y):
    #Function to measure eucluidian distances between points.
    #Input: points X and Y 
    #Output: euclidean distance between X and Y
    squared_diff = np.sum((x - y)**2) 
    return np.sqrt(squared_diff)

def load_dataset(filename="dataset"):
    #function to reads the dataset file and separate labels from features - looks for a file called dataset and if its not found then throws a file not found error.
    #Input: filename (default = 'dataset')
    #Output: labels (np.array), features (np.array)
    labels = [] # initialise a labels array
    features = [] # intialise a features array

    try:
        with open(filename, "r") as f: #read file and assign it to variable 'f'
            for line in f: #loop through each line
                parts = line.strip().split() #split all the data at the whitespaces between
                if len(parts) < 2:
                    continue
                labels.append(parts[0]) # add the first part of the line to labels
                features.append([float(x) for x in parts[1:]]) #add the remaining parts of the lines to features
    except FileNotFoundError:
        print("Error: dataset file not found.")
        exit() # throw error and exit the progrma if file is not found
    if len(features) < 2:
        print("Error: Not enough data points for clustering.")
        exit()  

    return np.array(labels), np.array(features) #return the labels and features

def initialSelection(data, k):
    #Chooses initial centroids randomly.
    #Input: data from arrays, number of clusters k
    #Output: list of k randomly selected points from the data
    return random.sample(data.tolist(), k)

def assignClusterIds(data, centroids): 
    #Assign points to the nearest centroid
    #Input: data from arrays, centroids list
    #Output: list of clusters (each cluster is a list of points)
    clusters = [[] for _ in range(len(centroids))] #create arrays for 'number of centroids' number of clusters
    for point in data: #loop for each point in the data 
        distances = [ComputeDistance(point, centroid) for centroid in centroids] #calculating distance to each centroid
        cluster_index = np.argmin(distances)#find minimum distance to assign points to the nearest centroid
        clusters[cluster_index].append(point)
    return clusters #return the array of clusters

def computeClusterRepresentatives(clusters):
    #Updates centroids by computing the mean of each cluster.
    #Input: clusters array
    #Output: list of updated centroids
    centroids = []
    for cluster in clusters: #loop through each cluster and find the average vector for the cluster
        centroids.append(np.mean(cluster, axis=0))
    return centroids

def computeSumfSquare(cluster):
    #Compute the sum of squared distances within a cluster from the centroid and return it 
    #Input: cluster and the centroid of that cluster
    #Output: sum of square distances
    center = np.mean(cluster, axis=0)
    return np.sum([ComputeDistance(p, center)**2 for p in cluster])

def kmeans_split(data):
    #Performs k-means clustering for k=2.
    #Input: data array
    #Output: clusters after splitting into 2 (list)
    centroids = initialSelection(data, 2) #randomly selecting 2 initial centroids 
    while True: #repeat until converegence 
        clusters = assignClusterIds(data, centroids) #assign clusters
        new_centroids = computeClusterRepresentatives(clusters) #find new centroids of the clusters
        if np.allclose(centroids, new_centroids): #check if centroids are correct
            break
        centroids = new_centroids
    return clusters #return clusters

def bisecting_kmeans(data, max_clusters=9):
    #Performs bisecting k-means to identify the 9 clusters
    #Input: data array, max_clusters (9 for the task)
    #Output: hierarchy list of clusterings
    
    clusters = [data.tolist()] #Put all data as 1 cluster
    hierarchy = [clusters.copy()] #Initialise the heirarchy as that single cluster

    while len(clusters) < max_clusters:
        #Compute sum of square distances for each cluster
        ssd = [computeSumfSquare(cluster) for cluster in clusters]
        split_index = np.argmax(ssd) # Prepare to split at cluster with highest ssd
        clusters_split = np.array(clusters[split_index])
        new_clusters = kmeans_split(clusters_split) # Split the cluster in 2 using kmeans_split function
        clusters = clusters[:split_index] + new_clusters + clusters[split_index+1:] #Replace old cluster with the 2 new ones
        hierarchy.append(clusters.copy()) # Update the heirarchy of clusters to contain these new ones
    return hierarchy #return finalised heirarchy with all the clusters split down into the 9 ones

def silhouette_score(data, clusters):
    #Function to calculate Silhoutte scores
    #Input: dataset, clusters
    #Output: average silhouette coefficient as a float
    scores = []
    for point in data: #loop through all data points
        cluster_index = None
        #identify the cluster the point belongs to
        for i, cluster in enumerate(clusters):
            if any(np.all(np.isclose(point, p)) for p in cluster):
                cluster_index = i
                break
        
        if cluster_index is None or len(clusters[cluster_index]) <= 1:  #if only 1 point in cluster or none found in the cluster
            scores.append(0)
            continue

        #a = average distance between the points in the same cluster computed by looping through each other point and using the compute distance function
        a = np.mean([ComputeDistance(point, other_point)
                     for other_point in clusters[cluster_index]
                     if not np.all(np.isclose(point, other_point))])

        other_clusters = [c for j, c in enumerate(clusters) if j != cluster_index and len(c) > 0]
        if not other_clusters:
            scores.append(0)
            continue

        #b = minimum average distance to the points in any other clusters
        b = min([np.mean([ComputeDistance(point, other_point) for other_point in cluster])
                 for cluster in other_clusters]) # loops through every point in every other cluster
        # use the silhouette formula from lecture to get the scores
        scores.append((b - a) / max(a, b))
    return np.mean(scores) #return silhouett scores.

def clustername(data, k):
    #Runs Bisecting k-means clustering until k clusters are found and returns silhouette score.
    #Input: data (np.array), k (int)
    #Output: clusters and silhouette score (tuple)

    #Run bisecting k-means until getting hierarchy up to k clusters 
    hierarchy = bisecting_kmeans(data, max_clusters=k)
    clusters_np = [np.array(c) for c in hierarchy[-1]]
    score = silhouette_score(data, clusters_np) #Calculate silhouette score for the clusters
    return clusters_np, score

def plot_silhouettee(scores):
    #Plots the relationship between k number of clusters and silhouette coefficient
    #Input: list of silhouette scores where array index corresponds to number of clusters
    #Outpit: plot of scores to clusters
    k_values = list(range(1, len(scores)+1))
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Score vs. Number of Clusters (K-Means Bisecting)')
    plt.grid(True)
    plt.savefig('bisectingKMeans-Plot.png')  #Saves the plot to a png file in folder
    plt.show()

def main():
    #Main function to runs the clustering algorithms for k = 1 - 9
    _, data = load_dataset()
    scores = []

    for k in range(1, 10): #loop through for 10 clusters
        if k == 1:
            scores.append(0)
            continue
        _, score = clustername(data, k)
        scores.append(score) #add the scores to the list where each index represents K number of clusters

    plot_silhouettee(scores) #plot the scores into a new png 

if __name__ == "__main__":
    main() #runs the main program when the script is run.