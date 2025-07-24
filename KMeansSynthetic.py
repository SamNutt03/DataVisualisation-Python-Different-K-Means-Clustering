#Sam Nuttall - 201608203 - Data Mining and Visualisation Assignment 2

##For this code I reused my code for the created functions for Question 1 KMeans.py

import numpy as np
import random
import matplotlib.pyplot as plt

#Fix the seed value in order not to get new graphs every re-run
random.seed(42)
np.random.seed(42)

def ComputeDistance(x, y):
    #Function to measure eucluidian distances between points.
    #Input: points X and Y 
    #Output: euclidean distance between X and Y
    squared_diff = np.sum((x - y)**2) 
    return np.sqrt(squared_diff)

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

def clustername(x, k):
    #Runs k-means clustering and returns silhouette score.
    #Input: dataset x , number of clusters k
    #Output: clusters and silhouette score

    #use the other functions we specified in order to complete the tasks
    centroids = initialSelection(x, k)
    #loops computing new centroids and clusters until the centroids dont change
    while True:
        clusters = assignClusterIds(x, centroids)
        new_centroids = computeClusterRepresentatives(clusters)
        if np.allclose(centroids, new_centroids): # checking for convergence
            break # stops if centroids do not change 
        centroids = new_centroids
    score = silhouette_score(x, clusters)
    return clusters, score

def plot_silhouettee(scores):
    #Plots the relationship between k number of clusters and silhouette coefficient
    #Input: list of silhouette scores where array index corresponds to number of clusters
    #Outpit: plot of scores to clusters
    k_values = list(range(1, len(scores)+1))
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.title('Silhouette Score vs. Number of Clusters (Synthetic Data)')
    plt.grid(True)
    plt.savefig('KMeansSynthetic-Plot.png')  #Saves the plot to a png file in folder
    plt.show()

def main():
    #Main function that generates a synthetic dataset with same size as original data file

    try: #This part reads the data file and seperates the data into labels and features as done in the load_dataset function for previous code
        labels = []
        features = []
        with open("dataset", "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                labels.append(parts[0])
                features.append([float(x) for x in parts[1:]])
        features = np.array(features)
    except FileNotFoundError:
        print("Error: 'dataset' file not found.")
        return
    if len(features) < 2:
        print("Error: Not enough data points for clustering.")
        exit() 

    # Then it generates synthetic data with same shape as the data file using np.random with normal distribution
    num_samples, num_features = features.shape
    synthetic_data = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, num_features))

    # running clustering for k = 1 - 9
    scores = []
    for k in range(1, 10): #loop through for 10 clusters
        if k == 1:
            scores.append(0)
            continue
        _, score = clustername(synthetic_data, k)
        scores.append(score) #add the scores to the list where each index represents K number of clusters

    plot_silhouettee(scores) #plot the scores into a new png 

if __name__ == "__main__":
    main() #runs the main program when the script is run.