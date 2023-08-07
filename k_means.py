import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output

file = "all_seasons.csv"
features = ["player_height", "player_weight", "draft_year", "draft_round", "draft_number"]
 
def initialise_data(file, features):
    players = pd.read_csv(file)
    players = players.dropna(subset=features) # Removes NULL/missing values
    data = players[features].copy()
    data = ((data - data.min()) / (data.max() - data.min())) * 99 + 1 # Data scaling
    return data

def initialise_centroids(data, k):
    centroids = []
    for i in range(k): 
        centroid = data.apply(lambda x: float(x.sample()))
        centroids.append(centroid)
    return pd.concat(centroids, axis=1)

def get_labels(data, centroids):
    distances = centroids.apply(lambda x: np.sqrt(((data - x) ** 2).sum(axis=1)))
    return distances.idxmin(axis=1)

def update_centroids(data, labels):
    return data.groupby(labels).apply(lambda x: np.exp(np.log(x).mean())).T # Splits dataframe by cluster, finds geometric mean of each feature

def plot_clusters(data, labels, centroids, iteration):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    centroids_2d = pca.transform(centroids.T)
    clear_output(wait=True)
    plt.title(f'Iteration {iteration}')
    plt.scatter(x=data_2d[:,0], y=data_2d[:,1], c=labels)
    plt.scatter(x=centroids_2d[:,0], y=centroids_2d[:,1])
    plt.show()

def main():
    file = "all_seasons.csv"
    features = ["player_height", "player_weight", "draft_year", "draft_round", "draft_number"]
    data = initialise_data(file, features)

    max_iters = 100
    k = 3

    centroids = initialise_centroids(data, k)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iters and not centroids.equals(old_centroids):
        old_centroids = centroids
        labels = get_labels(data, centroids)
        centroids = update_centroids(data, labels)
        plot_clusters(data, labels, centroids, iteration)
        iteration += 1

    return centroids

if __name__ == "__main__":
    centroids = main()
    print(centroids)
