import pandas as pd
import numpy as np

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

data = initialise_data(file, features)
centroids = initialise_centroids(data, 5)
