import pandas as pd
import numpy as np

players = pd.read_csv("all_seasons.csv")

features = ["player_height", "player_weight", "draft_year", "draft_round", "draft_number"]
players = players.dropna(subset=features) # Removes NULL/missing values
data = players[features].copy()


data = ((data - data.min()) / (data.max() - data.min())) * 99 + 1 # Data scaling
print(data.describe())
