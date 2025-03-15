import pandas as pd

df = pd.read_csv("weather_data.csv")
N = df.shape[0]
feature_cols = df.shape[1]-1

Y = df.iloc[:,-1]
X = df.iloc[:, 0:feature_cols]

