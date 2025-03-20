import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

pd_df = pd.read_csv("weather_data.csv")
df = pd_df.to_numpy()
N = df.shape[0]
feature_cols = df.shape[1]-1

Y = df[:,-1].reshape(-1,1)
X = df[:, 0:feature_cols]

# Detect outliers that lies outside 1.5*IQR --> returns a list of outlier indices for each feature
def detect_out(data: np.ndarray) -> list:
    cols = data.shape[1]
    outliers = []
    def column_out(data_col, out=outliers):
        Q1 = np.percentile(data_col, 25)
        Q3 = np.percentile(data_col, 75)
        IQR = Q3-Q1
        upper_bound, lower_bound = Q3+1.5*IQR, Q1-1.5*IQR
        curr_out = list(np.where((data_col<lower_bound) | (data_col>upper_bound))[0])
        out.append(curr_out)

    for i in range(cols):
        col = data[:, i]
        column_out(col)
    return outliers

# log feature datapoints if that given feature contains outliers
def feature_log(data: np.ndarray, outliers: list) -> np.ndarray:
    for feature_idx in range(len(outliers)):
        if outliers[feature_idx]:
            feature = data[:, feature_idx]
            if np.any(feature == 0):
                # add a small value (1e-10) to entire data set
                feature = feature + 1e-10
            logged_feature = np.log(feature)
            data[:, feature_idx] = logged_feature

# QQ-plots after log transformation
def qq_plot(data: np.ndarray):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Normal Q-Q plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.grid(True)
    plt.show()

feature_log(X, detect_out(X))
# can try plotting X[:, i], i from 0 to 5
qq_plot(X[:, 0])
qq_plot(X[:, 1])


# Z-score standardisation
def standardiseData(X):
    return stats.zscore(X)

def remove_outliers(X, Y , k):
    X_standard = standardiseData(X)
    outliers = np.any(np.abs(X_standard) > k, axis=1)

    X_clean = X[~outliers]
    Y_clean = Y[~outliers]

    return X_clean, Y_clean

# Normalisation 
def normalisation(X):
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)

    X_norm = (X - X_min)/(X_max - X_min)
    return X_norm

X_standard, Y_standard = standardiseData(X), standardiseData(Y)

X_norm = normalisation(X)
Y_norm = normalisation(Y)
# Remove outliers beyond a certain threshold -3 and 3
X_clean, Y_clean = remove_outliers(X_standard, Y_standard, 3)


