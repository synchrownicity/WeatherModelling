import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson, probplot, zscore

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
    count = []
    def column_out(data_col, out=outliers):
        Q1 = np.percentile(data_col, 25)
        Q3 = np.percentile(data_col, 75)
        IQR = Q3-Q1
        upper_bound, lower_bound = Q3+1.5*IQR, Q1-1.5*IQR
        curr_out = list(np.where((data_col<lower_bound) | (data_col>upper_bound))[0])
        out.append(curr_out)
        count.append(len(curr_out))

    for i in range(cols):
        col = data[:, i]
        column_out(col)
    return outliers, count

# Normalisation
def normalisation(X):
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)

    X_norm = (X - X_min)/(X_max - X_min)
    return X_norm

# Function transform feature datapoints using Yeo Johnson if that given feature contains outliers.
# Function returns maximum log-likelihood parameter of Yeo Johnson
def feature_transform(data: np.ndarray, outliers: list) -> np.ndarray:
    lambda_values = []
    for feature_idx in range(len(outliers)):
        # only transform data if there are outliers
        if outliers[feature_idx]:
            feature = data[:, feature_idx]
            transformed_data, lambda_value = yeojohnson(feature)
            # Yeo Johnson shifted the min and max values out of 0 to 1 range, hence normalization is performed again
            data[:, feature_idx] = normalisation(transformed_data)
            lambda_values.append(lambda_value)
    return lambda_values

# QQ-plots after log transformation
def qq_plot(data: np.ndarray):
    probplot(data, dist="norm", plot=plt)
    plt.title('Normal Q-Q plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.grid(True)
    plt.show()

# Z-score standardisation
def standardiseData(X):
    count = []
    standardised = zscore(X)
    for feature_idx in range(standardised.shape[1]):
        feature = standardised[:, feature_idx]
        outlier_idx = np.where((zscore(feature) < -3) | (zscore(feature) > 3))[0]
        count.append(len(outlier_idx))
    return zscore(X), count

#def remove_outliers(X, Y , k):
    #X_standard = standardiseData(X)
    #outliers = np.any(np.abs(X_standard) > k, axis=1)

    #X_clean = X[~outliers]
    #Y_clean = Y[~outliers]

    #return X_clean, Y_clean


# outlier detection before transformation using IQR
print(detect_out(X)[1])

# feature scaling
X_norm = normalisation(X)
Y_norm = normalisation(Y)

# feature transformation (Yeo Johnson)
feature_transform(X_norm, detect_out(X_norm)[0])

# validation of outlier after transformation using Z-score and IQR
X_standard, Y_standard = standardiseData(X_norm)[0], standardiseData(Y_norm)[0]
print(detect_out(X_norm)[1]) # by IQR
print(standardiseData(X_norm)[1]) # by z-score
# print(len(np.where((X_standard < -3) | (X_standard > 3))))



# can try plotting X[:, i], i from 0 to 5
# qq_plot(X[:, 0])
# qq_plot(X_norm[:, 1])

# can see on boxplot too --> yeojohnson on normalized data produce better distribution
# plt.boxplot(X_norm)
# plt.show()