import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import stats as stats, yeojohnson

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
def feature_transform(data: np.ndarray, outliers: list) -> np.ndarray:
    lambda_values = []
    for feature_idx in range(len(outliers)):
        if outliers[feature_idx]:
            feature = data[:, feature_idx]
            print(feature)
            x = feature.astype(np.float64, copy=False)
            # data[:, feature_idx] = transformed_data
            # lambda_values.append(lambda_value)
    return lambda_values

# QQ-plots after log transformation
def qq_plot(data: np.ndarray):
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Normal Q-Q plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.grid(True)
    plt.show()

# feature_transform(X, detect_out(X))
# plt.boxplot(X)
# plt.show()
# can try plotting X[:, i], i from 0 to 5
# qq_plot(X[:, 0])

a, b = yeojohnson(X[:, 2])
print(a, b)