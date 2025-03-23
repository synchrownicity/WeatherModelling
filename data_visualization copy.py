import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson, probplot, zscore
from statsmodels.tsa.seasonal import STL
import seaborn as sns

pd_df = pd.read_csv("weather_data.csv")
df = pd_df.to_numpy()
N = df.shape[0]
feature_cols = df.shape[1]-1

Y = df[:,-1].reshape(-1,1)
X = df[:, 0:feature_cols]

# Detect outliers that lies outside 1.5*IQR
def detect_out(data: np.ndarray) -> list:
    """
    data: numpy array, shape = [N, D] or [D]
    return:
        list of outlier indices, shape [K, D]
        list of outlier count in each feature, shape [K]
    """
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

# Yeo Johnson feature transformation
def feature_transform(data: np.ndarray, outliers: list) -> list:
    lambda_values = []
    transformed_arr = data.copy()
    for feature_idx in range(len(outliers)):
        # only transform data if there are outliers
        if outliers[feature_idx]:
            feature = data[:, feature_idx]
            # the 3rd feature, mean_sea_level_pressure's values are too high, hence we scaled it down by 1000x
            if feature_idx == 2:
                transformed_data, lambda_value = yeojohnson(feature / 1000)
            else:
                transformed_data, lambda_value = yeojohnson(feature)
            transformed_arr[:, feature_idx] = transformed_data
            lambda_values.append(lambda_value)
    return transformed_arr, lambda_values

def seasonal_decompostion(data: np.ndarray, seasons: int):
    cols = data.shape[1]
    decomposed_feature = {}
    for feature_idx in range(cols):
        feature = data[:, feature_idx]
        stl = STL(feature, seasonal=seasons, robust=True).fit()
        trend = stl.trend
        seasonal = stl.seasonal
        residual = stl.resid
        decomposed_feature[feature_idx] = [trend, seasonal, residual]
    return decomposed_feature



# Remove outliers
def remove_outliers(X, Y, k):
    X_standard = standardiseData(X)
    outliers = np.any(np.abs(X_standard) > k, axis=1)

    X_clean = X[~outliers]
    Y_clean = Y[~outliers]

    return X_clean, Y_clean

# Z-score/ standardisation
def standardiseData(X):
    count = []
    standardised = zscore(X)
    for feature_idx in range(standardised.shape[1]):
        feature = standardised[:, feature_idx]
        outlier_idx = np.where((zscore(feature) < -3) | (zscore(feature) > 3))[0]
        count.append(len(outlier_idx))
    return zscore(X), count

# Normalisation
def normalisation(X):
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)

    X_norm = (X - X_min)/(X_max - X_min)
    return X_norm

# QQ-plots
def qq_plot(data: np.ndarray):
    probplot(data, dist="norm", plot=plt)
    plt.title('Normal Q-Q plot')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Ordered Values')
    plt.grid(True)
    plt.show()

def multi_plots(data):
    col = data.shape[1]
    fig, axs = plt.subplots(col, 1, figsize=(10, 12))
    ax_lst = axs.flat
    # Flatten axs (to iterate over it easily, even if 2D)
    for i in range(col):
        ax = ax_lst[i]
        feature = data[:, i]
        ax.plot(feature)
    plt.tight_layout()
    plt.show()


# outlier detection before transformation using IQR
outliers_before_transformation = detect_out(X)[1]
print(outliers_before_transformation)

# feature transformation (Yeo Johnson)
transformed_X, lambda_values = feature_transform(X, detect_out(X)[0])

# validation of outlier after transformation using Z-score and IQR
outliers_after_transformation = detect_out(transformed_X)[1]
print(outliers_after_transformation)

# visualise normal distribution
# multi_plots(X)
# multi_plots(transformed_X)
print(X[:, [0, 2,3,4]])
print(seasonal_decompostion(X[:, [0, 2,3,4]], 8000))

# can try plotting X[:, i], i from 0 to 5
# qq_plot(X[:, 0])
# qq_plot(X_norm[:, 1])

# can see on boxplot too --> yeojohnson on normalized data produce better distribution
# plt.boxplot(X_norm)
# plt.show()

# feature_transform(X, detect_out(X))
# plt.boxplot(X)
# plt.show()
# can try plotting X[:, i], i from 0 to 5
# qq_plot(X[:, 0])

#heatmaps
def heatmaps(data):
    df = pd.DataFrame(data)
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.title("Correlation Heatmap of Weather Data")
    plt.show()

heatmaps(pd_df)
