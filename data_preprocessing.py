import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson, probplot, zscore
from statsmodels.tsa.seasonal import STL
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

pd_df = pd.read_csv("weather_data.csv")
df = pd_df.to_numpy()
N = df.shape[0]
feature_cols = df.shape[1]-1

Y = df[:,-1].reshape(-1,1)
X = df[:, 0:feature_cols]

# Normalisation
def normalisation(X):
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)

    X_norm = (X - X_min)/(X_max - X_min)
    return X_norm

# Detect outliers that lies outside 1.5*IQR
def detect_out(data: np.ndarray) -> list:
    """
    data: numpy array, shape = [N, D] or [D]
    return:
        list of outlier indices, shape [K, D]
        list of outlier count in each feature, shape [K]
    """
    cols = data.shape[1]
    outliers = set()
    count = []

    def column_out(data_col):
        Q1 = np.percentile(data_col, 25)
        Q3 = np.percentile(data_col, 75)
        IQR = Q3-Q1
        upper_bound, lower_bound = Q3+1.5*IQR, Q1-1.5*IQR
        curr_out = set(np.where((data_col<lower_bound) | (data_col>upper_bound))[0])
        count.append(len(curr_out))
        return curr_out
    for i in range(cols):
        col = data[:, i]
        outliers = outliers | column_out(col)
    return outliers, count

# Yeo Johnson feature transformation
def feature_transform(data: np.ndarray, outlier_count: list):
    lambda_values = []
    transformed_arr = data.copy()
    for feature_idx in range(len(outlier_count)):
        # only transform data if there are outliers
        if outlier_count[feature_idx]:
            feature = data[:, feature_idx]
            # the 3rd feature, mean_sea_level_pressure's values are too high, hence we scaled it down by 1000x
            if feature_idx == 2:
                transformed_data, lambda_value = yeojohnson(normalisation(feature))
            else:
                transformed_data, lambda_value = yeojohnson(feature)
            transformed_arr[:, feature_idx] = transformed_data
            lambda_values.append(lambda_value)
    return transformed_arr, lambda_values

def seasonal_decompostion(data: np.ndarray, period: int):
    cols = data.shape[1]
    trend_arr, seasonal_arr, residual_arr = [], [], []
    for feature_idx in range(cols):
        feature = data[:, feature_idx]
        stl = STL(feature, period=period, robust=True).fit()
        trend_arr.append(stl.trend)
        seasonal_arr.append(stl.seasonal)
        residual_arr.append(stl.resid)
        seasonal_std = stl.seasonal.std()
        total_std = final_df["temperature"].std()
        strength = seasonal_std / total_std
        print(f"Seasonal strength (yearly): {strength:.2%}")
    trend = np.array(trend_arr).T
    seasonal = np.array(seasonal_arr).T
    residual = np.array(residual_arr).T


    return trend, seasonal, residual



# Remove outliers
def remove_outliers(X, Y, outlier_idx):
    X_clean = np.delete(X, list(outlier_idx), axis=0)
    Y_clean = np.delete(Y, list(outlier_idx), axis=0)

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

# Remove outliers
def remove_outliers(X, Y, outlier_idx):
    X_clean = np.delete(X, list(outlier_idx), axis=0)
    Y_clean = np.delete(Y, list(outlier_idx), axis=0)

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

# Multicollinearity Check using Variance Inflation Factor
def compute_vif(data, column_names):
    df = pd.DataFrame(data, columns = column_names)
    X = add_constant(df)
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def compare_vif_before_after(data, column_names):
    vif_before = compute_vif(data, column_names)

    _, outlier_counts = detect_out(data)
    transformed_X, _ = feature_transform(data, outlier_counts)

    vif_after = compute_vif(transformed_X, column_names)

    comparison = pd.merge(vif_before, vif_after, on = "feature", suffixes = ("_before", "_after"))
    return comparison

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


# heatmaps
def heatmaps(data):
    df = pd.DataFrame(data)
    corr = df.corr()

    variables = ["temperature", "wind speed", "mean sea level pressure",
                 "surface solar radiation", "surface thermal radiation", "total cloud cover"]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                xticklabels=variables, yticklabels=variables)
    plt.tight_layout()
    plt.title("Correlation Heatmap of Weather Data")

    plt.ion()
    plt.show()

# outlier detection before transformation using IQR
outliers_before_transformation = detect_out(X)[1]
# print(outliers_before_transformation)

# feature transformation (Yeo Johnson)
transformed_X, lambda_values = feature_transform(X, detect_out(X)[1])
# print(np.std(transformed_X, axis=0))


# validation of outlier after transformation using Z-score and IQR
outlier_count_after_transformation = detect_out(transformed_X)[1]

# print(outlier_count_after_transformation)

outlier_idx_after_transformation = detect_out(transformed_X)[0]
final_X, final_y = remove_outliers(transformed_X, Y, outlier_idx_after_transformation)
final_X = normalisation(final_X)

# visualise normal distribution
# multi_plots(X)
# multi_plots(final_X)
# print(X[:, [0, 2,3,4]])
# decomposed_X = seasonal_decompostion(transformed_X[:, [0, 1, 2, 3, 4]], 24)[0]
# multi_plots(decomposed_X)

#if __name__ == "__main__":
    # VIF Check
    #columns = pd_df.columns.tolist()[:-1:1]
    #print(columns)
    #vif_result = compute_vif(final_X, columns)
    #print(vif_result)

    #heatmaps(final_X)
    #final_df = pd.DataFrame(final_X, columns=columns)
    



