import data_preprocessing as dp
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from keras import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd


# Perform groupwise train-test split
# X_train, X_test, y_train, y_test = groupwise_split(final_X, final_y)

# Train and save model
#train_and_save_model(X_train, y_train, model_path="random_forest_model.pkl")

# Evaluate the saved model
#evaluate_model("random_forest_model.pkl", X_test, y_test)


# Number of data points in train-validation and test sets
data_N = len(dp.final_y); print(f"\nTotal no. of datapoints: {data_N}")
train_val_N = int(data_N*0.8); print(f"80% datapoints in (train + val) sets: {train_val_N}")
val_N = int(data_N*0.1); print(f"10% datapoints in validatate set: {val_N}")
test_N = int(data_N*0.2); print(f"20% datapoints in test set: {test_N}\n")

# Train-Validation, test sets
train_val_X = dp.final_X[:train_val_N, :]
train_val_y = dp.final_y[:train_val_N]
test_X = dp.final_X[train_val_N:, :]
test_y = dp.final_y[train_val_N:]

# Hyperparameters to be tuned
N = [24, 168, 720] # half-daily, daily, weekly, monthly, yearly
delays = [1, 6, 24] # 1hr, 6hr, 24hr


def create_time_series_features(X, y, n, delay):
    """
    Create feature vectors for random forest by combining lagged observations
    """
    X_new = []
    y_new = []

    for i in range(n, len(X) - delay):
        # For each time step, include n previous observations as features
        features = X[i-n:i, :].flatten()
        target = y[i + delay]
        X_new.append(features)
        y_new.append(target)

    return np.array(X_new), np.array(y_new).ravel()

results = {
    'N': [],
    'delay': [],
    'fold': [],
    'test_rmse': [],
    'test_mae': []
}

def rf_model(N, delays, train_val_X=train_val_X, train_val_y=train_val_y, test_X=test_X, test_y=test_y):
    for delay in delays:
        # In each delay
        for n in N:
            # For each N parameters
            gap = n + delay
            # Split dataset into train and val
            max_split = 4
            tss = TimeSeriesSplit(n_splits=max_split, test_size=9774, gap=gap)
            fold_rmse = []
            fold_mae = []
            for idx, (train_index, val_index) in enumerate(tss.split(train_val_X)):
                # In each fold
                X_train, X_val = train_val_X[train_index, :], train_val_X[val_index,:]
                y_train, y_val = train_val_y[train_index], train_val_y[val_index]



                #validate
                # create rolling window for prediction
                X_train_rf, y_train_rf = create_time_series_features(X_train, y_train, n, delay)
                X_val_rf, y_val_rf = create_time_series_features(X_val, y_val, n, delay)
                X_test_rf, y_test_rf = create_time_series_features(test_X, test_y, n, delay)

                # fit training and validation data into model
                model = RandomForestRegressor(
                    n_estimators = 25, # Number of trees in the forest
                    max_depth = 5, # Maximum depth of the tree
                    min_samples_split = 15, # Minimum number of samples required to split an internal node
                    random_state = 42 # Random state for reproducibility
                )

                    # train-validate model (no early stopping)
                history = model.fit(X_train_rf, y_train_rf)

                    # test model on test set
                pred = model.predict(X_val_rf)
                # Validate
                val_pred = model.predict(X_val_rf)
                val_rmse = np.sqrt(mean_squared_error(y_val_rf, val_pred))
                val_mae = mean_absolute_error(y_val_rf, val_pred)
                print(f"    Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")

                # Test
                test_pred = model.predict(X_test_rf)
                test_rmse = np.sqrt(mean_squared_error(y_test_rf, test_pred))
                test_mae = mean_absolute_error(y_test_rf, test_pred)
                print(f"    Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

                # Add the test metrics to the history dictionary
                # Save results
                results['N'].append(n)
                results['delay'].append(delay)
                results['fold'].append(idx)
                results['test_rmse'].append(test_rmse)
                results['test_mae'].append(test_mae)

                fold_rmse.append(test_rmse)
                fold_mae.append(test_mae)
                joblib.dump(model, f"rf_model_N{n}_delay{delay}_fold{idx}.pkl")
            avg_rmse = np.mean(fold_rmse)
            avg_mae = np.mean(fold_mae)
            print(f"  Average for N={n}, delay={delay}: RMSE={avg_rmse:.4f}, MAE={avg_mae:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv('rf_model_results.csv', index=False)

    # Calculate and print average results by N and delay
    print("\nAverage Results by Configuration:")
    avg_results = results_df.groupby(['N', 'delay']).agg({
        'test_rmse': 'mean',
        'test_mae': 'mean'
    }).reset_index()

    print(avg_results)

    return avg_results



def plot_metrics_vs_N(results_df):
    """
    Plot RMSE and MAE vs N values for different delay configurations

    Parameters:
    results_df: DataFrame containing the average results with columns 'N', 'delay', 'test_rmse', 'test_mae'
    """
    # Create figure with two subplots side by side
    plt.figure(figsize=(16, 6))

    # Plot RMSE vs N
    plt.subplot(1, 2, 1)
    for delay in results_df['delay'].unique():
        delay_data = results_df[results_df['delay'] == delay]
        plt.plot(delay_data['N'], delay_data['test_rmse'], 'o-', linewidth=2, label=f'Delay = {delay}')

    plt.title('RMSE vs Window Size (N)', fontsize=14)
    plt.xlabel('Window Size (N)', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Using log scale for N since values vary widely
    plt.xticks(results_df['N'].unique(), labels=results_df['N'].unique())  # Force labels to be actual values
    plt.legend()

    # Plot MAE vs N
    plt.subplot(1, 2, 2)
    for delay in results_df['delay'].unique():
        delay_data = results_df[results_df['delay'] == delay]
        plt.plot(delay_data['N'], delay_data['test_mae'], 'o-', linewidth=2, label=f'Delay = {delay}')

    plt.title('MAE vs Window Size (N)', fontsize=14)
    plt.xlabel('Window Size (N)', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')  # Using log scale for N since values vary widely
    plt.xticks(results_df['N'].unique(), labels=results_df['N'].unique())  # Force labels to be actual values
    plt.legend()

    # Add overall title and adjust layout
    plt.suptitle('Random Forest Performance vs Window Size', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the figure
    plt.savefig('rf_performance_vs_N.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Also plot metrics vs delay (alternative view)
    plt.figure(figsize=(16, 6))

    # Plot RMSE vs delay
    plt.subplot(1, 2, 1)
    for n in results_df['N'].unique():
        n_data = results_df[results_df['N'] == n]
        plt.plot(n_data['delay'], n_data['test_rmse'], 'o-', linewidth=2, label=f'N = {n}')

    plt.title('RMSE vs Prediction Delay', fontsize=14)
    plt.xlabel('Prediction Delay', fontsize=12)
    plt.ylabel('Root Mean Squared Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(results_df['delay'].unique())
    plt.legend()

    # Plot MAE vs delay
    plt.subplot(1, 2, 2)
    for n in results_df['N'].unique():
        n_data = results_df[results_df['N'] == n]
        plt.plot(n_data['delay'], n_data['test_mae'], 'o-', linewidth=2, label=f'N = {n}')

    plt.title('MAE vs Prediction Delay', fontsize=14)
    plt.xlabel('Prediction Delay', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(results_df['delay'].unique())
    plt.legend()

    # Add overall title and adjust layout
    plt.suptitle('Random Forest Performance vs Prediction Delay', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    # Save the figure
    plt.savefig('rf_performance_vs_delay.png', dpi=300, bbox_inches='tight')
    plt.show()


  rf_model(N=N, delays=delays)
  plot_metrics_vs_N(avg_results)