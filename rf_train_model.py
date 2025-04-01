import joblib
from sklearn.ensemble import RandomForestRegressor

def train_and_save_model(X_train, y_train, model_path="random_forest_model.pkl"):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.ravel())
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")
    return rf