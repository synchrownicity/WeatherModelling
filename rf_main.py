from data_preprocessing import final_X, final_y
from rf_train_model import train_and_save_model
from rf_train_evaluate import evaluate_model
from rf_train_test_split import groupwise_split

# Perform groupwise train-test split
X_train, X_test, y_train, y_test = groupwise_split(final_X, final_y)

# Train and save model
train_and_save_model(X_train, y_train, model_path="random_forest_model.pkl")

# Evaluate the saved model
evaluate_model("random_forest_model.pkl", X_test, y_test)