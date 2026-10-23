import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Splitting_Dataset import T_X_Train, T_Y_Train
# =========================================================
# Ex_2_1-j: 5-Fold Cross-Validation - BGD for SLR
# =========================================================
# 1. Extract the Training subset prepared in Ex_2_1-b
T_X = T_X_Train["TV"].to_numpy(dtype=float)
T_y = T_Y_Train.to_numpy(dtype=float)
# 2. Set cross-validation and BGD parameters


T_kfold = KFold(n_splits=5, shuffle=True, random_state=42)
T_learning_rate = 0.1
T_tolerance = 1e-12
T_gradient_tolerance = 1e-10
T_max_epochs = 100000
T_fold_results = []
# 3. Train one fresh BGD model in each fold
for T_fold_number, (T_train_index, T_val_index) in enumerate(T_kfold.split(T_X), start=1):
    T_X_fold_train = T_X[T_train_index]
    T_y_fold_train = T_y[T_train_index]
    T_X_fold_val = T_X[T_val_index]
    T_y_fold_val = T_y[T_val_index]
    # Standardize TV using only the fold-training subset
    T_X_mean = np.mean(T_X_fold_train)
    T_X_std = np.std(T_X_fold_train)
    T_X_fold_train_scaled = (T_X_fold_train - T_X_mean) / T_X_std
    T_X_fold_val_scaled = (T_X_fold_val - T_X_mean) / T_X_std
    T_n_fold_train = len(T_X_fold_train)
    # Initialize fresh weights for this fold
    T_w0 = 0.0
    T_w1 = 0.0
# Batch Gradient Descent iterations
for T_epoch in range(1, T_max_epochs + 1):
    T_y_fold_train_pred = T_w0 + T_w1 * T_X_fold_train_scaled
    T_error = T_y_fold_train - T_y_fold_train_pred
    T_cost = np.sum(T_error ** 2) / (2 * T_n_fold_train)
    T_gradient_w0 = -np.mean(T_error)
    T_gradient_w1 = -np.mean(T_X_fold_train_scaled * T_error)
    T_w0 = T_w0 - T_learning_rate * T_gradient_w0
    T_w1 = T_w1 - T_learning_rate * T_gradient_w1
    T_y_fold_train_pred_new = T_w0 + T_w1 * T_X_fold_train_scaled
    T_error_new = T_y_fold_train - T_y_fold_train_pred_new
    T_new_cost = np.sum(T_error_new ** 2) / (2 * T_n_fold_train)
    T_gradient_w0_new = -np.mean(T_error_new)
    T_gradient_w1_new = -np.mean(T_X_fold_train_scaled * T_error_new)
    T_cost_change = abs(T_cost - T_new_cost)
    T_gradient_magnitude = np.sqrt(T_gradient_w0_new ** 2 + T_gradient_w1_new ** 2)
if T_cost_change < T_tolerance and T_gradient_magnitude < T_gradient_tolerance:
# Predict the held-out fold using its fold-training scaling values

    T_y_fold_val_pred = T_w0 + T_w1 * T_X_fold_val_scaled
    # Calculate validation metrics for this fold
    T_mae = mean_absolute_error(T_y_fold_val, T_y_fold_val_pred)
    T_mse = mean_squared_error(T_y_fold_val, T_y_fold_val_pred)
    T_rmse = np.sqrt(T_mse)
    T_r2 = r2_score(T_y_fold_val, T_y_fold_val_pred)
    T_fold_results.append([
    T_fold_number, len(T_X_fold_train), len(T_X_fold_val), T_epoch,
    T_mae, T_mse, T_rmse, T_r2
    ])
# 4. Display metrics for each fold
T_fold_table = pd.DataFrame(T_fold_results, columns=[
"Fold", "Training n", "Validation n", "Epochs",
"MAE", "MSE", "RMSE", "R2"
])
print("\n==============================================")
print("5-Fold Cross-Validation - SLR using BGD")
print("==============================================")
print("Training observations used for CV:", len(T_X))
print("Number of folds : 5")
print("BGD learning rate :", T_learning_rate)
print("\nMetrics for Individual Folds:")
print(T_fold_table.to_string(index=False, formatters={
"MAE": "{:.6f}".format,
"MSE": "{:.6f}".format,
"RMSE": "{:.6f}".format,
"R2": "{:.6f}".format
}))
# 5. Calculate mean and population standard deviation across folds
T_metric_values = T_fold_table[["MAE", "MSE", "RMSE", "R2"]].to_numpy(dtype=float)
T_mean_metrics = np.mean(T_metric_values, axis=0)
T_std_metrics = np.std(T_metric_values, axis=0)
T_cv_summary = pd.DataFrame({
"Metric": ["MAE", "MSE", "RMSE", "R2"],
"Mean across folds": T_mean_metrics,
"Std. Dev. across folds": T_std_metrics
})
print("\nCross-Validation Summary:")
print(T_cv_summary.to_string(index=False, formatters={
"Mean across folds": "{:.6f}".format,
"Std. Dev. across folds": "{:.6f}".format
}))
print("\nMean Cross-Validation R2 :", round(T_mean_metrics[3], 6))


print("R2 standard deviation :", round(T_std_metrics[3], 6))
print("Lower MAE, MSE and RMSE indicate smaller prediction errors.")
print("A lower fold-to-fold standard deviation indicates more stable performance.")
print("The separate Validation and Testing datasets remain unused in this CV exercise.")