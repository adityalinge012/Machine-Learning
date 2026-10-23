import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Implement_Dataset import (
        T_X_Train,
        T_Y_Train,
        T_Y_Val,
        T_Y_Test,
        Y_Train_Predict,
        Y_Val_Predict,
        Y_Test_Predict
        )

# =========================================================
# Ex_2_1-f
# Performance Metrics and Quality of Fit - Batch Gradient Descent
# =========================================================
# 1. Function to calculate performance metrics
def T_calculate_metrics(T_y_actual, T_y_pred, T_n, T_p):
    T_mae = mean_absolute_error(T_y_actual, T_y_pred)
    T_mse = mean_squared_error(T_y_actual, T_y_pred)
    T_rmse = np.sqrt(T_mse)
    T_r2 = r2_score(T_y_actual, T_y_pred)
    T_adjusted_r2 = 1 - ((1 - T_r2) * (T_n - 1) / (T_n - T_p - 1))
    return T_mae, T_mse, T_rmse, T_r2, T_adjusted_r2
# 2. Number of predictors in SLR
T_p = T_X_Train.shape[1]
# 3. Calculate metrics for each dataset using BGD predictions
T_train_metrics = T_calculate_metrics(T_Y_Train, Y_Train_Predict, len(T_Y_Train), T_p)
T_val_metrics = T_calculate_metrics(T_Y_Val, Y_Val_Predict, len(T_Y_Val), T_p)
T_test_metrics = T_calculate_metrics(T_Y_Test, Y_Test_Predict, len(T_Y_Test), T_p)
# 4. Display performance metrics
print("\n==============================================")
print("Performance Metrics - SLR using Batch Gradient Descent")
print("==============================================")


print("\nNumber of Predictors (p) :", T_p)
print("\nPerformance Metrics:")
print("-----------------------------------------------------------------------")
print(
f"{'Dataset':<12}{'MAE':>12}{'MSE':>12}"
f"{'RMSE':>12}{'R2':>12}{'Adjusted R2':>15}"
)
print("-----------------------------------------------------------------------")
for T_name, T_values in [
("Training", T_train_metrics),
("Validation", T_val_metrics),
("Testing", T_test_metrics)
]:
    T_formatted_values = "".join(f"{T_value:12.6f}" for T_value in T_values)
print(f"{T_name:<12}{T_formatted_values}")
print("-----------------------------------------------------------------------")
# 5. Display R2 values for interpreting quality of fit
print("\nQuality of Fit based on R2 (BGD):")
print("--------------------------------")
print("Training R2 =", round(T_train_metrics[3], 4))
print("Validation R2 =", round(T_val_metrics[3], 4))
print("Testing R2 =", round(T_test_metrics[3], 4))
print("R2 closer to 1 indicates greater explanatory power.")
print("Lower MAE, MSE and RMSE indicate smaller prediction errors.")
print("Compare Training, Validation and Testing results to assess generalization.")