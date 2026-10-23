import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Implement_Dataset import (
    T_X_Train,
    T_X_Val,
    T_X_Test,
    T_Y_Train,
    T_Y_Val,
    T_Y_Test,
    Y_Train_Predict,
    Y_Val_Predict,
    Y_Test_Predict
)

# =========================================================
# Ex_2_1-e
# Performance Metrics and Quality of Fit - Scikit-learn
# =========================================================


# 1. Function to calculate performance metrics
def T_calculate_metrics(T_y_actual, T_y_pred, T_n, T_p):

    # Mean Absolute Error
    T_mae = mean_absolute_error(
        T_y_actual,
        T_y_pred
    )

    # Mean Squared Error
    T_mse = mean_squared_error(
        T_y_actual,
        T_y_pred
    )

    # Root Mean Squared Error
    T_rmse = np.sqrt(T_mse)

    # R2 Score
    T_r2 = r2_score(
        T_y_actual,
        T_y_pred
    )

    # Adjusted R2
    T_adjusted_r2 = 1 - (
        ((1 - T_r2) * (T_n - 1))
        / (T_n - T_p - 1)
    )

    return (
        T_mae,
        T_mse,
        T_rmse,
        T_r2,
        T_adjusted_r2
    )


# =========================================================
# 2. Number of predictors
# =========================================================

T_p = T_X_Train.shape[1]


# =========================================================
# 3. Calculate Training Metrics
# =========================================================

T_train_metrics = T_calculate_metrics(
    T_Y_Train,
    Y_Train_Predict,
    len(T_Y_Train),
    T_p
)


# =========================================================
# 4. Calculate Validation Metrics
# =========================================================

T_val_metrics = T_calculate_metrics(
    T_Y_Val,
    Y_Val_Predict,          # CORRECTED
    len(T_Y_Val),
    T_p
)


# =========================================================
# 5. Calculate Testing Metrics
# =========================================================

T_test_metrics = T_calculate_metrics(
    T_Y_Test,
    Y_Test_Predict,
    len(T_Y_Test),
    T_p
)


# =========================================================
# 6. Store metrics in a table
# =========================================================

T_metrics = np.array([
    T_train_metrics,
    T_val_metrics,
    T_test_metrics
])


# =========================================================
# Display Performance Metrics
# =========================================================

print("\n==============================================")
print("Performance Metrics - SLR using Scikit-learn")
print("==============================================")


print("\nNumber of Predictors (p) :", T_p)


print("\nPerformance Metrics:")
print("--------------------------------------------------------------------------")

print(
    "Dataset       MAE          MSE         RMSE        R2       Adjusted R2"
)

print("--------------------------------------------------------------------------")


# Training
print(
    f"Training   "
    f"{T_train_metrics[0]:10.6f} "
    f"{T_train_metrics[1]:10.6f} "
    f"{T_train_metrics[2]:10.6f} "
    f"{T_train_metrics[3]:8.6f} "
    f"{T_train_metrics[4]:11.6f}"
)


# Validation
print(
    f"Validation "
    f"{T_val_metrics[0]:10.6f} "
    f"{T_val_metrics[1]:10.6f} "
    f"{T_val_metrics[2]:10.6f} "
    f"{T_val_metrics[3]:8.6f} "
    f"{T_val_metrics[4]:11.6f}"
)


# Testing
print(
    f"Testing    "
    f"{T_test_metrics[0]:10.6f} "
    f"{T_test_metrics[1]:10.6f} "
    f"{T_test_metrics[2]:10.6f} "
    f"{T_test_metrics[3]:8.6f} "
    f"{T_test_metrics[4]:11.6f}"
)


print("--------------------------------------------------------------------------")


# =========================================================
# 7. Quality of Fit Based on R2
# =========================================================

print("\nQuality of Fit based on R2:")
print("--------------------------------")


print(
    "Training R2   =",
    round(T_train_metrics[3], 4)
)


print(
    "Validation R2 =",
    round(T_val_metrics[3], 4)
)


print(
    "Testing R2    =",
    round(T_test_metrics[3], 4)
)


print(
    "\nHigher R2 values indicate greater explanatory "
    "power of the model."
)


print(
    "Lower MAE, MSE and RMSE values indicate smaller "
    "prediction errors."
)
