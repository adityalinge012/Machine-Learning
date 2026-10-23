import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from Splitting_Dataset import (
    T_y_train,
    T_y_val,
    T_y_test
)

from BGD import (
    T_y_train_pred,
    T_y_val_pred,
    T_y_test_pred
)

# =========================================================
# Ex_2_2-d: Performance Metrics and Quality of Fit - BGD
# =========================================================

# 1. Number of independent predictors in the MLR model
T_p = 3

# 2. Define a function to calculate regression performance metrics
def T_calculate_metrics(T_y_actual, T_y_predicted, T_predictor_count):

    T_n = len(T_y_actual)

    T_mae = mean_absolute_error(T_y_actual, T_y_predicted)
    T_mse = mean_squared_error(T_y_actual, T_y_predicted)
    T_rmse = np.sqrt(T_mse)
    T_r2 = r2_score(T_y_actual, T_y_predicted)

    if T_n > T_predictor_count + 1:
        T_adjusted_r2 = 1 - (
            (1 - T_r2)
            * (T_n - 1)
            / (T_n - T_predictor_count - 1)
        )
    else:
        T_adjusted_r2 = np.nan

    return {
        "MAE": T_mae,
        "MSE": T_mse,
        "RMSE": T_rmse,
        "R2": T_r2,
        "Adjusted R2": T_adjusted_r2
    }


# 3. Calculate metrics for Training, Validation, and Testing datasets
T_training_metrics = T_calculate_metrics(
    T_y_train, T_y_train_pred, T_p
)

T_validation_metrics = T_calculate_metrics(
    T_y_val, T_y_val_pred, T_p
)

T_testing_metrics = T_calculate_metrics(
    T_y_test, T_y_test_pred, T_p
)


# 4. Display metric comparison
T_metrics_table = pd.DataFrame(
    [
        T_training_metrics,
        T_validation_metrics,
        T_testing_metrics
    ],
    index=["Training", "Validation", "Testing"]
)

print("\nPerformance Metrics - MLR using BGD")
print("====================================")
print(
    T_metrics_table.to_string(
        float_format=lambda T_value: f"{T_value:.6f}"
    )
)


# 5. Interpret the quality of fit
print("\nQuality-of-Fit Interpretation:")
print("Lower MAE, MSE, and RMSE indicate smaller prediction errors.")
print(
    "An R2 value closer to 1 indicates that the model "
    "explains more variation in Sales."
)
print(
    "A negative R2 indicates performance worse than "
    "predicting the mean Sales value."
)
print(
    "Compare Training and Testing metrics to assess generalization."
)