import numpy as np
import pandas as pd

from Splitting_Dataset import (
    T_X_train, T_X_val, T_X_test,
    T_y_train, T_y_val, T_y_test
)

# =========================================================
# Ex_2_2-c: Multiple Linear Regression using Batch Gradient
# Descent (BGD)
# =========================================================

# 1. Convert Training, Validation, and Testing data to arrays
T_X_train_array = T_X_train.to_numpy(dtype=float)
T_X_val_array = T_X_val.to_numpy(dtype=float)
T_X_test_array = T_X_test.to_numpy(dtype=float)

T_y_train_array = T_y_train.to_numpy(dtype=float)

# 2. Standardize predictors using Training data only
T_X_mean = np.mean(T_X_train_array, axis=0)
T_X_std = np.std(T_X_train_array, axis=0)

T_X_std[T_X_std == 0] = 1.0

T_X_train_scaled = (T_X_train_array - T_X_mean) / T_X_std
T_X_val_scaled = (T_X_val_array - T_X_mean) / T_X_std
T_X_test_scaled = (T_X_test_array - T_X_mean) / T_X_std

# 3. Add a column of ones for the intercept
T_n_train = len(T_y_train_array)

T_X_train_design = np.column_stack(
    (np.ones(T_n_train), T_X_train_scaled)
)

# 4. Initialize weights and BGD parameters
T_w_scaled = np.zeros(T_X_train_design.shape[1])

T_learning_rate = 0.01
T_max_epochs = 100000
T_tolerance = 1e-12
T_gradient_tolerance = 1e-8

T_previous_cost = 0.5 * np.mean(
    (T_X_train_design @ T_w_scaled - T_y_train_array) ** 2
)

T_epochs_completed = 0

# 5. Train the model using Batch Gradient Descent
for T_epoch in range(1, T_max_epochs + 1):

    T_y_train_pred_scaled = T_X_train_design @ T_w_scaled

    T_errors = T_y_train_pred_scaled - T_y_train_array

    T_gradient = (
        T_X_train_design.T @ T_errors
    ) / T_n_train

    T_w_scaled = T_w_scaled - T_learning_rate * T_gradient

    T_y_train_pred_scaled = T_X_train_design @ T_w_scaled

    T_current_cost = 0.5 * np.mean(
        (T_y_train_pred_scaled - T_y_train_array) ** 2
    )

    T_gradient_current = (
        T_X_train_design.T
        @ (T_y_train_pred_scaled - T_y_train_array)
    ) / T_n_train

    T_epochs_completed = T_epoch

    if (
        abs(T_previous_cost - T_current_cost) < T_tolerance
        and np.max(np.abs(T_gradient_current))
        < T_gradient_tolerance
    ):
        break

    T_previous_cost = T_current_cost

# 6. Convert standardized weights to the original predictor scale
T_w_original = np.zeros_like(T_w_scaled)

T_w_original[1:] = T_w_scaled[1:] / T_X_std

T_w_original[0] = T_w_scaled[0] - np.sum(
    T_w_scaled[1:] * T_X_mean / T_X_std
)

T_w0 = float(T_w_original[0])
T_w1 = float(T_w_original[1])
T_w2 = float(T_w_original[2])
T_w3 = float(T_w_original[3])

# 7. Generate predictions in the original feature scale
T_y_train_pred = (
    T_w0 + T_X_train_array @ T_w_original[1:]
)

T_y_val_pred = (
    T_w0 + T_X_val_array @ T_w_original[1:]
)

T_y_test_pred = (
    T_w0 + T_X_test_array @ T_w_original[1:]
)

# 8. Display model parameters and equation
print("\nMultiple Linear Regression using Batch Gradient Descent")
print("========================================================")

print("Intercept (w0):", T_w0)
print("Coefficient for TV:", T_w1)
print("Coefficient for Radio:", T_w2)
print("Coefficient for Newspaper:", T_w3)

print("\nRegression Equation:")
print(
    f"Sales = {T_w0:.6f} "
    f"+ ({T_w1:.6f} * TV) "
    f"+ ({T_w2:.6f} * Radio) "
    f"+ ({T_w3:.6f} * Newspaper)"
)

print("\nEpochs completed:", T_epochs_completed)
print("Final training cost J:", T_current_cost)

# 9. Display predictions for each dataset

# Training Results
T_training_result = T_X_train.copy()
T_training_result["Actual Sales"] = T_y_train.to_numpy()
T_training_result["Predicted Sales"] = T_y_train_pred

print("\nTraining Results:")
print(T_training_result)

# Validation Results
T_validation_result = T_X_val.copy()
T_validation_result["Actual Sales"] = T_y_val.to_numpy()
T_validation_result["Predicted Sales"] = T_y_val_pred

print("\nValidation Results:")
print(T_validation_result)

# Testing Results
T_testing_result = T_X_test.copy()
T_testing_result["Actual Sales"] = T_y_test.to_numpy()
T_testing_result["Predicted Sales"] = T_y_test_pred

print("\nTesting Results:")
print(T_testing_result)