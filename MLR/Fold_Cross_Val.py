import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from Splitting_Dataset import T_X_train, T_y_train

# =========================================================
# Ex_2_2-f: 5-Fold Cross-Validation - Batch Gradient Descent
# =========================================================

# 1. Define BGD training and prediction for one fold
def T_fit_bgd_for_fold(
    T_X_fold_train,
    T_y_fold_train,
    T_X_fold_valid
):

    T_X_train_array = T_X_fold_train.to_numpy(dtype=float)
    T_X_valid_array = T_X_fold_valid.to_numpy(dtype=float)
    T_y_train_array = T_y_fold_train.to_numpy(dtype=float)

    # Standardize using the current fold-training data only
    T_X_mean = np.mean(T_X_train_array, axis=0)
    T_X_std = np.std(T_X_train_array, axis=0)

    T_X_std[T_X_std == 0] = 1.0

    T_X_train_scaled = (
        T_X_train_array - T_X_mean
    ) / T_X_std

    T_X_valid_scaled = (
        T_X_valid_array - T_X_mean
    ) / T_X_std

    # Add a column of ones for the intercept
    T_n_train = len(T_y_train_array)

    T_X_train_design = np.column_stack(
        (np.ones(T_n_train), T_X_train_scaled)
    )

    T_X_valid_design = np.column_stack(
        (np.ones(len(T_X_valid_scaled)), T_X_valid_scaled)
    )

    # Initialize weights and BGD parameters
    T_w_scaled = np.zeros(T_X_train_design.shape[1])

    T_learning_rate = 0.01
    T_max_epochs = 100000
    T_tolerance = 1e-12
    T_gradient_tolerance = 1e-8

    T_previous_cost = 0.5 * np.mean(
        (
            T_X_train_design @ T_w_scaled
            - T_y_train_array
        ) ** 2
    )

    T_epochs_completed = 0

    # Train a fresh model for this fold
    for T_epoch in range(1, T_max_epochs + 1):

        T_y_train_pred_scaled = (
            T_X_train_design @ T_w_scaled
        )

        T_errors = (
            T_y_train_pred_scaled - T_y_train_array
        )

        T_gradient = (
            T_X_train_design.T @ T_errors
        ) / T_n_train

        T_w_scaled = (
            T_w_scaled - T_learning_rate * T_gradient
        )

        T_y_train_pred_scaled = (
            T_X_train_design @ T_w_scaled
        )

        T_current_cost = 0.5 * np.mean(
            (
                T_y_train_pred_scaled
                - T_y_train_array
            ) ** 2
        )

        T_gradient_current = (
            T_X_train_design.T
            @ (T_y_train_pred_scaled - T_y_train_array)
        ) / T_n_train

        T_epochs_completed = T_epoch

        if (
            abs(T_previous_cost - T_current_cost)
            < T_tolerance
            and np.max(np.abs(T_gradient_current))
            < T_gradient_tolerance
        ):
            break

        T_previous_cost = T_current_cost

    # Predict the validation fold
    T_y_valid_pred_scaled = (
        T_X_valid_design @ T_w_scaled
    )

    return (
        T_y_valid_pred_scaled,
        T_epochs_completed,
        T_current_cost
    )


# 2. Configure 5-fold cross-validation on the Training dataset only
T_kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

T_fold_results = []


# 3. Train and evaluate one fresh BGD model in each fold
for T_fold_number, (
    T_train_indices,
    T_valid_indices
) in enumerate(
    T_kfold.split(T_X_train),
    start=1
):

    T_X_fold_train = T_X_train.iloc[T_train_indices]
    T_X_fold_valid = T_X_train.iloc[T_valid_indices]

    T_y_fold_train = T_y_train.iloc[T_train_indices]
    T_y_fold_valid = T_y_train.iloc[T_valid_indices]

    (
        T_y_fold_pred,
        T_epochs,
        T_final_cost
    ) = T_fit_bgd_for_fold(
        T_X_fold_train,
        T_y_fold_train,
        T_X_fold_valid
    )

    T_y_fold_actual = T_y_fold_valid.to_numpy(dtype=float)

    T_fold_mse = mean_squared_error(
        T_y_fold_actual,
        T_y_fold_pred
    )

    T_fold_results.append({
        "Fold": T_fold_number,
        "MAE": mean_absolute_error(
            T_y_fold_actual,
            T_y_fold_pred
        ),
        "MSE": T_fold_mse,
        "RMSE": np.sqrt(T_fold_mse),
        "R2": r2_score(
            T_y_fold_actual,
            T_y_fold_pred
        ),
        "Epochs": T_epochs,
        "Final Training Cost": T_final_cost
    })


# 4. Display fold-wise results and cross-validation summary
T_fold_results_table = (
    pd.DataFrame(T_fold_results)
    .set_index("Fold")
)

print("\n5-Fold Cross-Validation Results - MLR using BGD")
print("================================================")

print(
    T_fold_results_table.to_string(
        float_format=lambda T_value: f"{T_value:.6f}"
    )
)

print("\nCross-Validation Summary (mean and standard deviation)")
print("-------------------------------------------------------")

for T_metric_name in ["MAE", "MSE", "RMSE", "R2"]:

    T_metric_mean = (
        T_fold_results_table[T_metric_name].mean()
    )

    T_metric_std = (
        T_fold_results_table[T_metric_name].std(ddof=0)
    )

    print(
        f"{T_metric_name:5s} "
        f"Mean = {T_metric_mean:.6f}, "
        f"Std = {T_metric_std:.6f}"
    )


# 5. Conclusion
print(
    "\nCross-validation uses only the Training subset "
    "from Ex_2_2-b."
)

print(
    "The separate Validation and Testing subsets "
    "remain untouched."
)