import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from Splitting_Dataset import (
    T_X_Train,
    T_X_Val,
    T_X_Test,
    T_Y_Train,
    T_Y_Val,
    T_Y_Test
)


# =========================================================
# Ex_2_1-d
# To Implement Simple Linear Regression using
# Batch Gradient Descent
# =========================================================


# ---------------------------------------------------------
# 1. Convert training data into NumPy arrays
# ---------------------------------------------------------

T_X_train_array = T_X_Train["TV"].to_numpy()

T_y_train_array = T_Y_Train.to_numpy()

T_n = len(T_y_train_array)


# ---------------------------------------------------------
# 2. Initialize weights
# ---------------------------------------------------------

T_w0 = 0.0

T_w1 = 0.0


# ---------------------------------------------------------
# 3. Set learning rate
# ---------------------------------------------------------

T_learning_rate = 1e-5


# ---------------------------------------------------------
# 4. Set convergence criteria
# ---------------------------------------------------------

T_tolerance = 1e-12

T_gradient_tolerance = 1e-10


# ---------------------------------------------------------
# 5. Initialize convergence variables
# ---------------------------------------------------------

T_epoch = 0

T_previous_cost = float("inf")

T_cost_history = []

T_max_epochs = 1000000


# =========================================================
# 6. Batch Gradient Descent
# =========================================================

while T_epoch < T_max_epochs:

    T_epoch = T_epoch + 1


    # -----------------------------------------------------
    # Calculate predicted Sales (ŷ)
    # -----------------------------------------------------

    T_y_hat = (
        T_w0 +
        T_w1 * T_X_train_array
    )


    # -----------------------------------------------------
    # Calculate error
    # -----------------------------------------------------

    T_error = (
        T_y_hat -
        T_y_train_array
    )


    # -----------------------------------------------------
    # Calculate gradients using ALL training observations
    # -----------------------------------------------------

    T_dw0 = (
        1 / T_n
    ) * np.sum(T_error)


    T_dw1 = (
        1 / T_n
    ) * np.sum(
        T_X_train_array * T_error
    )


    # -----------------------------------------------------
    # Update weights
    # -----------------------------------------------------

    T_w0 = (
        T_w0 -
        T_learning_rate * T_dw0
    )


    T_w1 = (
        T_w1 -
        T_learning_rate * T_dw1
    )


    # -----------------------------------------------------
    # Calculate new predictions after weight update
    # -----------------------------------------------------

    T_y_hat = (
        T_w0 +
        T_w1 * T_X_train_array
    )


    # -----------------------------------------------------
    # Calculate new errors
    # -----------------------------------------------------

    T_error = (
        T_y_hat -
        T_y_train_array
    )


    # -----------------------------------------------------
    # Calculate cost function
    # -----------------------------------------------------

    T_cost = (
        1 / (2 * T_n)
    ) * np.sum(
        T_error ** 2
    )


    T_cost_history.append(
        T_cost
    )


    # -----------------------------------------------------
    # Calculate change in cost
    # -----------------------------------------------------

    if T_previous_cost == float("inf"):

        T_cost_change = float("inf")

    else:

        T_cost_change = abs(
            T_previous_cost -
            T_cost
        )


    # -----------------------------------------------------
    # Calculate gradient magnitude
    # -----------------------------------------------------

    T_gradient_magnitude = np.sqrt(
        T_dw0 ** 2 +
        T_dw1 ** 2
    )


    # -----------------------------------------------------
    # Check convergence
    # -----------------------------------------------------

    if (
        T_cost_change < T_tolerance
        and
        T_gradient_magnitude < T_gradient_tolerance
    ):

        break


    T_previous_cost = T_cost


# =========================================================
# 7. Display BGD results
# =========================================================

print("\n==============================================")
print("Simple Linear Regression - BGD")
print("==============================================")


print(
    "Number of training samples :",
    T_n
)

print(
    "Number of epochs           :",
    T_epoch
)

print(
    "Final Cost J               :",
    T_cost
)

print(
    "Cost change                :",
    T_cost_change
)

print(
    "Gradient magnitude         :",
    T_gradient_magnitude
)


# =========================================================
# 8. Display final model parameters
# =========================================================

print("\nFinal Model Parameters:")
print("--------------------------------")

print("w0 =", T_w0)

print("w1 =", T_w1)


# =========================================================
# 9. Display regression equation
# =========================================================

print("\nRegression Equation:")

print(
    f"Sales = {T_w0:.6f} + "
    f"({T_w1:.6f}) * TV"
)


# =========================================================
# 10. Prediction on Training data
# =========================================================

T_y_train_pred = (
    T_w0 +
    T_w1 *
    T_X_Train["TV"].to_numpy()
)


# =========================================================
# 11. Prediction on Validation data
# =========================================================

T_y_val_pred = (
    T_w0 +
    T_w1 *
    T_X_Val["TV"].to_numpy()
)


# =========================================================
# 12. Prediction on Testing data
# =========================================================

T_y_test_pred = (
    T_w0 +
    T_w1 *
    T_X_Test["TV"].to_numpy()
)


# =========================================================
# 13. Display Training Results
# =========================================================

T_training_result = pd.DataFrame({

    "TV":
        T_X_Train["TV"].values,

    "Actual Sales":
        T_Y_Train.values,

    "Predicted Sales":
        T_y_train_pred
})


print("\nTraining Results:")
print("--------------------------------")

print(T_training_result)


# =========================================================
# 14. Display Validation Results
# =========================================================

T_validation_result = pd.DataFrame({

    "TV":
        T_X_Val["TV"].values,

    "Actual Sales":
        T_Y_Val.values,

    "Predicted Sales":
        T_y_val_pred
})


print("\nValidation Results:")
print("--------------------------------")

print(T_validation_result)


# =========================================================
# 15. Display Testing Results
# =========================================================

T_testing_result = pd.DataFrame({

    "TV":
        T_X_Test["TV"].values,

    "Actual Sales":
        T_Y_Test.values,

    "Predicted Sales":
        T_y_test_pred
})


print("\nTesting Results:")
print("--------------------------------")

print(T_testing_result)


# =========================================================
# 16. Plot Regression Line
# =========================================================

T_X_line = np.linspace(

    T_X_Train["TV"].min(),

    T_X_Train["TV"].max(),

    100
)


T_y_line = (
    T_w0 +
    T_w1 * T_X_line
)


plt.figure(figsize=(8, 5))


plt.scatter(

    T_X_Train["TV"],

    T_Y_Train,

    label="Training Data"
)


plt.plot(

    T_X_line,

    T_y_line,

    linewidth=2,

    label="BGD Regression Line"
)


plt.xlabel("TV Advertising Expenditure")

plt.ylabel("Sales")


plt.title(
    "Simple Linear Regression using Batch Gradient Descent"
)


plt.legend()

plt.grid(True)

plt.show()


# =========================================================
# 17. Plot Cost versus Epoch
# =========================================================

plt.figure(figsize=(8, 5))


plt.plot(

    range(
        1,
        len(T_cost_history) + 1
    ),

    T_cost_history
)


plt.xlabel("Epoch")

plt.ylabel("Cost J")


plt.title(
    "Cost Function versus Epoch - SLR-BGD"
)


plt.grid(True)

plt.show()