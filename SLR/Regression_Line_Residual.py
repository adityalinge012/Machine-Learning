import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Implement_Dataset import (
        T_X_Train,
        T_X_Test,
        T_Y_Train,
        T_Y_Test,
        w0,
        w1,
        Y_Test_Predict
        )
# =========================================================
# Ex_2_1-h
# Regression Line and Residual Plot - Batch Gradient Descent
# =========================================================
# 1. Display the BGD regression equation
print("\n==============================================")
print("Regression Line and Residual Plot - BGD")
print("==============================================")
print("\nRegression Equation:")
print(f"Sales = {w0:.6f} + ({w1:.6f}) * TV")
# 2. Plot the Training observations and fitted regression line
T_X_line = pd.DataFrame({
"TV": np.linspace(T_X_Train["TV"].min(), T_X_Train["TV"].max(), 100)
})
T_y_line = w0 + w1 * T_X_line["TV"].to_numpy(dtype=float)
plt.figure(figsize=(8, 5))
plt.scatter(T_X_Train["TV"], T_Y_Train, label="Training Data")
plt.plot(T_X_line["TV"], T_y_line, linewidth=2, label="BGD Regression Line")
plt.xlabel("TV Advertising Expenditure")
plt.ylabel("Sales")
plt.title("Regression Line using Batch Gradient Descent")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 3. Calculate residuals for the Testing dataset
T_residuals_test = T_Y_Test.to_numpy(dtype=float) - np.asarray(Y_Test_Predict, dtype=float)
# 4. Display the Testing residual table
T_residual_table = pd.DataFrame({
"TV": T_X_Test["TV"].to_numpy(),
"Actual Sales": T_Y_Test.to_numpy(),
"Predicted Sales": np.asarray(Y_Test_Predict, dtype=float),
"Residual": T_residuals_test
})
print("\nTesting Residuals:")
print("-------------------------------------------------------------")
print(T_residual_table.to_string(index=False))
# 5. Plot Testing residuals against TV advertising expenditure
plt.figure(figsize=(8, 5))
plt.scatter(T_X_Test["TV"], T_residuals_test, label="Testing Residuals")
plt.axhline(y=0, linestyle="--", label="Zero Residual Line")
plt.xlabel("TV Advertising Expenditure")
plt.ylabel("Residual (Actual Sales - Predicted Sales)")
plt.title("Residual Plot - BGD (Testing Dataset)")



plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()