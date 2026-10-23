import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Splitting_Dataset import T_X_test, T_y_test
from BGD import T_y_test_pred


# =========================================================
# Ex_2_2-e: Predicted-vs-Actual Plot and Residual Plot - BGD
# =========================================================
# 1. Calculate residuals on the Testing dataset
T_test_actual = T_y_test.to_numpy(dtype=float)
T_test_predicted = np.asarray(T_y_test_pred, dtype=float)
T_test_residuals = T_test_actual - T_test_predicted
# 2. Display the Testing residual table
T_residual_table = pd.DataFrame({
"Actual Sales": T_test_actual,
"Predicted Sales": T_test_predicted,
"Residual (Actual - Predicted)": T_test_residuals
}, index=T_y_test.index)
print("\nTesting Dataset: Actual, Predicted, and Residual Values")
print("-------------------------------------------------------")
print(T_residual_table)
# 3. Plot Actual Sales against Predicted Sales
# MLR has three predictors, so a single 2D regression line is not sufficient.
T_plot_min = min(np.min(T_test_actual), np.min(T_test_predicted))
T_plot_max = max(np.max(T_test_actual), np.max(T_test_predicted))
plt.figure(figsize=(8, 5))
plt.scatter(T_test_actual, T_test_predicted, label="Testing observations")
plt.plot([T_plot_min, T_plot_max], [T_plot_min, T_plot_max],
linestyle="--", label="Ideal prediction line (y = x)")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("MLR using BGD: Actual vs Predicted Sales (Testing Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# 4. Plot residuals against Predicted Sales
plt.figure(figsize=(8, 5))
plt.scatter(T_test_predicted, T_test_residuals, label="Testing residuals")
plt.axhline(y=0, linestyle="--", label="Zero-residual reference")
plt.xlabel("Predicted Sales")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("MLR using BGD: Residual Plot (Testing Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()