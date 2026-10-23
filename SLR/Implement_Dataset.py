import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from Splitting_Dataset import (
    T_X_Train,
    T_X_Val,
    T_X_Test,
    T_Y_Train,
    T_Y_Val,
    T_Y_Test
)

# We Implement SLR Using Scikit-learn

# 1) Create Simple Linear Regression model
# We use Linear Regression algorithm to predict target.

Model = LinearRegression()

# 2) We Train Model Using Trained Data 

Model.fit(T_X_Train, T_Y_Train)

# 3) Obtain slope and intercept
# This is algorithm concept called weights.

w0 = Model.intercept_
w1 = Model.coef_[0]


print("\n***************************")
print("Simple Linear Regression - Scikit Learn")
print("\n***************************")
print("\nfinal Model Parameters")
print("\n***************************")
print("Intercept (w0) :", w0)
print("Slope (w1) :", w1)


#  5) Display regression equation 

print("\nRegression Equation :")
print(f"Sales = {w0:.6f} + ({w1:.6f} * TV)")

# 6) prediction on Training data 

Y_Train_Predict = Model.predict(T_X_Train)

# 7) Prediction on Validation data

Y_Val_Predict = Model.predict(T_X_Val)

# 8) Prediction on Testing data

Y_Test_Predict = Model.predict(T_X_Test)

# 9) Display Training results

Training_Results = pd.DataFrame({
    "TV" : T_X_Train["TV"].values,
    "Actual Sales": T_Y_Train.values,
    "Predicted Sales": Y_Train_Predict
})
print("\nTraining Results")
print("***********************")
print(Training_Results)

# 10) Display Validation Results

Validation_Results = pd.DataFrame({
    "TV" : T_X_Val["TV"].values,
    "Actual Sales": T_Y_Val.values,
    "Predicted Sales": Y_Val_Predict
})
print("\nValidation Results")
print("***********************")
print(Validation_Results)

# 11) Display Testing Results

Testing_Results = pd.DataFrame({
    "TV" : T_X_Test["TV"].values,
    "Actual Sales": T_Y_Test.values,
    "Predicted Sales": Y_Test_Predict
})
print("\nTesting Results")
print("***********************")
print(Testing_Results)


# 12. Create values for Regression Line
T_X_line = pd.DataFrame({
    "TV": np.linspace(
        T_X_Train["TV"].min(),
        T_X_Train["TV"].max(),
        100
    )
})

# Predict Sales for Regression Line
T_y_line = Model.predict(T_X_line)

# 13. Plot Regression Line
plt.figure(figsize=(8, 5))

plt.scatter(
    T_X_Train["TV"],
    T_Y_Train,
    label="Training Data"
)

plt.plot(
    T_X_line["TV"],
    T_y_line,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("TV Advertising Expenditure")
plt.ylabel("Sales")
plt.title("Simple Linear Regression using Scikit-learn")
plt.legend()
plt.grid(True)
plt.show()