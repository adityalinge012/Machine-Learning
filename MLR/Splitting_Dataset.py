import pandas as pd
from sklearn.model_selection import train_test_split
from Load_Read_Data import T_X, T_y
# =========================================================
# Ex_2_2-b: Split Dataset into Training, Validation, Testing
# =========================================================
# 1. Split into Training and Temporary datasets
# 80% Training; 20% Temporary
T_X_train, T_X_temp, T_y_train, T_y_temp = train_test_split(T_X, T_y, test_size=0.20, random_state=42)
# 2. Split the Temporary dataset equally into Validation and Testing


T_X_val, T_X_test, T_y_val, T_y_test = train_test_split(T_X_temp, T_y_temp, test_size=0.50, random_state=42)
# 3. Display the number of observations
print("Dataset Split:")
print("--------------------------------")
print("Training samples :", len(T_X_train))
print("Validation samples :", len(T_X_val))
print("Testing samples :", len(T_X_test))
print("Total samples :", len(T_X_train) + len(T_X_val) + len(T_X_test))
# 4. Display the Training, Validation, and Testing datasets
T_training_data = T_X_train.copy()
T_training_data["Sales"] = T_y_train
T_validation_data = T_X_val.copy()
T_validation_data["Sales"] = T_y_val
T_testing_data = T_X_test.copy()
T_testing_data["Sales"] = T_y_test
print("\nTraining Dataset:")
print(T_training_data)
print("\nValidation Dataset:")
print(T_validation_data)
print("\nTesting Dataset:")
print(T_testing_data)
# 5. Display dimensions
print("\nDataset Dimensions:")
print("--------------------------------")
print("Training X shape :", T_X_train.shape)
print("Training y shape :", T_y_train.shape)
print("Validation X shape :", T_X_val.shape)
print("Validation y shape :", T_y_val.shape)
print("Testing X shape :", T_X_test.shape)
print("Testing y shape :", T_y_test.shape)