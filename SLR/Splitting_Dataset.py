import pandas as pd
import numpy as np

from Loading_Reading_SLR_Dataset import T_X, T_Y


# =========================================================
# 1) Split dataset into Training & Temporary datasets
# =========================================================

# Training = 80%
# Temporary = 20%
#
# Temporary contains:
# Validation = 10%
# Testing = 10%


# ---------------------------------------------------------
# Set random state
# ---------------------------------------------------------

np.random.seed(42)


# ---------------------------------------------------------
# Shuffle the dataset
# ---------------------------------------------------------

T_indices = np.random.permutation(len(T_X))


# ---------------------------------------------------------
# Calculate Training size
# ---------------------------------------------------------

T_train_size = int(0.80 * len(T_X))


# ---------------------------------------------------------
# Get Training and Temporary indices
# ---------------------------------------------------------

T_train_indices = T_indices[:T_train_size]

T_temp_indices = T_indices[T_train_size:]


# ---------------------------------------------------------
# Create Training Dataset
# ---------------------------------------------------------

T_X_Train = T_X.iloc[T_train_indices].copy()

T_Y_Train = T_Y.iloc[T_train_indices].copy()


# ---------------------------------------------------------
# Create Temporary Dataset
# ---------------------------------------------------------

T_X_Temp = T_X.iloc[T_temp_indices].copy()

T_Y_Temp = T_Y.iloc[T_temp_indices].copy()


# =========================================================
# 2) Split Temporary dataset into Validation & Testing
# =========================================================

# Temporary = 20%
#
# Validation = 50% of Temporary = 10% total
# Testing = 50% of Temporary = 10% total


# ---------------------------------------------------------
# Shuffle Temporary dataset
# ---------------------------------------------------------

T_temp_indices_2 = np.random.permutation(len(T_X_Temp))


# ---------------------------------------------------------
# Calculate Validation size
# ---------------------------------------------------------

T_val_size = int(0.50 * len(T_X_Temp))


# ---------------------------------------------------------
# Get Validation and Testing indices
# ---------------------------------------------------------

T_val_indices = T_temp_indices_2[:T_val_size]

T_test_indices = T_temp_indices_2[T_val_size:]


# ---------------------------------------------------------
# Create Validation Dataset
# ---------------------------------------------------------

T_X_Val = T_X_Temp.iloc[T_val_indices].copy()

T_Y_Val = T_Y_Temp.iloc[T_val_indices].copy()


# ---------------------------------------------------------
# Create Testing Dataset
# ---------------------------------------------------------

T_X_Test = T_X_Temp.iloc[T_test_indices].copy()

T_Y_Test = T_Y_Temp.iloc[T_test_indices].copy()


# =========================================================
# 3) Displaying Training Dataset
# =========================================================

print("\nTraining Dataset:")
print("**********************")

TrainingDataset = pd.concat(
    [
        T_X_Train,
        T_Y_Train.rename("Sales")
    ],
    axis=1
)

print(TrainingDataset)


# =========================================================
# 4) Displaying Validation Dataset
# =========================================================

print("\nValidation Dataset:")
print("************************")

ValidationDataset = pd.concat(
    [
        T_X_Val,
        T_Y_Val.rename("Sales")
    ],
    axis=1
)

print(ValidationDataset)


# =========================================================
# 5) Displaying Testing Dataset
# =========================================================

print("\nTesting Dataset:")
print("************************")

TestingDataset = pd.concat(
    [
        T_X_Test,
        T_Y_Test.rename("Sales")
    ],
    axis=1
)

print(TestingDataset)


# =========================================================
# 6) Display Dimensions
# =========================================================

print("\nDataset Dimensions:")
print("*****************************")

print("Training X shape    :", T_X_Train.shape)

print("Training Y shape    :", T_Y_Train.shape)

print("Validation X shape  :", T_X_Val.shape)

print("Validation Y shape  :", T_Y_Val.shape)

print("Testing X shape     :", T_X_Test.shape)

print("Testing Y shape     :", T_Y_Test.shape)