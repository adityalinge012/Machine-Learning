import pandas as pd

# =========================================================
# Ex_2_2-a: Loading and Reading the Advertising Dataset
# =========================================================

# 1. Specify the Excel file path
T_file_path = r"C:\Users\Hp\Desktop\ML\MLR\MLR_Dataset.xlsx"

# 2. Read the Excel dataset
T_data = pd.read_excel(T_file_path)
T_data.columns = T_data.columns.astype(str).str.strip()

# 3. Display the original dataset
print("Original Advertising Dataset:")
print("--------------------------------")
print(T_data)

# 4. Display the number of observations and column names
print("\nTotal number of samples:")
print(len(T_data))

print("\nColumn Names:")
print("--------------------------------")
print(T_data.columns)

# 5. Separate input features and target
T_X = T_data[["TV", "Radio", "Newspaper"]]
T_y = T_data["Sales"]

# 6. Display input features and target
print("\nInput Features and Target:")
print("--------------------------------")
T_input_target_data = T_X.copy()
T_input_target_data["Sales"] = T_y
print(T_input_target_data)

# 7. Display dimensions
print("\nDataset Dimensions:")
print("--------------------------------")
print("X shape :", T_X.shape)
print("y shape :", T_y.shape)

print("\nCONCLUSION")
print("The dataset is ready for Multiple Linear Regression.")