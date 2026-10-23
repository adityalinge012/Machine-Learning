#Name : Aditya Mohan Linge
#Roll : 3468
# Exp_2.1.a Loading and Reading the dataset - SLR(Simple Linear Regression)

import pandas as pd

# 1) We give Dataset path to python using 'r' keyword.
T_file_path = r"C:\Users\Hp\Desktop\ML\Ex_2.1_Simple Linear Regression_DataSet.xlsx"


# 2) Read the Excel dataset by using the function 'pd.read_excel()'
T_data = pd.read_excel(T_file_path)

# 3) Remove leading trailintg spaces from column names
T_data.columns = T_data.columns.astype(str).str.strip()

print("Original Dataset")
print("***********************")

# 4) Displaying the Original dataset after loading and reading the dataset of 'TV' & 'Sales'
# Here we get complete clean dataset.
print(T_data)

# 5) This shows how many no of data (Samples) are there present in dataset.
print("\nTotal number of samples: ")
print(len(T_data))

# 6) Displaying the columns names with there datatype using 'columns' keyword.
print("\nColumnss Names:")
print("********************")
print(T_data.columns)

# 7) Seprate input 'features' and 'target'.
T_X = T_data[["TV"]]
T_Y = T_data["Sales"]
print("**************Without DataFramez***************")
print(T_X)
print(T_Y)
# 8) Display Input X and Target Y .
print("\nInput X and Target Y :")
print("***************************")

# Using Dataframes we togethered both columns "tv" & "sales" in table.
T_X_Y = pd.DataFrame({"TV": T_X["TV"].values, "Sales": T_Y})
print(T_X_Y)

# 9) Display dimensions of X and Y By Using 'shape' keyword 
print("\nDataset Dimensions :")
print("***************************")
print("X shape :", T_X.shape)
print("Y shape :", T_Y.shape)

# 10) Final Conclusion
print("\n=========================")
print("Conclusion")
print("=========================")

print("The Dataset is successfully Loaded and Read.")

print("TV is selected as the independent variable (x) and Sales is selected as the dependent variable (Y).")
print("The dataset is ready for Simple Regeression")