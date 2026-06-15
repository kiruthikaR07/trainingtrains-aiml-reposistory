import pandas as pd

# Read CSV file
df = pd.read_csv("students.csv")

# Display complete dataset
print("===== Complete Dataset =====")
print(df)

# First 5 rows
print("\n===== First 5 Rows =====")
print(df.head())

# Last 5 rows
print("\n===== Last 5 Rows =====")
print(df.tail())

# Dataset information
print("\n===== Dataset Information =====")
print(df.info())

# Statistical summary
print("\n===== Statistical Summary =====")
print(df.describe())

# Column names
print("\n===== Column Names =====")
print(df.columns)

# Shape of dataset
print("\n===== Dataset Shape =====")
print(df.shape)

# Select a column
print("\n===== Maths Marks =====")
print(df["Maths"])

# Average marks
print("\n===== Average Marks =====")
print("Maths Average:", df["Maths"].mean())
print("Science Average:", df["Science"].mean())
print("English Average:", df["English"].mean())

# Maximum marks
print("\n===== Highest Marks =====")
print("Maths:", df["Maths"].max())
print("Science:", df["Science"].max())
print("English:", df["English"].max())

# Minimum marks
print("\n===== Lowest Marks =====")
print("Maths:", df["Maths"].min())
print("Science:", df["Science"].min())
print("English:", df["English"].min())

# Add a new column
df["Total"] = df["Maths"] + df["Science"] + df["English"]

print("\n===== Dataset with Total Marks =====")
print(df)

# Student with highest total
top_student = df.loc[df["Total"].idxmax()]

print("\n===== Top Student =====")
print(top_student)

# Save modified dataset
df.to_csv("students_updated.csv", index=False)

print("\nUpdated dataset saved successfully!")