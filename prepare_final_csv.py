import pandas as pd

# Load the HousesInfo.txt file
df = pd.read_csv("new_dataset/HousesInfo.txt", sep=" ", header=None)

# Assign column names
df.columns = ["bed", "bath", "sqft", "zipcode", "price"]

# Add house_id column to match folder names
df["house_id"] = df.index

# Reorder columns
df = df[["house_id", "bed", "bath", "sqft", "zipcode", "price"]]

# Save final CSV
df.to_csv("houses_dataset_final.csv", index=False)

print("CSV created successfully: houses_dataset_final.csv")
print(df.head())
