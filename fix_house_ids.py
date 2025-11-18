import pandas as pd

df = pd.read_csv("houses_dataset_final.csv")

df["house_id"] = df["house_id"] + 1   # shift IDs

df.to_csv("houses_dataset_final_fixed.csv", index=False)

print(df.head())
print("\nFixed CSV saved as houses_dataset_final_fixed.csv")
