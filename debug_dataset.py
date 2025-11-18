# debug_dataset.py
import os, glob, pandas as pd

CSV_PATH = "houses_dataset_final_fixed.csv"
IMAGES_ROOT = "images_final"

print("Using CSV_PATH:", CSV_PATH)
print("Using IMAGES_ROOT:", IMAGES_ROOT)
print()

# CSV head
if not os.path.exists(CSV_PATH):
    print("ERROR: CSV not found at", CSV_PATH); raise SystemExit()
df = pd.read_csv(CSV_PATH)
print("CSV rows:", len(df))
print("CSV house_id sample:", df["house_id"].head(15).tolist())
print()

# folders
if not os.path.exists(IMAGES_ROOT):
    print("ERROR: images root not found:", IMAGES_ROOT); raise SystemExit()
folders = sorted([d for d in os.listdir(IMAGES_ROOT) if os.path.isdir(os.path.join(IMAGES_ROOT, d))])
print("Total folders found:", len(folders))
print("First 20 folders:", folders[:20])
print()

# Check existence of outdoor.jpg for first 20 folders
missing = []
for f in folders[:50]:
    p = os.path.join(IMAGES_ROOT, f, "outdoor.jpg")
    if not os.path.exists(p):
        missing.append((f, p))
print("Missing outdoor count (first 50 checked):", len(missing))
if missing:
    print("Missing examples:", missing[:10])
else:
    print("All first-50 folders have outdoor.jpg")
print()

# Show mapping example: folder -> csv row by house_id match
sample = folders[:20]
matches = []
for f in sample:
    try:
        hid = int(f)
        row = df[df["house_id"] == hid]
        matches.append((f, not row.empty))
    except:
        matches.append((f, False))
print("Sample folder->csv match (folder, has_csv_row):")
for m in matches:
    print(" ", m)
