import os, glob, shutil
from pathlib import Path
import pandas as pd

NEW_DATASET = "new_dataset"
OUTPUT = "images_final"
K_INDOOR = 4

os.makedirs(OUTPUT, exist_ok=True)

def get_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return files

files = get_images(NEW_DATASET)
print("Total images found:", len(files))

# Grouping logic
groups = {}
for p in files:
    parent = Path(p).parent.name
    stem = Path(p).stem

    # If parent folder is numeric use as house id
    if parent.isdigit():
        hid = parent

    # If filename starts with number (e.g. 123_outdoor)
    elif "_" in stem and stem.split("_")[0].isdigit():
        hid = stem.split("_")[0]
    else:
        # fallback: just use parent folder
        hid = parent

    groups.setdefault(hid, []).append(p)

print("Total house groups:", len(groups))

def choose_outdoor(files):
    names = [f.lower() for f in files]
    for f in files:
        fn = f.lower()
        if any(k in fn for k in ["outdoor", "front", "exterior", "facade"]):
            return f
    return files[0]

# Create images_final/<house_id>/...
for hid, lst in groups.items():
    folder = os.path.join(OUTPUT, hid)
    os.makedirs(folder, exist_ok=True)

    outdoor = choose_outdoor(lst)
    shutil.copy2(outdoor, os.path.join(folder, "outdoor.jpg"))

    indoors = [f for f in lst if f != outdoor][:K_INDOOR]
    if not indoors:
        indoors = [outdoor]

    for i, src in enumerate(indoors):
        shutil.copy2(src, os.path.join(folder, f"indoor_{i+1}.jpg"))

print("DONE: images_final created.")

# Create CSV with house IDs
df = pd.DataFrame({"house_id": list(groups.keys())})
df.to_csv("socal2_new.csv", index=False)
print("CSV saved as socal2_new.csv")
