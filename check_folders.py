import os

folders = sorted([int(f) for f in os.listdir("images_final") if f.isdigit()])
print("Min folder:", min(folders), "Max folder:", max(folders))
print("Total folders:", len(folders))
print("First 20 folders:", folders[:20])

print("\nChecking outdoor.jpg availability:")
missing = []
for folder in folders:
    if not os.path.exists(f"images_final/{folder}/outdoor.jpg"):
        missing.append(folder)

print("Missing outdoor count:", len(missing))
print("Missing outdoor examples:", missing[:20])

