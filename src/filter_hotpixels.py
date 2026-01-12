import os
import pandas as pd

# ------------------ PATHS ------------------
INPUT_DIR = "//wsl$/Ubuntu-20.04/home/arj/catkin_ws/data/chunks/New folder"
OUTPUT_DIR = r"C:\Arjun\Thesis\data\New folder\filtered chunks\New folder"
HOT_PIXEL_FILE = r"C:\Arjun\Thesis\data\New folder\dvs_vpr_2020-04-21-17-03-03_hot_pixels.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD HOT PIXELS ------------------
hot_pixels = set()

with open(HOT_PIXEL_FILE, "r") as f:
    for line in f:
        x, y = line.strip().split(",")
        hot_pixels.add((int(x), int(y)))

print(f"Loaded {len(hot_pixels)} hot pixels")
2
# ------------------ PROCESS FILES ------------------
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Read CSV
    df = pd.read_csv(input_path)

    # Filter out hot pixels
    mask = ~df.apply(lambda r: (r["x"], r["y"]) in hot_pixels, axis=1)
    df_filtered = df[mask]

    # Save filtered CSV
    df_filtered.to_csv(output_path, index=False)

    print(
        f"{filename}: "
        f"removed {len(df) - len(df_filtered)} events, "
        f"remaining {len(df_filtered)}"
        
    )

print("✅ All files processed successfully.")
