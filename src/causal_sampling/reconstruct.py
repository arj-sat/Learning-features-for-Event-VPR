import os
import pandas as pd
import numpy as np
import cv2
import datetime
from zoneinfo import ZoneInfo

EVENT_FOLDER = r"C:\Arjun\Thesis\data\20200421_170039-sunset1\filtered chunks\subsampled"
OUTPUT_FOLDER = r"C:\Arjun\Thesis\data\20200421_170039-sunset1\images"
IMG_WIDTH = 346
IMG_HEIGHT = 260


IMG_W = 346
IMG_H = 260

# =============================
# Event → image accumulation
# =============================
def accumulate_events(events):
    """
    events: Nx4 array [x, y, polarity, timestamp]
    """
    img = np.zeros((IMG_H, IMG_W), dtype=np.float32)

    for x, y, p, _ in events:
        x = int(x)
        y = int(y)

        # Handle DAVIS polarity formats
        if p == 0:
            p = -1

        img[y, x] += p

    return img

# =============================
# Normalize for visualization
# =============================
def normalize(img):
    max_val = np.max(np.abs(img))
    if max_val == 0:
        return np.zeros_like(img, dtype=np.uint8)

    img = img / max_val
    img = (img + 1.0) * 0.5
    return (img * 255).astype(np.uint8)

# =============================
# ROS timestamp → human readable
# =============================
brisbane_tz = ZoneInfo("Australia/Brisbane")
def ros_time_to_str(ts):
    """
    ts: float seconds since epoch (ROS bag time)
    """
    return (
        datetime
        .fromtimestamp(ts, tz=ZoneInfo("UTC"))
        .astimezone(brisbane_tz)
        .strftime("%Y-%m-%d %H:%M:%S.%f")
    )

# =============================
# Main processing
# =============================
csv_files = sorted(f for f in os.listdir(EVENT_DIR) if f.endswith(".csv"))

for file_idx, fname in enumerate(csv_files):

    path = os.path.join(EVENT_DIR, fname)
    df = pd.read_csv(path)

    # Sort by timestamp
    #df = df.sort_values("timestamp")
    events = df.values

    ts_min = events[0, 3]
    ts_max = events[-1, 3]
    ts_mid = 0.5 * (ts_min + ts_max)

    # Split events
    events_first = events[events[:, 3] <= ts_mid]
    events_second = events[events[:, 3] > ts_mid]

    # Accumulate
    img_first = normalize(accumulate_events(events_first))
    img_second = normalize(accumulate_events(events_second))

    # Save images
    base = os.path.splitext(fname)[0]

    out1 = f"{base}_part1.png"
    out2 = f"{base}_part2.png"

    cv2.imwrite(os.path.join(OUTPUT_DIR, out1), img_first)
    cv2.imwrite(os.path.join(OUTPUT_DIR, out2), img_second)

    # Print human-readable timestamps
    print(
        f"{fname} | "
        f"Frame1 start: {ros_time_to_str(ts_min)} | "
        f"Frame2 start: {ros_time_to_str(ts_mid)}"
    )
    break

print("Done. Generated 2 frames per CSV.")