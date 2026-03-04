# hot_pixels.py
def load_hot_pixels(path: str) -> set[tuple[int, int]]:
    hot_pixels = set()
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                x, y = line.strip().split(",")
                hot_pixels.add((int(x), int(y)))
    return hot_pixels
