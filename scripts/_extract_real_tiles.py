"""One-off: extract real LiDAR map tiles to /tmp for the hero GIF (cached)."""
import sys, time
import numpy as np, laspy

DIR = "/media/sasaki/aiueo/rosbag/sample_lx/mme_comparison"
HALF = 35.0
C = np.load("/tmp/tile_centers.npy")
buckets = [[] for _ in C]
t0 = time.time()
with laspy.open(f"{DIR}/old_map.las") as f:
    for ci, chunk in enumerate(f.chunk_iterator(5_000_000)):
        x = np.asarray(chunk.x); y = np.asarray(chunk.y); z = np.asarray(chunk.z)
        for k, (cx, cy) in enumerate(C):
            m = (x > cx - HALF) & (x < cx + HALF) & (y > cy - HALF) & (y < cy + HALF)
            if m.any():
                buckets[k].append(np.stack([x[m], y[m], z[m]], 1))
        print(f"chunk {ci} done {time.time()-t0:.0f}s", flush=True)

rng = np.random.default_rng(0)
tiles = []
for k, b in enumerate(buckets):
    P = np.vstack(b)
    g = np.percentile(P[:, 2], 3)
    P = P[(P[:, 2] > g + 0.6) & (P[:, 2] < g + 20)]
    xy = P[:, :2] - C[k]
    if len(xy) > 30000:
        xy = xy[rng.choice(len(xy), 30000, replace=False)]
    tiles.append(xy.astype(np.float32))
    print(f"tile {k}: {len(xy)} pts", flush=True)

np.savez("/tmp/real_tiles.npz", **{f"t{k}": t for k, t in enumerate(tiles)})
print("SAVED /tmp/real_tiles.npz", flush=True)
