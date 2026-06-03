"""Extract ONLY seq00 velodyne (.bin) from the public KITTI odometry archive.

The velodyne archive is one 84 GB zip; we never download it whole. remotezip
fetches the central directory (tail range request) then one HTTP range request
per member, so we pull only seq00 (~9 GB, 4541 scans).
"""
import sys
from pathlib import Path
from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
DEST = Path("$HOME/datasets/kitti_seq00_velodyne")
DEST.mkdir(parents=True, exist_ok=True)

with RemoteZip(URL) as z:
    bins = sorted(n for n in z.namelist() if "/sequences/00/velodyne/" in n and n.endswith(".bin"))
    print(f"seq00 scans: {len(bins)}", flush=True)
    for i, name in enumerate(bins):
        out = DEST / Path(name).name
        if out.exists() and out.stat().st_size > 0:
            continue
        with z.open(name) as src:
            out.write_bytes(src.read())
        if i % 100 == 0 or i == len(bins) - 1:
            print(f"  {i + 1}/{len(bins)}", flush=True)
print("DONE", flush=True)
