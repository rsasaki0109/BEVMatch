"""Extract seq 05/06/07/08 velodyne scans from per-sequence range-downloaded blocks.

Each /tmp/velo_seqSS.part is a contiguous range download starting at byte `lo`
(from /tmp/velo_plan.json). We parse local file headers (offset = absolute
header_offset - lo) and slice each STORED .bin out into its own dataset dir.
Missing tail scans (if a block was a few KB short) are back-filled via remotezip.
"""
import json
import struct
from pathlib import Path
from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
plan = json.load(open("/tmp/velo_plan.json"))

with RemoteZip(URL) as z:
    import re
    by_seq = {}
    for i in z.infolist():
        m = re.search(r"/sequences/(\d\d)/velodyne/.*\.bin$", i.filename)
        if m and m.group(1) in plan:
            by_seq.setdefault(m.group(1), []).append(i)

    for s, members in by_seq.items():
        lo = plan[s]["lo"]
        part = Path(f"/tmp/velo_seq{s}.part")
        dest = Path(f"$HOME/datasets/kitti_seq{s}_velodyne")
        dest.mkdir(parents=True, exist_ok=True)
        blob = part.stat().st_size
        ok, missing = 0, []
        with open(part, "rb") as f:
            for m in members:
                rel = m.header_offset - lo
                f.seek(rel)
                lh = f.read(30)
                if len(lh) < 30 or lh[:4] != b"PK\x03\x04":
                    missing.append(m); continue
                name_len, extra_len = struct.unpack("<HH", lh[26:30])
                ds = rel + 30 + name_len + extra_len
                if ds + m.compress_size > blob:
                    missing.append(m); continue
                f.seek(ds)
                (dest / Path(m.filename).name).write_bytes(f.read(m.compress_size))
                ok += 1
        # back-fill any missing via remotezip
        for m in missing:
            with z.open(m) as src:
                (dest / Path(m.filename).name).write_bytes(src.read())
        print(f"seq{s}: extracted {ok}, back-filled {len(missing)} -> {dest} "
              f"({len(list(dest.glob('*.bin')))} bins)", flush=True)
print("DONE", flush=True)
