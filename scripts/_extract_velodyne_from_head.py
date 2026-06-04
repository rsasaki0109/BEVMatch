"""Extract seq00 .bin scans from the partial velodyne zip (first 8.83 GB block).

seq00 is a contiguous STORED block at the start of the archive. We fetch the
central directory (filename -> header_offset, compress_size) via remotezip, then
slice each member straight out of the locally downloaded head blob by parsing
its local file header. No decompression (the scans are stored, not deflated).
"""
import struct
import sys
from pathlib import Path
from remotezip import RemoteZip

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
HEAD = Path("/tmp/kitti_velo_head.zip")
DEST = Path("/home/sasaki/datasets/kitti_seq00_velodyne")
DEST.mkdir(parents=True, exist_ok=True)

with RemoteZip(URL) as z:
    members = [i for i in z.infolist()
               if "/sequences/00/velodyne/" in i.filename and i.filename.endswith(".bin")]
print(f"seq00 members: {len(members)}", flush=True)

blob_size = HEAD.stat().st_size
with open(HEAD, "rb") as f:
    ok = 0
    for m in members:
        f.seek(m.header_offset)
        lh = f.read(30)
        if lh[:4] != b"PK\x03\x04":
            print(f"  bad local header at {m.filename}", flush=True)
            continue
        name_len, extra_len = struct.unpack("<HH", lh[26:30])
        data_start = m.header_offset + 30 + name_len + extra_len
        if data_start + m.compress_size > blob_size:
            print(f"  truncated: {m.filename} needs {data_start + m.compress_size}, have {blob_size}", flush=True)
            continue
        f.seek(data_start)
        data = f.read(m.compress_size)
        out = DEST / Path(m.filename).name
        out.write_bytes(data)
        ok += 1
        if ok % 500 == 0 or ok == len(members):
            print(f"  extracted {ok}/{len(members)}", flush=True)
print(f"DONE: {ok} scans -> {DEST}", flush=True)
