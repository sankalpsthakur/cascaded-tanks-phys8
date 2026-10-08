"""Load Cascaded Tanks through the loader's Google Drive URL (ct/protocol.py) and verify the checksums of what was downloaded.

usage: python tools/ct_data_check.py <sha256 of CascadedTanksFiles.zip> <sha256 of dataBenchmark.mat>
Exit code 3 on a checksum mismatch. Prints sizes and checksums only, no statistics of the records.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


want_zip, want_mat = sys.argv[1], sys.argv[2]
train, test = P._load_both()
zp = os.path.join(P.DATA_DIR, "Cascaded_Tanks", "CascadedTanksFiles.zip")
mp = os.path.join(P.DATA_DIR, "Cascaded_Tanks", "CascadedTanksFiles", "dataBenchmark.mat")
got = dict(zip_bytes=os.path.getsize(zp), zip_sha256=sha(zp), mat_bytes=os.path.getsize(mp), mat_sha256=sha(mp),
           n_train=len(train.u), n_test=len(test.u), Ts=float(train.sampling_time),
           test_state_init_window=test.state_initialization_window_length, url=P.GDRIVE_URL)
ok = got["zip_sha256"] == want_zip and got["mat_sha256"] == want_mat
got["match_expected"] = ok
print("DATA_CHECK " + json.dumps(got))
sys.exit(0 if ok else 3)
