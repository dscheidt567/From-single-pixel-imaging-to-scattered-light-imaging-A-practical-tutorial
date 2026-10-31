"""Data access for the tutorial "From single-pixel imaging to scattered light imaging".

    import sli_data
    DATA_ROOT = sli_data.get_data(quick=True)        # quick run: recorded data of the published run
    DATA_ROOT = sli_data.get_data()                  # full run: raw TIFF frames (~6 GB)

Quick run: the data set holds everything that one full run reads from the raw frames (the 43 x 43
binned stacks, the full-resolution data of the four 100 x 100 px ROIs and the calibration-region
profiles). install_replay() answers the data loaders of the notebook from this recording, so the
quick run reproduces the published settings and results without the raw frames.

Data location, in this order:
  1. Environment variable SLI_DATA_ROOT (full data: a folder) or SLI_QUICK_ROOT (quick data: a folder
     or sli_quick.zip) -> used directly, nothing is downloaded.
  2. Otherwise the zips are downloaded once into ./data or ./data_quick next to this file.
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- sources
# Zenodo record (doi:10.5281/zenodo.23825920); one id for all zips, or {"full": "...", "quick": "..."}
ZENODO_RECORD = "23825920"
# Google Drive file ids ("Anyone with the link"); fallback if the Zenodo download fails
GDRIVE_IDS = {
    "can_16.zip": "1pPUO0OWzbh0z3iqRgQLV49dqCJCSzY8m",
    "had_16.zip": "1-TaeTM8Ulb7IwQsQ0JK7J5y8FV6ScsD-",
    "can_32.zip": "1YIQfSAW8HbGzSEDL5Zv0bwh9FM8oxdNO",
    "had_32.zip": "1dA2ccrrc2n15CLqxnwX2Y-sFuwLwNw8w",
    "can_64.zip": "1dZmxrm1BtBcRckBcEKsTsozGufECH5zY",
    "had_64.zip": "1ujAdvw-SWafMFHd6ucbJLB29P-qK1pPc",
    "sli_quick.zip": "145uSPUxel3jSaXFxDO-Urg3WrUk5ok1Y",
}
# md5 checksums of the zips; a download with a different checksum is rejected
MD5 = {
    "can_16.zip": "8275851a927e5ee4fafd623a4153ebfb",
    "had_16.zip": "e2c21952a92325711419b6dc4ee855c0",
    "can_32.zip": "715352e77fc7072677c3549984cd7f30",
    "had_32.zip": "5d7e8760e8e32c9254d38200eaa800ea",
    "can_64.zip": "5a4bf05d5794ae6296372d0069c570b2",
    "had_64.zip": "5401a75cb7a800b1d51879d0374eba42",
    "sli_quick.zip": "603b80bdb64540396d9b3d6205713205",
}

DATASETS = {
    "full": ["can_16", "had_16", "can_32", "had_32", "can_64", "had_64"],
    "quick": ["sli_quick"],
}
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = {"full": HERE / "data", "quick": HERE / "data_quick"}  # separate, so the stack search finds no duplicates
ENV_VAR = {"full": "SLI_DATA_ROOT", "quick": "SLI_QUICK_ROOT"}
_DONE = ".complete"


# ===================================================================== download
def _mount_if_drive(path):
    if str(path).startswith("/content/drive") and not os.path.exists("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")


def _sources(name):
    """Download sources for name.zip, in the order they are tried."""
    zname = f"{name}.zip"
    record = ZENODO_RECORD
    if isinstance(record, dict):
        record = record.get("quick" if name in DATASETS["quick"] else "full")
    srcs = []
    if record:
        srcs.append(f"https://zenodo.org/records/{record}/files/{zname}?download=1")
    if GDRIVE_IDS.get(zname):
        srcs.append("gdrive:" + GDRIVE_IDS[zname])
    if not srcs:
        raise RuntimeError(f"No download source for {zname}. Set ZENODO_RECORD or GDRIVE_IDS in sli_data.py, "
                           "or point SLI_DATA_ROOT / SLI_QUICK_ROOT to an existing copy of the data.")
    return srcs


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(src, dest):
    tmp = dest.with_name(dest.name + ".part")
    if src.startswith("gdrive:"):
        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
            import gdown
        if gdown.download(id=src.split(":", 1)[1], output=str(tmp), quiet=False) is None:
            raise IOError(f"Google Drive download failed for {dest.name} "
                          "(check the sharing settings, or the daily download quota was hit).")
    else:
        with urllib.request.urlopen(src) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
    tmp.rename(dest)


def get_data(quick=False, root=None, keep_zip=False):
    """Folder of the data set; downloads and unpacks missing parts first."""
    kind = "quick" if quick else "full"
    env = os.environ.get(ENV_VAR[kind])
    if root is None and env:
        _mount_if_drive(env)
        env = Path(env)
        if not env.exists():
            raise FileNotFoundError(f"{ENV_VAR[kind]}={env} does not exist.")
        if env.suffix != ".zip":
            print("Using data in", env)
            return str(env)
        root = DEFAULT_ROOT[kind]                      # a zip (e.g. on Drive): unpack a fresh local copy
        shutil.rmtree(root / env.stem, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(env) as z:
            z.extractall(root)
        (root / env.stem / _DONE).touch()
        print(f"Unpacked {env} to {root / env.stem}")
        return str(root / env.stem)

    root = Path(root or DEFAULT_ROOT[kind])
    root.mkdir(parents=True, exist_ok=True)
    for name in DATASETS[kind]:
        if (root / name / _DONE).exists():
            continue
        zpath = root / f"{name}.zip"
        if not zpath.exists():
            for i, src in enumerate(_sources(name)):
                print(f"Downloading {zpath.name} from {'Zenodo' if src.startswith('http') else 'Google Drive'} ...")
                try:
                    _download(src, zpath)
                    break
                except Exception as err:
                    zpath.with_name(zpath.name + ".part").unlink(missing_ok=True)
                    if i == len(_sources(name)) - 1:
                        raise
                    print(f"  failed ({err}); trying the next source")
        expected = MD5.get(zpath.name)
        if expected and _md5(zpath) != expected:
            zpath.unlink()
            raise IOError(f"Checksum mismatch for {zpath.name}; the file was deleted, please run again.")
        print(f"Unpacking {zpath.name} ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
        (root / name / _DONE).touch()
        if not keep_zip:
            zpath.unlink()
    data_root = root / "sli_quick" if quick else root
    print("Data ready in", data_root)
    return str(data_root)


def quick_meta(data_root):
    """Description of the quick data set (quick_meta.json), or None for the raw data."""
    p = Path(data_root) / "quick_meta.json"
    return json.loads(p.read_text()) if p.exists() else None


# ===================================================================== replay of the quick data set
_LOADERS = ("load_binned_stack", "load_roi_profiles", "load_stack_and_rois")


def _norm(x):
    """Hashable, JSON-friendly form of call arguments (ROIs, bins, N)."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return [[str(k), _norm(v)] for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _key(name, stack, args, kwargs):
    return json.dumps([name, stack, _norm(list(args)), _norm(kwargs)])


def _decode(obj, arrays):
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    if isinstance(obj, dict):
        if "__nd__" in obj:
            return arrays[obj["__nd__"]].copy()            # a fresh copy for every call
        if "__tuple__" in obj:
            return tuple(_decode(v, arrays) for v in obj["__tuple__"])
        if "__dict__" in obj:
            return {_hashable(_decode(k, arrays)): _decode(v, arrays) for k, v in obj["__dict__"]}
    return obj


def _hashable(k):
    return tuple(_hashable(v) for v in k) if isinstance(k, (list, tuple)) else k


def install_replay(ns, module, data_root):
    """Replace the data loaders in `ns` (and in the helper module) by look-ups in the quick data set."""
    root = Path(data_root)
    index = json.loads((root / "index.json").read_text())
    meta, calls = index["meta"], index["calls"]
    npz = np.load(root / "arrays.npz", allow_pickle=False)

    class _Arrays(dict):                                   # loads every array once, on first use
        def __missing__(self, k):
            self[k] = npz[k]
            return self[k]
    arrays = _Arrays()

    def discover_stacks(data_root, ns_wanted, *args, **kwargs):
        ok = [n for n in ns_wanted if f"can_{n}" in meta["stacks"] and f"had_{n}" in meta["stacks"]]
        return {(b, n): f"quick:{b}_{n}" for n in ok for b in ("can", "had")}, ok

    def frame_files(d, n, *args, **kwargs):
        return [f"{d}/{i}" for i in range(n)]

    def read_frame(path, *args, **kwargs):
        # only the frame size is used; the raw frames are not part of the quick data set
        return np.broadcast_to(np.zeros((), meta["frame_dtype"]), meta["frame_shape"])

    def replay(name):
        def f(d, *args, **kwargs):
            key = _key(name, str(d).replace("quick:", "", 1), args, kwargs)
            if key not in calls:
                raise KeyError(f"{name}{(d,) + args} is not part of the quick data set. It holds exactly the data "
                               "of the published run (same ROIs, binning and N); for other settings use QUICK_RUN = False.")
            return _decode(calls[key], arrays)
        f.__name__ = name
        return f

    funcs = {"discover_stacks": discover_stacks, "frame_files": frame_files, "read_frame": read_frame}
    funcs.update({name: replay(name) for name in _LOADERS})
    for name, f in funcs.items():
        ns[name] = f
        if module is not None and hasattr(module, name):
            setattr(module, name, f)
    print(f"Quick run: data loaders answer from the recorded data ({meta['n_calls']} calls, frames "
          f"{meta['frame_shape'][0]} x {meta['frame_shape'][1]} px).")
