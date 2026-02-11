import os
import json
import ijson
import numpy as np
from collections.abc import Mapping
from typing import Tuple, Dict, List

def stream_array_to_memmap(json_path: str, key: str, out_npy: str, dtype=np.float32):
    # First pass: count rows and infer columns from first item
    n = 0
    cols = None
    with open(json_path, "rb") as f:
        for item in ijson.items(f, f"{key}.item"):
            if cols is None:
                if not isinstance(item, list):
                    raise ValueError(f"Expected list rows in {key}")
                cols = len(item)
            n += 1

    if cols is None:
        raise ValueError(f"No items found for key {key} in {json_path}")

    # Create memmap and fill in second pass
    mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype=dtype, shape=(n, cols))
    idx = 0
    with open(json_path, "rb") as f:
        for item in ijson.items(f, f"{key}.item"):
            mm[idx] = np.asarray(item, dtype=dtype)
            idx += 1
    # flush
    del mm
    return n, cols

def stream_kv_to_files(json_path: str, dict_key: str, out_dir: str):
    # Writes each key/value pair (value=2D list) to out_dir/<key>.npy
    os.makedirs(out_dir, exist_ok=True)
    with open(json_path, "rb") as f:
        for k, v in ijson.kvitems(f, f"{dict_key}"):
            arr = np.asarray(v, dtype=np.float32)
            np.save(os.path.join(out_dir, f"{k}.npy"), arr)

def stream_list_to_json(json_path: str, key: str, out_json: str):
    items = []
    with open(json_path, "rb") as f:
        for it in ijson.items(f, f"{key}.item"):
            items.append(it)
    with open(out_json, "w") as f:
        json.dump(items, f)

def stream_dict_to_json(json_path: str, key: str, out_json: str):
    d = {}
    with open(json_path, "rb") as f:
        for k, v in ijson.kvitems(f, key):
            d[k] = v
    with open(out_json, "w") as f:
        json.dump(d, f)

def convert_vindr_json_to_disk(json_path: str, out_dir: str):
    """
    Produces:
      - out_dir/y_true.npy  (memmap-able)
      - out_dir/y_score.npy
      - out_dir/image_ids.json
      - out_dir/orig_shapes.json
      - out_dir/attn_maps/<image_id>.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    print("[INFO] Converting big JSON to disk-backed format (streaming)...")
    y_true_n, y_true_c = stream_array_to_memmap(json_path, "y_true", os.path.join(out_dir, "y_true.npy"))
    print(f"[INFO] Written y_true: {y_true_n} x {y_true_c}")
    y_score_n, y_score_c = stream_array_to_memmap(json_path, "y_score", os.path.join(out_dir, "y_score.npy"))
    print(f"[INFO] Written y_score: {y_score_n} x {y_score_c}")

    stream_list_to_json(json_path, "image_ids", os.path.join(out_dir, "image_ids.json"))
    print(f"[INFO] Written image_ids.json")
    stream_dict_to_json(json_path, "orig_shapes", os.path.join(out_dir, "orig_shapes.json"))
    print(f"[INFO] Written orig_shapes.json")

    attn_out = os.path.join(out_dir, "attn_maps")
    stream_kv_to_files(json_path, "attn_maps", attn_out)
    print(f"[INFO] Written per-image attention maps to {attn_out}")

class RawOutputsLazy:
    """
    Lazy access to converted outputs on disk.
    - y_true and y_score are memory-mapped via np.load(..., mmap_mode='r')
    - image_ids and orig_shapes are small JSON files loaded into memory
    - attn maps are loaded per-image from `attn_maps/<image_id>.npy`
    """
    def __init__(self, base_dir: str, attn_dirname: str = "attn_maps"):
        self.base_dir = base_dir
        self.y_true_path = os.path.join(base_dir, "y_true.npy")
        self.y_score_path = os.path.join(base_dir, "y_score.npy")
        self.image_ids_path = os.path.join(base_dir, "image_ids.json")
        self.orig_shapes_path = os.path.join(base_dir, "orig_shapes.json")
        self.attn_dir = os.path.join(base_dir, attn_dirname)

        if not os.path.exists(self.y_true_path) or not os.path.exists(self.y_score_path):
            raise FileNotFoundError("y_true.npy / y_score.npy not found - run conversion first")

        self.y_true = np.load(self.y_true_path, mmap_mode="r")
        self.y_score = np.load(self.y_score_path, mmap_mode="r")

        with open(self.image_ids_path, "r") as f:
            self.image_ids = json.load(f)
        with open(self.orig_shapes_path, "r") as f:
            self.orig_shapes = json.load(f)

        # small LRU cache for attn maps (keep last few in memory)
        self._attn_cache: Dict[str, np.ndarray] = {}
        self._cache_size = 16

    def get_attn(self, image_id: str) -> np.ndarray:
        path = os.path.join(self.attn_dir, f"{image_id}.npy")
        if image_id in self._attn_cache:
            return self._attn_cache[image_id]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attention map file missing: {path}")
        a = np.load(path)  # small (~256x256) so safe to fully load
        # maintain LRU style simple cache
        self._attn_cache[image_id] = a
        if len(self._attn_cache) > self._cache_size:
            # pop oldest inserted key
            oldest = next(iter(self._attn_cache))
            self._attn_cache.pop(oldest, None)
        return a

    def __len__(self):
        return len(self.image_ids)

    def iter_items(self):
        # yields (image_id, y_true_row, y_score_row, attn_map_lazy, orig_shape)
        for idx, iid in enumerate(self.image_ids):
            yt = self.y_true[idx]
            ys = self.y_score[idx]
            shape = tuple(self.orig_shapes[iid])
            yield iid, yt, ys, lambda _iid=iid: self.get_attn(_iid), shape

class LazyAttnMaps(Mapping):
    """Mapping-like view that loads attention maps on demand from RawOutputsLazy."""
    def __init__(self, loader):
        self.loader = loader
        # assume loader.image_ids lists ids that have attention files
        self._ids = list(loader.image_ids)

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return iter(self._ids)

    def keys(self):
        return self._ids

    def __contains__(self, k):
        return k in self._ids

    def __getitem__(self, k):
        if k not in self._ids:
            raise KeyError(k)
        return self.loader.get_attn(k)


# Example usage:
# convert_vindr_json_to_disk("vindr_raw_outputs.json", "vindr_raw_disk")
# loader = RawOutputsLazy("vindr_raw_disk")
# # Access y_true and y_score without loading all attention maps:
# print(loader.y_true.shape, loader.y_score.shape)
# # Get attention for a single image:
# att = loader.get_attn(loader.image_ids[0])
