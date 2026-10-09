"""
prepare_ddd.py
==============

Automated cleaning + tensor structuring pipeline for the
Driver Drowsiness Dataset (DDD).

Input layout (as shipped):
    <data-dir>/
        Drowsy/       A0001.png, A0002.png, ...   (label = 1)
        Non Drowsy/   a0002.png, a0003.png, ...   (label = 0)

What it does
------------
1. Scans both class folders and parses the filename into
   `subject` (letters) + `frame` (digits).
2. Cleans the data:
      - validates every PNG decodes correctly (Pillow),
      - forces RGB, rejects images that are not the native size,
      - drops exact duplicates (SHA-256),
      - writes a full `cleaning_report.csv`.
3. Builds an index (path, subject, frame, label).
4. Splits train/val/test **by subject** (grouped) so the same face
   never leaks between splits (70/15/15 by default, seed 42).
5. Materializes tensors as uint8 `.npz` per split
   (X: [N, 227, 227, 3], y: [N]).
6. Saves per-channel normalization stats (mean/std) computed on the
   train split only -> `normalization.json`.

The heavy arrays are written through a memory-mapped `.npy` buffer so the
pipeline never holds a full split in RAM.

Example
-------
    python src/prepare_ddd.py \
        --data-dir "D:/IA/ProyectoRedesNeuronales/DriversDrowsinessDataset/Driver Drowsiness Dataset (DDD)" \
        --out-dir  "D:/IA/ProyectoRedesNeuronales/processed"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASS_DIRS = {"Drowsy": 1, "Non Drowsy": 0}
NAME_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    path: str
    subject: str
    frame: int
    label: int
    status: str = "ok"
    reason: str = ""


@dataclass
class CleaningStats:
    total: int = 0
    kept: int = 0
    dropped: int = 0
    dropped_unreadable: int = 0
    dropped_wrong_size: int = 0
    dropped_duplicate: int = 0
    dropped_badname: int = 0
    report_rows: List[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_of_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_name(stem: str):
    m = NAME_RE.match(stem)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def validate_image(path: str, native_size: int):
    """Decode fully; return (ok, reason, shape)."""
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            w, h = im.size
            if (w, h) != (native_size, native_size):
                return False, f"wrong_size_{w}x{h}", None
        return True, "", (h, w, 3)
    except Exception as exc:  # noqa: BLE001 - report anything that fails
        return False, f"unreadable:{type(exc).__name__}", None


# --------------------------------------------------------------------------- #
# Scan + clean
# --------------------------------------------------------------------------- #
def scan_and_clean(data_dir: str, native_size: int, do_hash: bool) -> tuple[list, CleaningStats]:
    stats = CleaningStats()
    records: List[Record] = []
    seen_hashes: Dict[str, str] = {}

    for dirname, label in CLASS_DIRS.items():
        folder = os.path.join(data_dir, dirname)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        for fname in tqdm(files, desc=f"Cleaning {dirname}", unit="img"):
            stats.total += 1
            path = os.path.join(folder, fname)
            stem = os.path.splitext(fname)[0]
            subject, frame = parse_name(stem)

            rec = Record(path=path, subject=subject or "?", frame=frame or -1, label=label)

            if subject is None:
                rec.status, rec.reason = "dropped", "bad_name"
                stats.dropped += 1
                stats.dropped_badname += 1
                stats.report_rows.append(vars(rec))
                continue

            ok, reason, _shape = validate_image(path, native_size)
            if not ok:
                rec.status, rec.reason = "dropped", reason
                stats.dropped += 1
                if reason.startswith("unreadable"):
                    stats.dropped_unreadable += 1
                else:
                    stats.dropped_wrong_size += 1
                stats.report_rows.append(vars(rec))
                continue

            if do_hash:
                digest = sha256_of_file(path)
                if digest in seen_hashes:
                    rec.status, rec.reason = "dropped", f"duplicate_of:{os.path.basename(seen_hashes[digest])}"
                    stats.dropped += 1
                    stats.dropped_duplicate += 1
                    stats.report_rows.append(vars(rec))
                    continue
                seen_hashes[digest] = path

            rec.status, rec.reason = "kept", ""
            stats.kept += 1
            records.append(rec)
            stats.report_rows.append(vars(rec))

    return records, stats


# --------------------------------------------------------------------------- #
# Grouped split by subject
# --------------------------------------------------------------------------- #
def grouped_split(records: List[Record], val: float, test: float, seed: int) -> pd.DataFrame:
    df = pd.DataFrame([vars(r) for r in records])
    # Drowsy filenames are UPPERCASE and Non-Drowsy lowercase, but they belong
    # to the SAME subject -> normalize case so grouping is leakage-free.
    df["subject"] = df["subject"].str.lower()
    subjects = sorted(df["subject"].unique().tolist())

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(round(n * test)))
    n_val = max(1, int(round(n * val)))
    test_subj = set(subjects[:n_test])
    val_subj = set(subjects[n_test:n_test + n_val])
    train_subj = set(subjects[n_test + n_val:])

    def assign(s):
        if s in test_subj:
            return "test"
        if s in val_subj:
            return "val"
        return "train"

    df["split"] = df["subject"].map(assign)
    return df


# --------------------------------------------------------------------------- #
# Tensor materialization
# --------------------------------------------------------------------------- #
def write_split_npz(df_split: pd.DataFrame, out_dir: str, split: str, native_size: int,
                    collect_stats: bool):
    """Fill a memmapped .npy buffer then pack it into <split>.npz."""
    n = len(df_split)
    if n == 0:
        return 0, None
    shape = (n, native_size, native_size, 3)

    tmp_x = os.path.join(out_dir, f"_tmp_{split}_X.npy")
    X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.uint8, shape=shape)
    y = np.zeros((n,), dtype=np.uint8)

    chan_sum = np.zeros(3, dtype=np.float64)
    chan_sumsq = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    paths = df_split["path"].tolist()
    labels = df_split["label"].tolist()

    for i, (path, lbl) in enumerate(tqdm(list(zip(paths, labels)),
                                         desc=f"Tensors  {split}", unit="img")):
        with Image.open(path) as im:
            im.load()
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        X[i] = arr
        y[i] = lbl
        if collect_stats:
            a = arr.astype(np.float64)
            chan_sum += a.sum(axis=(0, 1))
            chan_sumsq += (a * a).sum(axis=(0, 1))
            pixel_count += arr.shape[0] * arr.shape[1]

    X.flush()

    npz_path = os.path.join(out_dir, f"{split}.npz")
    np.savez(npz_path, X=X, y=y)

    del X
    os.remove(tmp_x)

    norm = None
    if collect_stats and pixel_count:
        mean = chan_sum / pixel_count
        var = np.maximum(chan_sumsq / pixel_count - mean ** 2, 0.0)
        norm = {
            "mean": (mean / 255.0).round(6).tolist(),
            "std": (np.sqrt(var) / 255.0).round(6).tolist(),
            "pixel_count": int(pixel_count),
            "note": "mean/std in [0,1] scale, computed on TRAIN split only",
        }
    return n, norm


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Clean & structure DDD into tensors.")
    p.add_argument("--data-dir", required=True, help="DDD root containing 'Drowsy' and 'Non Drowsy'.")
    p.add_argument("--out-dir", required=True, help="Where tensors/reports are written.")
    p.add_argument("--val", type=float, default=0.15, help="Validation fraction (by subject).")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction (by subject).")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--size", type=int, default=227, help="Expected native image size.")
    p.add_argument("--skip-hash", action="store_true", help="Skip duplicate detection (faster).")
    p.add_argument("--limit", type=int, default=0, help="Debug: cap images per class (0 = all).")
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  DDD pipeline - clean & structure tensors")
    print("=" * 70)

    # 1-2. clean
    records, stats = scan_and_clean(args.data_dir, args.size, do_hash=not args.skip_hash)
    if args.limit and len(records) > args.limit:
        records = records[: args.limit]

    report = pd.DataFrame(stats.report_rows)
    report_path = os.path.join(args.out_dir, "cleaning_report.csv")
    report.to_csv(report_path, index=False)

    # 3-4. index + split
    df = grouped_split(records, args.val, args.test, args.seed)
    tr_s = set(df.loc[df["split"] == "train", "subject"])
    va_s = set(df.loc[df["split"] == "val", "subject"])
    te_s = set(df.loc[df["split"] == "test", "subject"])
    overlap = (tr_s & va_s) | (tr_s & te_s) | (va_s & te_s)
    if overlap:
        raise RuntimeError(f"Subject leakage detected across splits: {sorted(overlap)}")
    splits_path = os.path.join(args.out_dir, "splits.csv")
    df.to_csv(splits_path, index=False)

    # 5. tensors
    summary = {}
    norm = None
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        n, nm = write_split_npz(sub, args.out_dir, split, args.size, collect_stats=(split == "train"))
        summary[split] = {
            "images": int(n),
            "subjects": int(sub["subject"].nunique()),
            "drowsy": int((sub["label"] == 1).sum()),
            "non_drowsy": int((sub["label"] == 0).sum()),
        }
        if nm:
            norm = nm

    # 6. normalization + class weights
    if norm:
        with open(os.path.join(args.out_dir, "normalization.json"), "w", encoding="utf-8") as fh:
            json.dump(norm, fh, indent=2)

    tr = df[df["split"] == "train"]
    n_tr = len(tr)
    n_pos = int((tr["label"] == 1).sum())
    n_neg = int((tr["label"] == 0).sum())
    class_weights = {
        "0": round(n_tr / (2 * n_neg), 6) if n_neg else None,
        "1": round(n_tr / (2 * n_pos), 6) if n_pos else None,
        "counts": {"0": n_neg, "1": n_pos},
    }
    with open(os.path.join(args.out_dir, "class_weights.json"), "w", encoding="utf-8") as fh:
        json.dump(class_weights, fh, indent=2)

    meta = {
        "data_dir": os.path.abspath(args.data_dir),
        "native_size": args.size,
        "seed": args.seed,
        "val_frac": args.val,
        "test_frac": args.test,
        "cleaning": {
            "total": stats.total,
            "kept": stats.kept,
            "dropped": stats.dropped,
            "dropped_unreadable": stats.dropped_unreadable,
            "dropped_wrong_size": stats.dropped_wrong_size,
            "dropped_duplicate": stats.dropped_duplicate,
            "dropped_badname": stats.dropped_badname,
        },
        "splits": summary,
        "class_weights": class_weights,
        "normalization": norm,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    # console summary
    print("\n" + "-" * 70)
    print(f"Total scanned : {stats.total}")
    print(f"Kept          : {stats.kept}")
    print(f"Dropped       : {stats.dropped} "
          f"(unreadable={stats.dropped_unreadable}, wrong_size={stats.dropped_wrong_size}, "
          f"dupes={stats.dropped_duplicate}, bad_name={stats.dropped_badname})")
    print("-" * 70)
    for split, s in summary.items():
        print(f"{split:>5}: {s['images']:>6} imgs | {s['subjects']:>2} subjects | "
              f"drowsy={s['drowsy']:>5} non_drowsy={s['non_drowsy']:>5}")
    if norm:
        print(f"train mean (0-1): {norm['mean']}")
        print(f"train std  (0-1): {norm['std']}")
    print(f"class_weights   : {class_weights}")
    print("-" * 70)
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")
    print("  train.npz / val.npz / test.npz, normalization.json, "
          "class_weights.json, splits.csv, cleaning_report.csv, metadata.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
