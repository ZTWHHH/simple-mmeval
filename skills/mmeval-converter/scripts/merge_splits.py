#!/usr/bin/env python3
"""Merge several per-split HF artifacts (each produced by ``convert.py --mode hf``
with a single split) into a single multi-split artifact ready for ``push_to_hf.py``.

Usage
-----
    python3 merge_splits.py \
        --inputs <dir1> <dir2> ... \
        --out <merged_dir>

Each input directory is expected to contain ``hf_dataset/`` (DatasetDict with
exactly one split) and ``hf_metadata/`` (DatasetDict with one split). The output
contains the union of splits in both configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from datasets import DatasetDict, load_from_disk


def merge(input_dirs: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    default_dd = {}
    metadata_dd = {}
    for d in input_dirs:
        ds = load_from_disk(str(d / "hf_dataset"))
        md = load_from_disk(str(d / "hf_metadata"))
        for split_name, sub in ds.items():
            if split_name in default_dd:
                raise ValueError(f"Duplicate split '{split_name}' across inputs")
            default_dd[split_name] = sub
        for split_name, sub in md.items():
            metadata_dd[split_name] = sub

    DatasetDict(default_dd).save_to_disk(str(out_dir / "hf_dataset"))
    DatasetDict(metadata_dd).save_to_disk(str(out_dir / "hf_metadata"))
    print(f"merged splits: {sorted(default_dd.keys())} -> {out_dir}")
    print(f"  rows per split: " + ", ".join(f"{k}={len(v)}" for k, v in default_dd.items()))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Per-split artifact directories (each has hf_dataset/ + hf_metadata/)")
    p.add_argument("--out", required=True, help="Output directory for the merged artifact")
    args = p.parse_args()
    merge([Path(d) for d in args.inputs], Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
