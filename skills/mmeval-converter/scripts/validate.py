#!/usr/bin/env python3
"""Round-trip a converted artifact through Simple-MMEval's data loaders to verify it.

Usage
-----

# Validate a local artifact (must be the directory with data.json + media/)
python3 validate.py --simple-mmeval /path/to/simple-mmeval --local /path/to/out --split val

# Validate an HF artifact on disk (no push required) — uses load_from_disk and
# monkey-patches mmeval.data.mmeval_hf.load_dataset accordingly.
python3 validate.py --simple-mmeval /path/to/simple-mmeval --hf /path/to/out --split val

What it does
------------
- For local: instantiates `LocalJSONDataset`, iterates a few rows, prints the
  rendered prompt and confirms the media files load as PIL Images.
- For HF: instantiates `MMEvalHFDataset` against the on-disk DatasetDict and
  confirms the same.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from unittest.mock import patch


def validate_local(simple_mmeval: Path, artifact: Path, split: str, n: int) -> int:
    sys.path.insert(0, str(simple_mmeval))
    from mmeval.data.local import LocalJSONDataset
    ns = argparse.Namespace(
        infile=str(artifact / "data.json"),
        img_dir=str(artifact / "media"),
        parallel_per_task=1, rank=0, template=None, resize=None,
    )
    ds = LocalJSONDataset(ns)
    ds.setup_parallel()
    print(f"[local] global rows: {ds.global_length}")
    for i in range(min(n, ds.global_length)):
        s = ds[i]
        msg = s["messages"][0]
        media_imgs = msg.get("media", [])
        sizes = [getattr(m, "size", "video") for m in media_imgs]
        print(f"  [{i}] id={s.get('id')} eval-id={s.get('eval-id')} media={sizes} "
              f"prompt={msg['prompt'][:80]!r}")
    print("[local] OK")
    return 0


def validate_hf(simple_mmeval: Path, artifact: Path, split: str, n: int) -> int:
    from datasets import load_from_disk
    sys.path.insert(0, str(simple_mmeval))
    from mmeval.data.mmeval_hf import MMEvalHFDataset

    def fake_load(*args, **kwargs):
        cfg = kwargs.get("name") or (args[1] if len(args) > 1 else None)
        sp = kwargs.get("split") or (args[2] if len(args) > 2 else None)
        path = artifact / ("hf_metadata" if cfg == "metadata" else "hf_dataset")
        return load_from_disk(str(path))[sp]

    with patch("mmeval.data.mmeval_hf.load_dataset", fake_load):
        ns = argparse.Namespace(
            dataset="mmeval_hf@local-disk", split=split, circular=False, resize=None,
            parallel_per_task=1, rank=0, template=None,
        )
        ds = MMEvalHFDataset(ns); ds.setup_parallel()
        print(f"[hf] global rows: {ds.global_length}")
        for i in range(min(n, ds.global_length)):
            s = ds[i]
            msg = s["messages"][0]
            media_imgs = msg.get("media", [])
            sizes = [getattr(m, "size", "video") for m in media_imgs]
            print(f"  [{i}] id={s.get('id')} eval-id={s.get('eval-id')} media={sizes} "
                  f"prompt={msg['prompt'][:80]!r}")
    print("[hf] OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--simple-mmeval", required=True,
                   help="Path to a simple-mmeval checkout (so mmeval/ is importable)")
    p.add_argument("--local", help="Path to local artifact (with data.json + media/)")
    p.add_argument("--hf", help="Path to HF artifact (with hf_dataset/ + hf_metadata/)")
    p.add_argument("--split", default="val")
    p.add_argument("-n", type=int, default=2, help="Rows to print per artifact")
    args = p.parse_args()
    if not args.local and not args.hf:
        print("ERROR: pass --local and/or --hf", file=sys.stderr)
        return 2
    smm = Path(args.simple_mmeval)
    rc = 0
    if args.local:
        rc |= validate_local(smm, Path(args.local), args.split, args.n)
    if args.hf:
        rc |= validate_hf(smm, Path(args.hf), args.split, args.n)
    return rc


if __name__ == "__main__":
    sys.exit(main())
