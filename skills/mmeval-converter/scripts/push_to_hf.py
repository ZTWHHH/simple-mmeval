#!/usr/bin/env python3
"""Push a converted mm-eval HF dataset (the artifact produced by ``convert.py --mode hf``) to HuggingFace Hub.

The user only needs to supply the HF token and the target repo name; everything else
is read from the artifact directory.

Usage
-----
    python3 push_to_hf.py \
        --artifact-dir /path/to/convert/output \
        --repo-id <user>/<repo> \
        --token <hf_token> \
        [--private]

The artifact directory must contain:
    artifact-dir/
      hf_dataset/      # DatasetDict for the `default` config
      hf_metadata/     # DatasetDict for the `metadata` config

After pushing, the script prints the exact ``simple-mmeval`` invocation to copy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", required=True, help="Directory produced by convert.py --mode hf")
    p.add_argument("--repo-id", required=True, help="Target HF repo id, e.g. username/dataset-name")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (or set HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--no-create", action="store_true",
                   help="Don't try to create the repo (assume it already exists)")
    args = p.parse_args()

    if not args.token:
        print("ERROR: provide --token or set HF_TOKEN", file=sys.stderr)
        return 2

    artifact = Path(args.artifact_dir)
    default_dir = artifact / "hf_dataset"
    metadata_dir = artifact / "hf_metadata"
    if not default_dir.exists() or not metadata_dir.exists():
        print(f"ERROR: expected {default_dir} and {metadata_dir} to exist (run convert.py --mode hf first)",
              file=sys.stderr)
        return 2

    from datasets import load_from_disk
    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    if not args.no_create:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                        private=args.private, exist_ok=True)
        print(f"[hf] ensured repo exists: {args.repo_id} (private={args.private})")

    print(f"[hf] loading {default_dir}")
    default_dd = load_from_disk(str(default_dir))
    print(f"[hf] pushing default config: splits={list(default_dd.keys())}")
    default_dd.push_to_hub(args.repo_id, config_name="default",
                           private=args.private, token=args.token)

    print(f"[hf] loading {metadata_dir}")
    metadata_dd = load_from_disk(str(metadata_dir))
    print(f"[hf] pushing metadata config: splits={list(metadata_dd.keys())}")
    metadata_dd.push_to_hub(args.repo_id, config_name="metadata",
                            private=args.private, token=args.token)

    print()
    print(f"Done. Run with simple-mmeval:")
    split = list(default_dd.keys())[0]
    print(f"  python mmeval/run.py \\")
    print(f"      --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \\")
    print(f"      --dataset mmeval_hf@{args.repo_id} \\")
    print(f"      --split {split} \\")
    print(f"      --out_dir work_dirs/$(basename {args.repo_id}) \\")
    print(f"      --gpu_per_parallel 1 --parallel_per_task 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
