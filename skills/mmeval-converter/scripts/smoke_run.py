#!/usr/bin/env python3
"""End-to-end smoke test for a converted artifact: tiny model + a few rows through Simple-MMEval.

Why
---
`validate.py` only proves the artifact loads through Simple-MMEval's data
loaders. It does NOT prove that the chat template / processor accepts the
rendered prompt + media count, that inference produces output, or that the
result file is well-formed. A converted dataset can pass `validate.py` and
still fail at run time (most often: processor sees a different `<image>`
count than the row's media list).

This script truncates the local artifact to N rows, spawns
`mmeval/run.py` against it with a small VL model, and asserts every row got
a non-empty `response`.

Local mode only — both the local and HF artifacts come from the same source
pass in `convert.py`, so a green local smoke is sufficient evidence that the
HF artifact will run too. (HF-mode smoke would need a load_from_disk shim
inside `mmeval/run.py`; not worth the complexity here.)

Usage
-----

    python3 smoke_run.py \\
        --simple-mmeval /path/to/simple-mmeval \\
        --local /path/to/converted/artifact \\
        [--model Qwen/Qwen2.5-VL-3B-Instruct] \\
        [--limit 2] \\
        [--gpu 0]

Exits non-zero if the run errored, `result.json` is missing, or any row's
`response` is empty.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _build_truncated_artifact(src: Path, limit: int, work: Path) -> Path:
    data = json.loads((src / "data.json").read_text())
    if not isinstance(data, list):
        raise SystemExit(f"ERROR: {src/'data.json'} is not a list")
    if limit < 1 or limit > len(data):
        limit = min(max(limit, 1), len(data))
    truncated = data[:limit]
    work.mkdir(parents=True, exist_ok=True)
    (work / "data.json").write_text(json.dumps(truncated, ensure_ascii=False, indent=2))
    media_link = work / "media"
    if media_link.exists() or media_link.is_symlink():
        media_link.unlink()
    media_link.symlink_to((src / "media").resolve(), target_is_directory=True)
    return work


def _spawn_run(simple_mmeval: Path, work: Path, model: str, out_dir: Path, gpu: str,
               python_bin: str, attn_impl: str) -> int:
    # LocalJSONDataset re-renders prompts using whatever template is loaded
    # (default: just `{{ question }}`), which silently strips `<image>` and any
    # post-prompt suffix from our pre-rendered prompt. Pass a passthrough
    # template so the loader uses the prompt we already rendered in convert.py.
    passthrough = work / "passthrough.j2"
    passthrough.write_text("{{ prompt }}")
    cmd = [
        python_bin, "mmeval/run.py",
        "--model_name_or_path", model,
        "--dataset", "local@json",
        "--infile", str(work / "data.json"),
        "--img_dir", str(work / "media"),
        "--out_dir", str(out_dir),
        "--gpu_per_parallel", "1",
        "--parallel_per_task", "1",
        "--attn_implementation", attn_impl,
        "--template", str(passthrough),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{simple_mmeval}:{env.get('PYTHONPATH', '')}"
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[smoke] cwd={simple_mmeval}")
    print(f"[smoke] cmd={' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(simple_mmeval), env=env).returncode


def _assert_responses(out_dir: Path, expected: int) -> int:
    rj = out_dir / "result.json"
    if not rj.exists():
        print(f"[smoke] FAIL: {rj} missing — run did not complete", file=sys.stderr)
        return 3
    rows = json.loads(rj.read_text())
    if len(rows) != expected:
        print(f"[smoke] FAIL: expected {expected} rows, got {len(rows)}", file=sys.stderr)
        return 4
    def flat_text(v):
        # mmeval may store `response` as str, list[str], or list[list[str]].
        # Treat anything that produces a non-empty leaf string as valid.
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return "".join(flat_text(x) for x in v)
        return str(v)

    empty = [r for r in rows if not flat_text(r.get("response")).strip()]
    if empty:
        print(f"[smoke] FAIL: {len(empty)}/{len(rows)} rows have empty `response`", file=sys.stderr)
        for r in empty[:2]:
            print(f"  empty row: id={r.get('id')} eval-id={r.get('eval-id')}", file=sys.stderr)
        return 5
    print(f"[smoke] OK — {len(rows)} rows, all have non-empty `response`")
    print(f"[smoke] sample response: {flat_text(rows[0]['response'])[:160]!r}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--simple-mmeval", required=True,
                   help="Path to a simple-mmeval checkout")
    p.add_argument("--local", required=True,
                   help="Path to local artifact dir (with data.json + media/)")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                   help="HF model id for the smoke run (small VL model recommended)")
    p.add_argument("--limit", type=int, default=2,
                   help="Number of rows to smoke-test (default 2)")
    p.add_argument("--gpu", default="",
                   help="CUDA_VISIBLE_DEVICES (e.g. '0'); defaults to whatever is set in env")
    p.add_argument("--python", default="python3",
                   help="Python executable to run mmeval/run.py (needs torch + datasets). "
                        "Inference still dispatches to the per-model conda env from registry.py; "
                        "this is just the orchestrator interpreter.")
    p.add_argument("--attn-implementation", default="sdpa",
                   help="Attention backend passed to mmeval/run.py. Defaults to sdpa because "
                        "the smoke test should not depend on a working flash-attn install in "
                        "the per-model conda env (a common failure mode that prompts for "
                        "interactive input and breaks the smoke run).")
    p.add_argument("--keep-tmp", action="store_true",
                   help="Leave the temp work + out dirs on disk for inspection")
    args = p.parse_args()

    src = Path(args.local).resolve()
    if not (src / "data.json").is_file():
        print(f"ERROR: {src/'data.json'} not found", file=sys.stderr)
        return 2
    if not (src / "media").is_dir():
        print(f"ERROR: {src/'media'} not found", file=sys.stderr)
        return 2

    smm = Path(args.simple_mmeval).resolve()
    if not (smm / "mmeval" / "run.py").is_file():
        print(f"ERROR: {smm/'mmeval/run.py'} not found", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="mmeval_smoke_"))
    work = tmp / "artifact"
    out_dir = tmp / "out"
    try:
        _build_truncated_artifact(src, args.limit, work)
        rc = _spawn_run(smm, work, args.model, out_dir, args.gpu, args.python, args.attn_implementation)
        if rc != 0:
            print(f"[smoke] FAIL: mmeval/run.py exited {rc}", file=sys.stderr)
            return rc
        actual = len(json.loads((work / "data.json").read_text()))
        return _assert_responses(out_dir, actual)
    finally:
        if args.keep_tmp:
            print(f"[smoke] kept tmp dir: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
