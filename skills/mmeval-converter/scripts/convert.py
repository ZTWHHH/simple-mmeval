#!/usr/bin/env python3
"""Convert any source dataset into Simple-MMEval-runnable format.

Two output modes:
  - local: writes <out>/data.json + <out>/media/  (use with --dataset local@json)
  - hf:    writes <out>/hf_dataset/ + <out>/hf_metadata/  (push_to_hub-ready)
  - both:  emits both in a single source pass.

Source can be a HuggingFace dataset (--hf <repo> --split <split>) or a local
JSON list (--json <path> --media-dir <dir>).

Column mapping is given as a list of `canonical=source` pairs:
  --map id=question_id question=question image=image answer=answers

Canonical keys understood by this script:
  id, question, answer, hint, options, choices,
  image  | images | media   (single media field — single value or list),
  category (passed through as a message field)

Any other `--map foo=bar` pairs are passed to the message dict.

Examples
--------

# 1. VizWiz-VQA val -> local JSON + media
python3 convert.py \
    --hf lmms-lab/VizWiz-VQA --split val \
    --map id=question_id question=question image=image answer=answers \
    --template '<image>{{ question }}\nAnswer the question using a single word or phrase.' \
    --mode local --out /tmp/vizwiz_val_local --workers 16

# 2. VizWiz-VQA val -> HF dataset (push_to_hub ready)
python3 convert.py \
    --hf lmms-lab/VizWiz-VQA --split val \
    --map id=question_id question=question image=image answer=answers \
    --template '<image>{{ question }}\nAnswer the question using a single word or phrase.' \
    --mode hf --out /tmp/vizwiz_val_hf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
              ".mpeg", ".mpg", ".m4v", ".3gp", ".3g2", ".ts", ".mts", ".vob"}
IMAGE_KEYS = {"image", "images", "media"}

# Force unbuffered stdout so background invocations stream their progress.
sys.stdout.reconfigure(line_buffering=True)


def is_video(path: str) -> bool:
    return os.path.splitext(path.split("?")[0])[-1].lower() in VIDEO_EXTS


def parse_map(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--map entry '{p}' must be canonical=source")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    if "id" not in out:
        raise ValueError("--map must include id=<source-field>")
    if "question" not in out:
        raise ValueError("--map must include question=<source-field>")
    return out


def load_template(template: Optional[str]) -> Optional[str]:
    if not template:
        return None
    if os.path.exists(template):
        with open(template) as f:
            return f.read()
    return template


def render_template(template: str, ctx: Dict[str, Any]) -> str:
    from jinja2 import Environment
    env = Environment()
    env.globals.update({"zip": zip, "enumerate": enumerate, "len": len, "range": range,
                        "list": list, "dict": dict, "str": str, "int": int, "float": float,
                        "bool": bool, "sum": sum, "max": max, "min": min})
    return env.from_string(template).render(**ctx)


def to_pil(value: Any):
    """Best-effort to turn a source value into a PIL.Image. Returns video paths as-is, None if unusable."""
    from PIL import Image
    import io, base64, requests
    if value is None:
        return None
    if hasattr(value, "size") and hasattr(value, "mode"):
        return value
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, str):
        if os.path.exists(value):
            if is_video(value):
                return value
            return Image.open(value)
        if value.startswith("http"):
            if is_video(value):
                return value
            return Image.open(requests.get(value, stream=True).raw)
        try:
            return Image.open(io.BytesIO(base64.b64decode(value)))
        except Exception:
            return None
    return None


def normalize_media_field(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return list(raw)
    return [raw]


def write_image_to_disk(item: Any, out_dir: Path, stem: str, idx: int,
                        image_format: str = "png", jpeg_quality: int = 92) -> Optional[str]:
    """Save an image-like item; returns basename. Videos are copied/downloaded."""
    import shutil
    pil_or_path = to_pil(item)
    if pil_or_path is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pil_or_path, str):
        ext = os.path.splitext(pil_or_path.split("?")[0])[-1].lower() or ".mp4"
        name = f"{stem}_{idx}{ext}"
        target = out_dir / name
        if pil_or_path.startswith("http"):
            import requests
            with requests.get(pil_or_path, stream=True) as r:
                r.raise_for_status()
                with open(target, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
        else:
            shutil.copyfile(pil_or_path, target)
        return name
    img = pil_or_path
    if image_format == "jpeg":
        if img.mode != "RGB":
            img = img.convert("RGB")
        name = f"{stem}_{idx}.jpg"
        img.save(out_dir / name, format="JPEG", quality=jpeg_quality, optimize=False)
    else:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        name = f"{stem}_{idx}.png"
        img.save(out_dir / name, format="PNG", optimize=False, compress_level=1)
    return name


def build_message(row: Dict[str, Any], colmap: Dict[str, str], answer_join: str,
                  answer_as_list: bool) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": "user"}
    msg["question"] = row.get(colmap["question"], "") or ""
    if "answer" in colmap:
        ans = row.get(colmap["answer"])
        if isinstance(ans, list) and not answer_as_list:
            ans = answer_join.join(str(a) for a in ans) if ans else ""
        msg["answer"] = ans if ans is not None else ""
    else:
        msg["answer"] = ""
    msg["hint"] = row.get(colmap["hint"], "") if "hint" in colmap else ""
    if "options" in colmap:
        opts = row.get(colmap["options"]) or {}
        if isinstance(opts, list):
            opts = {chr(ord("A") + i): v for i, v in enumerate(opts)}
        msg["options"] = opts
    else:
        msg["options"] = {}
    if "choices" in colmap:
        ch = row.get(colmap["choices"]) or []
        msg["choices"] = list(ch) if not isinstance(ch, list) else ch
    else:
        msg["choices"] = []
    for k, src in colmap.items():
        if k in {"id", "image", "images", "media", "question", "answer", "hint", "options", "choices"}:
            continue
        msg[k] = row.get(src)
    return msg


def get_media_field(row: Dict[str, Any], colmap: Dict[str, str]) -> List[Any]:
    if "images" in colmap:
        return normalize_media_field(row.get(colmap["images"]))
    if "media" in colmap:
        return normalize_media_field(row.get(colmap["media"]))
    if "image" in colmap:
        return normalize_media_field(row.get(colmap["image"]))
    return []


def iter_source(args) -> Iterator[Tuple[Dict[str, Any], int]]:
    """Iterate (row, global_index). Optionally explode a list-valued field into
    multiple rows, merging the inner dict's keys onto the outer row. The original
    id is preserved as args.id_template (default '{id}_q{idx}') so each exploded
    row stays unique."""
    if args.hf:
        from datasets import load_dataset
        base = load_dataset(args.hf, split=args.split)
        raw_iter = enumerate(base)
    else:
        with open(args.json) as f:
            data = json.load(f)
        raw_iter = enumerate(data)

    explode = getattr(args, "explode", None)
    # The original id source is captured by main() before any override.
    id_field = getattr(args, "_original_id_field", None)

    global_idx = 0
    for _, row in raw_iter:
        if not explode:
            yield row, global_idx
            global_idx += 1
            continue
        sub_rows = row.get(explode) or []
        outer_id = row.get(id_field, global_idx) if id_field else global_idx
        for j, sub in enumerate(sub_rows):
            merged = dict(row)
            if isinstance(sub, dict):
                merged.update(sub)
            else:
                merged["_explode_value"] = sub
            merged["_explode_idx"] = j
            merged["_outer_id"] = outer_id
            merged["_unique_id"] = args.id_template.format(id=outer_id, idx=j,
                                                           outer=outer_id)
            yield merged, global_idx
            global_idx += 1


def stream_convert(args, template_str: Optional[str], colmap: Dict[str, str]) -> Dict[str, Any]:
    """Single-pass streaming converter that writes local and/or HF outputs as it goes."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    do_local = args.mode in {"local", "both"}
    do_hf = args.mode in {"hf", "both"}

    media_dir: Optional[Path] = None
    local_rows: List[Dict[str, Any]] = []
    hf_ids: List[str] = []
    hf_media: List[List[Any]] = []
    hf_messages: List[str] = []
    # HF Image() fields accept dicts with bytes/path; using on-disk paths is far
    # faster than handing in PIL.Image objects (no in-memory re-encoding pass at
    # save_to_disk time). When --mode is hf-only, we stage images into a hidden
    # _hf_images/ subdir so the path approach still works.
    if do_local:
        media_dir = out / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
    elif do_hf:
        media_dir = out / "_hf_images"
        media_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    seen = 0
    t0 = time.time()
    last_log = t0

    def process_one(row: Dict[str, Any], i: int) -> Tuple[Optional[Dict[str, Any]],
                                                          Optional[Tuple[str, List[Any], str]]]:
        msg = build_message(row, colmap, args.answer_join, args.answer_list)
        media_raw = get_media_field(row, colmap)

        # Resolve local-JSON basenames against --media-dir
        if args.json and args.media_dir:
            resolved = []
            for m in media_raw:
                if isinstance(m, str) and not os.path.isabs(m) and not m.startswith("http"):
                    cand = os.path.join(args.media_dir, m)
                    if os.path.exists(cand):
                        m = cand
                resolved.append(m)
            media_raw = resolved

        local_entry: Optional[Dict[str, Any]] = None
        hf_entry: Optional[Tuple[str, List[Any], str]] = None
        stem = str(row.get(colmap["id"], i)).replace("/", "_") or f"row_{i}"

        # Always materialize media to disk once; both modes consume the same files.
        saved_basenames: List[str] = []
        for m_idx, m in enumerate(media_raw):
            saved = write_image_to_disk(m, media_dir, stem, m_idx,
                                         image_format=args.image_format,
                                         jpeg_quality=args.jpeg_quality)
            if saved is None:
                return None, None
            saved_basenames.append(saved)

        if do_local:
            local_msg = dict(msg)
            if template_str is not None:
                prompt = render_template(template_str, local_msg)
                if args.prompt_prefix and args.prompt_prefix not in prompt:
                    prompt = args.prompt_prefix + prompt
                local_msg["prompt"] = prompt
            local_entry = {"id": stem, "media": saved_basenames, "messages": [local_msg]}

        if do_hf:
            # Pass {bytes: <raw>} so save_to_disk doesn't trip the path-embed
            # codepath (which has a buggy null-mask construction in datasets
            # <=4.5 when many shards are present). Reading the encoded bytes
            # we just wrote is much cheaper than re-encoding PIL Images.
            hf_media_items = [{"path": None,
                               "bytes": (media_dir / b).read_bytes()}
                              for b in saved_basenames]
            hf_entry = (stem, hf_media_items, json.dumps([msg], ensure_ascii=False))
        return local_entry, hf_entry

    if args.workers and args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for row, i in iter_source(args):
                if args.limit is not None and i >= args.limit:
                    break
                futures.append(pool.submit(process_one, row, i))
                # Bound queue to avoid blowing memory
                if len(futures) >= args.workers * 4:
                    fut = futures.pop(0)
                    le, he = fut.result()
                    seen += 1
                    if le is None and he is None:
                        skipped += 1
                    else:
                        if le is not None:
                            local_rows.append(le)
                        if he is not None:
                            hf_ids.append(he[0]); hf_media.append(he[1]); hf_messages.append(he[2])
                    now = time.time()
                    if now - last_log > 5:
                        print(f"  …processed {seen} rows ({seen/(now-t0):.1f}/s)")
                        last_log = now
            for fut in futures:
                le, he = fut.result()
                seen += 1
                if le is None and he is None:
                    skipped += 1
                else:
                    if le is not None:
                        local_rows.append(le)
                    if he is not None:
                        hf_ids.append(he[0]); hf_media.append(he[1]); hf_messages.append(he[2])
    else:
        for row, i in iter_source(args):
            if args.limit is not None and i >= args.limit:
                break
            le, he = process_one(row, i)
            seen += 1
            if le is None and he is None:
                skipped += 1
            else:
                if le is not None:
                    local_rows.append(le)
                if he is not None:
                    hf_ids.append(he[0]); hf_media.append(he[1]); hf_messages.append(he[2])
            now = time.time()
            if now - last_log > 5:
                print(f"  …processed {seen} rows ({seen/(now-t0):.1f}/s)")
                last_log = now

    summary: Dict[str, Any] = {"mode": args.mode, "out": str(out),
                               "rows": seen - skipped, "skipped": skipped,
                               "elapsed_s": round(time.time() - t0, 2)}

    if do_local:
        local_rows.sort(key=lambda r: r["id"])
        data_path = out / "data.json"
        with open(data_path, "w") as f:
            json.dump(local_rows, f, ensure_ascii=False, indent=2)
        summary["local"] = {"data_path": str(data_path), "media_dir": str(media_dir),
                            "rows": len(local_rows)}
        print(f"[local] wrote {len(local_rows)} rows -> {data_path}")

    if do_hf:
        import shutil
        from datasets import Dataset, DatasetDict, Sequence, Image, Value, Features
        # Sort by id so output is deterministic
        order = sorted(range(len(hf_ids)), key=lambda i: hf_ids[i])
        hf_ids = [hf_ids[i] for i in order]
        hf_media = [hf_media[i] for i in order]
        hf_messages = [hf_messages[i] for i in order]

        features = Features({
            "id": Value("string"),
            "media": Sequence(Image()),
            "messages": Value("string"),
        })
        ds = Dataset.from_dict({"id": hf_ids, "media": hf_media, "messages": hf_messages},
                               features=features)
        dd = DatasetDict({args.split: ds})
        dd.save_to_disk(str(out / "hf_dataset"))

        meta_features = Features({
            "jinja_template": Value("string"),
            "version": Value("string"),
            "metadata": Value("string"),
        })
        meta_ds = Dataset.from_dict(
            {
                "jinja_template": [template_str or ""],
                "version": [args.version],
                "metadata": [json.dumps({"column_map": colmap,
                                         "source": (args.hf or args.json),
                                         "split": args.split}, ensure_ascii=False)],
            },
            features=meta_features,
        )
        meta_dd = DatasetDict({args.split: meta_ds})
        meta_dd.save_to_disk(str(out / "hf_metadata"))

        # If --mode hf only, the staging dir was hidden — leave it in place since
        # save_to_disk's Image() entries reference these absolute paths. Loading
        # from disk later still resolves them. The pusher embeds bytes on
        # push_to_hub, so the staging dir is no longer needed after pushing.
        summary["hf"] = {"hf_dataset": str(out / "hf_dataset"),
                         "hf_metadata": str(out / "hf_metadata"),
                         "rows": len(hf_ids)}
        print(f"[hf] wrote {len(hf_ids)} rows -> {out/'hf_dataset'} (+ {out/'hf_metadata'})")

    # v2 manifest: top-level metadata.json. push_to_hf.py uploads this to the
    # repo root and drops the legacy `metadata` config from the README.
    if args.task_type:
        manifest = build_v2_metadata(args, template_str, colmap)
        manifest_path = out / "metadata.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"[v2] wrote manifest -> {manifest_path}")
        summary["metadata_json"] = str(manifest_path)

    return summary


def build_v2_metadata(args, template_str: Optional[str], colmap: Dict[str, str]) -> Dict[str, Any]:
    """Build the v2 metadata.json from convert args + colmap + template.

    Schema: top-level {name, release_date, subsets:{<subset>:{...}}}. Inside the
    subset: language, modalities, task_type, prompt_template, mapping_from_source.
    """
    from datetime import datetime, timezone

    name = args.dataset_name or (args.hf.split("/")[-1] if args.hf else Path(args.out).name)
    release = args.release_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    canonical = {"id", "question", "answer", "image", "images", "media", "options",
                 "choices", "hint"}
    media_src = colmap.get("image") or colmap.get("images") or colmap.get("media")

    mapping: Dict[str, Any] = {}
    if media_src is not None:
        mapping["media"] = {"from": media_src, "type": "list",
                            "min_items": 1, "max_items": 1}
    if "id" in colmap:
        mapping["id"] = {"from": colmap["id"]}
    if "question" in colmap:
        mapping["question"] = {"from": colmap["question"]}
    if "answer" in colmap:
        mapping["answer"] = {"from": colmap["answer"], "optional": True}
    if "options" in colmap:
        mapping["options"] = {"from": colmap["options"], "optional": True}
    if "choices" in colmap:
        mapping["choices"] = {"from": colmap["choices"], "optional": True}
    if "hint" in colmap:
        mapping["hint"] = {"from": colmap["hint"], "optional": True}

    extras = {k: {"from": v} for k, v in colmap.items() if k not in canonical}
    if extras:
        mapping["extra"] = extras

    source_url = args.source_url or (
        f"https://huggingface.co/datasets/{args.hf}" if args.hf else None
    )
    if source_url:
        mapping["source"] = {
            "format": "json",
            "url": {args.split: source_url},
        }

    modalities = args.modalities or (
        ["single_image_start"] if (template_str and "<image>" in template_str)
        else ["text"]
    )

    subset = {
        "language": ["en"],
        "modalities": modalities,
        "task_type": args.task_type,
        "prompt_template": template_str or "",
        "mapping_from_source": mapping,
    }
    return {
        "name": name,
        "release_date": release,
        "subsets": {args.subset_name: subset},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf", help="HuggingFace dataset repo id")
    src.add_argument("--json", help="Local JSON file path (list of records)")
    p.add_argument("--split", default="val", help="HF split (default: val)")
    p.add_argument("--media-dir", help="Local media dir for --json source")
    p.add_argument("--map", nargs="+", required=True,
                   help="Column mapping: canonical=source pairs (id and question are required)")
    p.add_argument("--template", help="Jinja template string or file path")
    p.add_argument("--prompt-prefix", default="",
                   help="If template lacks an <image>/<video> placeholder, optionally prepend this")
    p.add_argument("--answer-join", default=" ",
                   help="Join string when source answer is a list (default: ' ')")
    p.add_argument("--answer-list", action="store_true",
                   help="Keep answer as a list (do not join)")
    p.add_argument("--version", default="v1", help="Version string for HF metadata config")
    p.add_argument("--mode", choices=["local", "hf", "both"], required=True,
                   help="Output mode")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Cap number of rows (for testing)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of worker threads for image encoding (default 8)")
    p.add_argument("--image-format", choices=["png", "jpeg"], default="png",
                   help="On-disk image format for --mode local (default png)")
    p.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality 1-100")
    p.add_argument("--explode", default=None,
                   help="Source list-valued field to explode into one row per element "
                        "(e.g. 'questions' for nested-MCQ datasets). The inner dict's "
                        "keys are merged onto each row; outer keys remain as fallback.")
    p.add_argument("--id-template", default="{id}_q{idx}",
                   help="Format string for post-explode ids (default '{id}_q{idx}'). "
                        "Available fields: {id} (outer), {outer}, {idx}.")
    # v2 manifest flags — when --task-type is set, convert.py also emits
    # <out>/metadata.json in the new top-level manifest format. push_to_hf.py
    # picks that up and uploads it to the repo root, replacing the legacy
    # `metadata` config.
    p.add_argument("--task-type", default=None,
                   help="v2 task_type (e.g. vqa, multiple_choice_vqa, captioning). "
                        "Required to emit metadata.json.")
    p.add_argument("--modalities", nargs="+", default=None,
                   help="v2 modalities list (default: ['single_image_start'] if template "
                        "contains <image>, else ['text']).")
    p.add_argument("--release-date", default=None,
                   help="v2 release_date YYYY-MM-DD (default: today UTC).")
    p.add_argument("--subset-name", default="main",
                   help="v2 subset key under `subsets:` (default: 'main').")
    p.add_argument("--source-url", default=None,
                   help="v2 source URL for this split (default: "
                        "https://huggingface.co/datasets/<--hf> if --hf is set).")
    p.add_argument("--dataset-name", default=None,
                   help="v2 top-level `name` field (default: last segment of --hf or "
                        "basename of --out).")
    args = p.parse_args()

    colmap = parse_map(args.map)
    args._original_id_field = colmap.get("id")
    if args.explode:
        # After explode, each row's id comes from the synthetic _unique_id field
        # built from the outer id + question index.
        colmap["id"] = "_unique_id"
    args._colmap = colmap
    template_str = load_template(args.template)

    print(f"Source: {'hf:' + args.hf if args.hf else 'json:' + args.json} (split={args.split})")
    print(f"Column map: {colmap}")
    print(f"Template: {template_str!r}")

    summary = stream_convert(args, template_str, colmap)
    summary_path = Path(args.out) / "convert_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['elapsed_s']}s — kept {summary['rows']} rows, skipped {summary['skipped']}.")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
