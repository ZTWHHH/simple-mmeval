---
name: dataset-upload
description: Convert any multimodal benchmark (HuggingFace dataset, local JSON, CSV/TSV with images) into Simple-MMEval runnable format and upload it to the HuggingFace Hub. Use this skill whenever the user wants to prepare/normalize/port a vision-language eval dataset for Simple-MMEval, run an existing benchmark through Simple-MMEval, build a `local@json` data file with an `img_dir`, build an `mmeval_hf@`-compatible HF dataset (canonical `media` / `messages` / `id` columns plus a top-level `metadata.json` v2 manifest with the prompt template + source mapping), or push a converted dataset to HuggingFace Hub. Trigger even if the user does not explicitly say "Simple-MMEval" — phrases like "convert this VQA dataset", "make this runnable in mmeval", "wrap this for evaluation", "turn this into an eval dataset", or "push this benchmark to HF for mmeval" all apply.
---

# dataset-upload

Convert a multimodal eval dataset into one of two runnable input formats accepted by [Simple-MMEval](https://github.com/mm-evaluation/simple-mmeval):

1. **`local`** — a single JSON file plus a directory of media files (`--dataset local@json --infile <data.json> --img_dir <media_dir>`).
2. **`hf`** — a HuggingFace `DatasetDict` for the `default` config plus a top-level `metadata.json` manifest at the repo root (`--dataset mmeval_hf@<user>/<repo>`; pass `--subset <name>` when the manifest packs multiple subsets). The legacy `metadata` config (a sibling DatasetDict) is no longer used; the v2 manifest replaces it.

The converter can emit one mode or **both at once in a single source pass** (`--mode both`), which is the recommended path when the user wants the local artifact for quick smoke runs *and* an HF artifact to push.

## Dependencies

```bash
pip install -r requirements.txt
# or, equivalently:
pip install datasets huggingface_hub jinja2 Pillow requests
```

If `push_to_hf.py` errors with `Mask must be a pyarrow.Array of type boolean`, upgrade `datasets` to the latest stable release first (this comes from Arrow/`embed_storage` edge cases in some versions). If it still reproduces on your shard layout, pin to a version your environment has verified — for example `pip install 'datasets<4.6'` — rather than assuming one pin fits all machines.

## Files in this skill

```
dataset-upload/
├── SKILL.md                   ← you are here
├── references/
│   ├── mmeval-format.md       ← exact target schema
│   ├── metadata-json.md       ← how to author metadata.json
│   └── jinja-templates.md     ← copy-paste templates
└── scripts/
    ├── inspect_source.py            ← peek source schema (HF or local JSON)
    ├── convert.py                   ← generic conversion (one split at a time)
    ├── convert_indexed_multimage.py ← MMMU-style: <image N> refs + per-subject configs
    ├── merge_splits.py              ← merge per-split HF artifacts into one DatasetDict
    ├── validate.py                  ← data-loader round-trip (no model)
    ├── smoke_run.py                 ← end-to-end smoke: tiny model + N rows through Simple-MMEval
    ├── push_to_hf.py                ← final HF push (token + repo)
    └── cleanup.py                   ← remove intermediate artifacts and HF cache after push
```

> The inspect script is named `inspect_source.py` rather than `inspect.py` because the latter shadows Python's stdlib `inspect` module when run from the same directory and breaks pandas import.

## Resource Management

Image benchmarks involve large data volumes. Follow these practices to avoid disk exhaustion and memory pressure throughout every conversion.

### Pre-flight: check available disk space and route temp/cache to project `.tmp`

Before starting any conversion, check free space on the partition that will hold the output and the HF download cache:

```bash
df -h /path/to/output   # check the output artifact partition
df -h /raid/ztw/simple-mmeval-skills-dev/.tmp  # (or whatever filesystem .tmp lives on)
```

A benchmark conversion needs roughly **source_size + output_size** of free space simultaneously. For image datasets, output ≈ source (images are re-compressed). As a rule: ensure at least `2 × estimated_dataset_size + 5 GB` of free space before starting.

`convert.py` automatically routes all HF download cache (`HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`) and system temp dirs (`TMPDIR`/`TEMP`/`TMP`) to a per-dataset subdirectory under `<project-root>/.tmp/` by default. This keeps all conversion-related I/O off `/tmp` and `~/.cache`:

```
<project-root>/.tmp/<out-basename>/
├── hf/          ← HF_HOME (all dataset parquet + model hub downloads)
│   ├── datasets/
│   └── hub/
└── tmp/         ← TMPDIR / TEMP / TMP
```

The converter prints the work-dir and available disk space at startup:
```
[work-dir] /raid/ztw/simple-mmeval-skills-dev/.tmp/my_dataset  (42.3/500.0 GB free/total)
```

Override the work-dir location or disable it entirely:
```bash
# Use a different base (e.g., a larger disk)
python3 scripts/convert.py ... --work-dir /mnt/large-disk/.tmp/my_dataset

# Disable (reverts to system defaults for HF cache / temp):
python3 scripts/convert.py ... --work-dir ''
```

### Datasets that require a config name: pre-download to JSON

`convert.py --hf` calls `load_dataset(repo, split=split)` **without a config name**. This works for repos whose only config is `default`, but fails for repos with named configs (e.g., `maritaca-ai/enem` with configs `2022/2023/2024`, `Hothan/OlympiadBench` with 18 configs, `ibm-granite/ChartNet`).

Detect these upfront:
```python
from datasets import get_dataset_config_names
configs = get_dataset_config_names("repo/name")
# If configs != ['default'], you must pre-download to JSON.
```

Pre-download pattern — **use base64 JPEG instead of saving image files to disk**. Separate image files accumulate fast and are equivalent in size to the JSON embed:

```python
import base64, io, json
from datasets import load_dataset
from pathlib import Path
from PIL import Image

def pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# Example: one config/split → one JSON
def download_split(repo: str, config: str, split: str, out_path: str) -> None:
    ds = load_dataset(repo, config, split=split)  # non-streaming for full dataset
    rows = []
    for i, row in enumerate(ds):
        img = row.get("image")
        rows.append({
            "id":       str(row.get("id", i)),
            "question": row.get("question", ""),
            "answer":   row.get("answer", ""),
            "image":    pil_to_b64(img) if img is not None else None,
            # add other fields as needed
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"  {len(rows)} rows → {out_path}")
```

`convert.py` reads the base64 string via `_value_to_pil` → `base64.b64decode`, so no special handling is needed in the conversion command.

**Mixed-modality rows** (some with images, some without): store `None` for absent images — **not an empty string** `""`. An empty string triggers `encode_failed` and silently drops the row:

```python
"image": pil_to_b64(img) if img is not None else None   # ✓ correct
"image": pil_to_b64(img) if img is not None else ""      # ✗ silently drops text-only rows
```

For mixed-modality datasets, also store a `has_image` integer field and use it in the template:
```jinja
{% if has_image %}<image>{% endif %}{{ question }}...
```

For datasets with variable image counts per row (0–N), store `n_images` and use `range()`:
```jinja
{% for i in range(n_images) %}<image>{% endfor %}{{ question }}...
```

Both `has_image` and `n_images` are stored in the HF message dict as integers and are available when the template is re-rendered at eval time.

### Very large datasets: use the HF rows API

For datasets where downloading the full parquet files is impractical (>10 GB), use the HuggingFace Datasets Server rows API to fetch metadata and image URLs without downloading the parquet files:

```python
import requests, json

TOKEN = "hf_..."
headers = {"Authorization": f"Bearer {TOKEN}"}

# 1. Check total rows
r = requests.get(
    "https://datasets-server.huggingface.co/rows"
    "?dataset=<repo>&config=<cfg>&split=<split>&offset=0&length=1",
    headers=headers, timeout=30,
)
total = r.json().get("num_rows_total")
print(f"Total rows: {total}")

# 2. Fetch in pages of 100, store image URL (convert.py fetches via HTTP)
rows = []
for offset in range(0, min(total, N_ROWS_NEEDED), 100):
    r = requests.get(
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset=<repo>&config=<cfg>&split=<split>&offset={offset}&length=100",
        headers=headers, timeout=60,
    )
    for item in r.json()["rows"]:
        row = item["row"]
        rows.append({
            "id":    row["id"],
            "question": ...,
            "image": row["image"]["src"],  # signed HTTPS URL – convert.py fetches it
        })
with open("out.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False)
```

The signed URLs expire within hours; run the conversion immediately after fetching.

**When to use this instead of streaming:**
- Dataset parquet files are >500 MB each (check: `HfFileSystem().ls("datasets/<repo>/<config>/", detail=True)`)
- Full download would exceed available disk space
- You only need a representative sample (e.g., 2000 rows from a 600k-row synthetic benchmark)

### Image format: default to JPEG

`convert.py` now defaults to `--image-format jpeg` (changed from PNG). JPEG at quality 92 is 5-10× smaller than PNG and is appropriate for all eval benchmarks except OCR benchmarks requiring pixel-perfect text rendering. Use `--image-format png` only when lossless fidelity is critical.

For pre-downloaded JSON preprocessing, always use JPEG at quality 85 (slightly lower than the converter's 92 to keep the JSON compact):
```python
img.save(buf, format="JPEG", quality=85)
```

### Process splits sequentially on disk-constrained systems

Never convert N splits in parallel when disk is tight. Each parallel conversion downloads N copies of the HF parquet files to cache and writes N output directories simultaneously. On a constrained partition, process one split at a time:

```bash
# Instead of parallel (&) with wait:
for split in 2022 2023 2024; do
    python3 scripts/convert.py --json preprocessed/enem_${split}.json ...
    # optionally inspect convert_summary.json here before continuing
done
```

After each split is converted and you've verified its `convert_summary.json`, you can free the preprocessed JSON immediately (the Arrow output is the durable artifact):
```bash
rm preprocessed/enem_${split}.json
```

## Workflow

### Step 1 — Inspect the source

```bash
python3 scripts/inspect_source.py --hf <repo_id> --split <split>
python3 scripts/inspect_source.py --json <data.json> --media-dir <dir>
```

Use the printed columns + types + sample row to decide the column mapping with the user.

> **Copy all identifiers verbatim from official sources.** Split names, config names, and column names are case-sensitive — using the wrong case silently produces wrong results or dropped rows. Before writing any `--map` or `metadata.json`: copy the HF repo ID directly from the URL bar (e.g., `Lin-Chen/MMStar`, not `lin-chen/mmstar`); take split and config names exactly as returned by `get_dataset_split_names` / `get_dataset_config_names`; use column names exactly as printed by `inspect_source.py` — never infer them from the dataset description or paper.

### Step 2 — Decide the column mapping and template

Ask the user (or infer) which source fields correspond to:

- `id` → unique id (required). Rows with `id == None` are skipped, not given a row-index fallback.
- `image` (or `images` / `media`) → media — single value or list (required for image/video benchmarks).
- `question` → required.
- `answer` → optional (often empty for test splits).
- `options` / `choices` / `hint` → only for MCQ-style benchmarks.

Any extra `--map foo=bar` is passed through to the message dict as `foo`. Pass-through values that are `None` are dropped (so the Jinja template never sees the literal string `"None"`); only schema fields (`question / answer / hint / options / choices`) keep their empty representations (`""` / `{}` / `[]`).

If `--explode <field>` is given, each row's id is replaced with `--id-template '{id}_q{idx}'.format(id=outer, idx=j)`. Rows whose outer id is `None` skip — the converter does not invent an `outer_{global_idx}` to keep them.

Source `options` given as a list (`["cat", "dog", ...]`) are normalized to `{"A": "cat", "B": "dog", ...}` — a documented label transform, so the dict-options MCQ template in `references/jinja-templates.md` works for both shapes.

Then pick a Jinja template. A good default for single-image VQA:

```jinja
<image>{{ question }}
Answer the question using a single word or phrase.
```

Match the prompt style of the original benchmark (terse for VizWiz/OK-VQA; "Please select the correct option" for MMBench). See `references/jinja-templates.md` for ready-made templates per benchmark style.

### Step 3 — Convert

`scripts/convert.py` reads the source, applies the column map and template, and writes the artifact(s).

**Source flags** (pick one):
- `--hf <repo_id> --split <split>` (HF **requires** an explicit split name)
- `--json <path> [--media-dir <dir>]` | `--tsv <path>` | `--csv <path>` (local splits optional; omit `--split` for a flat HF `Dataset` on disk / Hub split `train` after push)

**Mapping flags:**
- `--map id=<src> question=<src> image=<src> answer=<src> ...` — space-separated `canonical=source` pairs (omit when using `--metadata-json` unless you want overrides).
- With `--metadata-json`, `--map` is **override-only**: extra pairs like `category=cat` merge onto the seed mapping and **do not** need to repeat `id` / `question` (those come from the template).
- `--metadata-json <path>` — load `mapping_from_source` + default `prompt_template` from a draft `metadata.json` (see `references/metadata-json.md`).
- `--name`, `--subset`, `--language`, `--task-type`, `--modalities`, `--release-date`, `--source-url`, `--source-format` — control emitted `metadata.json`.
- `--template <path-or-string>` — Jinja template (string or file path).
- `--prompt-prefix "<image>"` — auto-prepend the placeholder if the template doesn't include it.
- `--answer-list` — keep `answer` as a list when the source provides a list (default joins with space).
- `--answer-join " "` — join string when reducing list answers to a single string.

**Output flags:**
- `--mode local | hf | both`
- `--out <dir>` — output directory.
- `--workers N` — parallelism for image encoding (default 8; 16 is a good fit on multi-core hosts).
- `--image-format jpeg | png` — JPEG (default, 5-10× smaller) or PNG (lossless). Use PNG only when pixel-perfect fidelity is required.
- `--jpeg-quality N` — JPEG quality 1–100 (default 92).
- `--work-dir <path>` — per-dataset directory for HF caches and temp files (default: `<project-root>/.tmp/<out-basename>`). Override when `.tmp` is on a full partition; pass `''` to disable.
- `--hf-cache-dir <path>` — redirect only `HF_DATASETS_CACHE` (narrower than `--work-dir`; prefer `--work-dir` for full redirection).
- `--limit N` — cap rows for testing. With `--hf` this also flips `load_dataset(streaming=True)` so you don't download the whole benchmark.
- `--explode <field>` — flatten a list-valued field (e.g. `--explode questions` for nested-MCQ datasets like CaptionQA). Each element becomes its own row, with the inner dict's keys merged onto the outer row, and the id auto-suffixed via `--id-template '{id}_q{idx}'` (default).
- `--id-template '{id}_q{idx}'` — id format string for exploded rows.
- `--save-num-proc N` — `num_proc` forwarded to `DatasetDict.save_to_disk` for `--mode hf` / `both` (default 1).
- `--json-indent N` — pretty-print `data.json` (default: most compact JSON via `separators=(",", ":")`; pass `2` for human-readable).

For datasets with **multiple domain splits** that should land in one HF repo (e.g. CaptionQA with `natural`/`document`/`ecommerce`/`embodiedai`), run `convert.py --mode hf` once per split into separate output dirs, then combine them:

```bash
python3 scripts/merge_splits.py \
    --inputs <out>/natural <out>/document <out>/ecommerce <out>/embodiedai \
    --out <out>/merged
```

The merged directory has one `hf_dataset/` and one merged `metadata.json` ready for `push_to_hf.py`.

**Output layout:**

```
<out>/
├── data.json               # local mode (compact JSON; --json-indent 2 to format)
├── media/                  # local mode (also reused for HF bytes)
├── hf_dataset/             # HF mode — DatasetDict for the `default` config
│                           #   (named split, or flat Dataset when --split is omitted)
├── metadata.json           # always written — v2 manifest, uploaded to repo root by push_to_hf.py
└── convert_summary.json    # rows kept/skipped + skip_reason_counts + first 50 skips
```

`convert_summary.json` records `skip_reason_counts` (`missing_required:id`, `encode_failed`, `unknown_video_ext`) and a sample of the first 50 skipped rows. Inspect it before declaring a conversion done — silently dropping 30% of a benchmark is the most common failure mode.

**Optional `metadata.json` overrides** (defaults are usually fine; see `references/metadata-json.md` for the full schema):

- `--task-type <type>` — default: inferred from the column mapping (`multiple_choice_vqa` when `options`/`choices` is mapped, otherwise `vqa`). Supported: `vqa`, `multiple_choice_vqa`, `captioning`.
- `--modalities <m> [...]` — default: inferred from the observed media counts (`single_image_start`, `multi_image_*`, `text`, etc.). Override only when the inference is wrong.
- `--release-date YYYY-MM-DD` — default: today (UTC).
- `--subset <name>` — default `main`. Use distinct names when packing multiple sub-benchmarks into one repo.
- `--source-url <url>` — default: `https://huggingface.co/datasets/<--hf>` for HF sources; otherwise omitted.
- `--name <name>` — top-level `name`. Default: `--hf` repo or basename of `--out`.
- `--language <lang> [...]` — default `["en"]` (or the seed value when `--metadata-json` is used).

The `mapping_from_source` block is auto-built from `--map`: canonical keys (`id`/`question`/`answer`/`image`/`options`/...) land at the top; everything else lands under `extra:`. The `media.min_items`/`max_items` are filled in from the actual per-row counts observed during conversion.

### Step 4 — Validate splits against the source

**Before finalizing any conversion, confirm that the converted dataset's splits exactly match the source dataset's splits** — same names, same row counts. Mismatches mean samples are missing or were double-counted.

For HuggingFace sources:

```python
from datasets import get_dataset_split_names, load_dataset

# Check all configs the source exposes
from datasets import get_dataset_config_names
configs = get_dataset_config_names("<source_repo>")
# For each config, check its splits and row counts
for cfg in configs:
    for split in get_dataset_split_names("<source_repo>", cfg):
        ds = load_dataset("<source_repo>", cfg, split=split)
        print(f"config={cfg!r} split={split!r}: {len(ds)} rows")
```

For GitHub / local sources, count rows in the source JSON manually and compare:

```python
import json
with open("raw/data.json") as f:
    data = json.load(f)
print(f"source rows: {len(data)}")
# Compare against convert_summary.json
with open("<out>/convert_summary.json") as f:
    summary = json.load(f)
print(f"converted rows: {summary['rows']}, skipped: {summary['skipped']}")
```

Then check the converted artifact for each expected split:

```python
from datasets import load_from_disk
dd = load_from_disk("<out>/hf_dataset")
for split_name, ds in dd.items():
    print(f"converted split={split_name!r}: {len(ds)} rows")
```

**Checklist before proceeding:**
- [ ] Every split from the source exists in the converted dataset (no missing splits).
- [ ] Row counts match for every split (skipped rows in `convert_summary.json` are justified — only rows with `id=None`, broken images, or unknown video extensions should be dropped).
- [ ] Split names match the source exactly (e.g., `testmini` not `test`).
- [ ] For HF sources with multiple configs (e.g., `testmini` + `testmini_text_only`), all configs are represented — either as separate HF splits in one artifact or documented as intentionally excluded.
- [ ] Source field names in `metadata.json mapping_from_source` (and in every `--map` invocation) match the actual column names from `inspect_source.py` exactly — same spelling and same case.
- [ ] `metadata.json name` is the official benchmark name as it appears in the paper or dataset card — not the HF repo slug (e.g., `"MMStar"` not `"Lin-Chen/MMStar"`).
- [ ] `metadata.json modalities` uses only the supported taxonomy: `single_image_start`, `single_video_start`, `multi_image_start`, `multi_image_interleave`, `multi_video_interleave`, `multi_image_video_interleave`, `text`. No legacy values (`"image"`, `"multi_image"`, etc.).
- [ ] `metadata.json task_type` is one of: `vqa` (free-form answer), `multiple_choice_vqa` (discrete option selection), `captioning` (free-form caption with no explicit question). No ad-hoc values.
- [ ] `metadata.json source` contains **only** `format` and `url`. No `repo` field, no `links` field. `url` is a `{split_name: URL, …}` dict — one key per uploaded split covered by that subset.
- [ ] `metadata.json` contains **no local machine paths**. The `source.path` key sometimes records a pre-downloaded JSON path during authoring — always remove it before pushing.
- [ ] For multi-subset datasets, each subset's `source.url` keys reference **only that subset's** uploaded split names — not a sibling subset's keys. Copy-paste errors across subsets are easy to miss.
- [ ] `choices` mapping is present only when the source actually provides a choices column. Do not add it for benchmarks where options are embedded in the question text or extracted by the converter.
- [ ] `validate.py --audit <out>/hf_dataset` passes with no issues (covers placeholder/media count, `<image>` inside question text, schema fields, no local paths).

If a source has both image and text-only configs for the same benchmark, convert each config separately (with appropriate templates), then merge the splits into one artifact:

```bash
# Convert image config → out_img/ (template with <image>)
python3 scripts/convert.py --hf <repo> --split <img_split> --template '<image>{{ question }}' …

# Convert text-only config → out_text/ (no image, different subset key)
python3 scripts/convert.py --hf <repo> --split <text_split> --subset text_only \
    --template '{{ question }}' --modalities text …

# Merge DatasetDicts and combine metadata.json subsets manually
python3 - << 'PYEOF'
import json
from datasets import DatasetDict, load_from_disk
from pathlib import Path

out_img = Path("out_img")
out_text = Path("out_text")
merged = Path("out_merged")
merged.mkdir(parents=True, exist_ok=True)

dd = DatasetDict({
    **load_from_disk(str(out_img / "hf_dataset")),
    **load_from_disk(str(out_text / "hf_dataset")),
})
dd.save_to_disk(str(merged / "hf_dataset"))

with open(out_img / "metadata.json") as f: meta = json.load(f)
with open(out_text / "metadata.json") as f: text_meta = json.load(f)
meta["subsets"].update(text_meta["subsets"])
with open(merged / "metadata.json", "w") as f: json.dump(meta, f, indent=2)
PYEOF
```

### Step 5 — Verify prompts and media

`scripts/validate.py` has two modes:

**Standalone audit (no Simple-MMEval install needed) — run this first:**
```bash
python3 scripts/validate.py --audit <out>/hf_dataset
# or with an explicit metadata.json path:
python3 scripts/validate.py --audit <out>/hf_dataset --metadata <out>/metadata.json
```
Checks: (1) `metadata.json` has no leaked local paths, (2) `source.url` keys match actual split names and no deprecated `repo`/`links` fields are present, (3) modalities/task_type use the supported taxonomy, (4) every row's `<image>`/`<video>` placeholder count equals its media count, (5) `<image>` inside question text when the template already supplies placeholders.

**Full round-trip (requires Simple-MMEval checkout):**
```bash
python3 scripts/validate.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --local <out> \
    --hf <out> \
    -n 3
```
Instantiates the actual `LocalJSONDataset` / `MMEvalHFDataset` data loaders, prints rendered prompts, and confirms images load as PIL with the right dimensions.

Run the standalone audit before the full round-trip; both should pass before declaring a conversion done.

### Step 5b — Pre-push smoke (local mode, 50 random rows)

`validate.py` only proves the artifact loads. It does NOT prove the chat
template + processor accept the rendered prompt + media count, that
inference produces output, or that the result file is well-formed. The
classic silent failure mode is: `validate.py` is green but the processor
sees a different `<image>` count than the row's media list at run time.

`scripts/smoke_run.py --local` closes that gap before you push. It
points `mmeval/run.py` at the full local artifact (using mmeval's own
`--sample_num 50 --sample_order random --sample_seed 42` sampler) with
`Qwen3-VL-2B-Instruct` (the unified test model used throughout this
skill), and gates the smoke on **every** sampled row passing the
fidelity checklist below.

```bash
python3 scripts/smoke_run.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --local <out> \
    --python /raid/miniconda3/envs/qwenvl/bin/python
```

GPU selection is automatic by default — see the GPU notes in Step 6b.
Local-mode smoke runs in a few minutes on a freed H200 once the model
is cached. It is the pre-push gate; do not push until it is green.

**Required flags & defaults (each fixes a real failure mode seen in the field):**

- `--python <bin>` (default `python3`) — orchestrator interpreter for
  `mmeval/run.py`, which imports `torch` at top-level. The default `python3`
  on the box often isn't a venv/conda env that has torch installed; point
  this at a conda env with `torch` + `datasets` (e.g.
  `/raid/miniconda3/envs/qwenvl/bin/python`). Per-model inference still
  dispatches to the env in `mmeval/registry.py` — this flag is only the
  orchestrator interpreter.
- `--attn-implementation` (default `sdpa`) — passed through to
  `mmeval/run.py`. The default sidesteps a common silent failure: if the
  per-model conda env has a broken `flash_attn` install (torch ABI
  mismatch), `transformers` falls into an interactive `input()` prompt
  asking whether to fetch a remote flash-attn kernel. With no stdin
  attached the prompt hits EOF, every sample is retried-then-skipped, and
  `result.json` ends up empty. `sdpa` always works.
- `--rows-per-split N` (default `50`, capped at the split size so smaller
  splits run end-to-end) — random samples per (subset, split). mmeval
  does the sampling internally via `--sample_num/--sample_order random`;
  no derived artifact, no row-selection shim. Every sampled row must
  pass the fidelity checks below for the smoke to be considered green.
- `--seed <int>` (default `42`) — forwarded to `--sample_seed` so smoke
  runs are reproducible across machines/sessions.
- `--gpu <id>` — explicit `CUDA_VISIBLE_DEVICES`. Default: auto-pick the
  least-utilized GPU with at least `--min-free-mb` (8 GiB) of free VRAM.
  Pass `--gpu -1` to disable pinning.

**Implementation details (handled automatically):**

1. **Passthrough template (local mode only).** `LocalJSONDataset` only
   uses the saved `messages[0].prompt` when no template override is in
   play; if anything else loads the default template (`{{ question }}`),
   `<image>` placeholders and any post-prompt suffix get stripped and
   the model answers "I'm a text-based AI". `smoke_run.py --local`
   writes a passthrough `{{ prompt }}` template into the work dir and
   passes it via `--template`, which forces the loader to re-emit the
   exact rendered prompt the converter wrote. If you ever invoke
   `mmeval/run.py` against a `local@json` artifact by hand, pass
   `--template <path-to-jinja>` with the same template you handed to
   `convert.py`. HF mode does not need this — `MMEvalHFDataset` reads
   `prompt_template` from `metadata.json` and renders correctly without
   an override.
2. **Response shape.** `mmeval`'s `res_handler.py` writes the model output
   into `result[i].messages[1].response` (the assistant message). It is
   stored as `str`, `list[str]`, or `list[list[str]]` depending on the
   inference backend; `smoke_run.py` flattens before checking emptiness.
   Tolerating a legacy top-level `r["response"]` is kept as a fallback.
3. **Placeholder/media alignment.** `res_handler.py` strips per-message
   `media` before saving, but preserves the sample-level `media` list
   (as `"Image Object"` strings, one per image). The fidelity check
   counts `<image>`/`<video>` placeholders in `messages[0].prompt` and
   compares to `len(media)` — a mismatch here is the most reliable
   signal that the prompt and media list desynced.

**Per-row fidelity checks (must pass for every sampled row in every (subset, split)):**

- `id` is present, non-empty, and unique within `result.json`.
- `messages[0].question` (when present) is a string — mapping-from-source
  drift that overwrites question with a non-string fails here.
- `messages[0].prompt` is a non-empty string (proves the template rendered).
- `<image>`/`<video>` placeholder count in the rendered prompt equals
  the sample-level `media` count.
- When present: `options` is a dict, `choices` is a list, `hint` is a
  string, `answer` is a string or list.
- `messages[1].role == "assistant"` and `response` is non-empty after
  flattening str / list[str] / list[list[str]].
- `result.json` row count equals `min(--rows-per-split, split_size)` —
  a partial result file (silent post-load drop) fails the smoke.

A dataset passes the smoke only when **every** sampled row in **every**
(subset, split) clears every check above; any failure exits non-zero and
leaves the artifacts under `<simple-mmeval>/.tmp/smoke_<dataset-slug>/`
for follow-up.

### Step 6 — Push to HuggingFace (for `hf` mode)

`scripts/push_to_hf.py` is the **final user-facing push script**. It only needs the artifact dir + an HF token + a target repo name:

```bash
python3 scripts/push_to_hf.py \
    --artifact-dir <out> \
    --repo-id <user>/<repo> \
    --token <hf_token> \
    [--private] \
    [--cleanup-artifact]   # delete <out> after a successful push
```

It pushes the `default` config and uploads `metadata.json` to the repo root, then prints the exact `simple-mmeval` invocation for the user to copy.

Add `--cleanup-artifact` on disk-constrained systems to delete the local Arrow artifact immediately after a successful push (the Hub becomes the durable copy).

> **Do not push until the split-count checklist in Step 4 passes.** A missing split cannot be patched post-push without a full re-push of the affected splits.

### Step 6b — Post-push framework smoke (HF mode, 50 random rows / (subset, split))

After the push lands, run `scripts/smoke_run.py --hf` to exercise the
**actual** uploaded artifact through `MMEvalHFDataset` — the same
dataloader that real evaluations use. This is the authoritative gate
between "push reported success" and "the dataset is safe to evaluate on":
it catches `metadata.json` upload bugs, prompt-template regressions, and
Arrow round-trip issues that pre-push local smoke cannot see.

```bash
python3 scripts/smoke_run.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --hf <user>/<repo> \
    --python /raid/miniconda3/envs/qwenvl/bin/python
# add --subset <name> to scope to one subset, e.g. --subset main
```

What it does, in order:

1. Picks the **least-utilized GPU with ≥ `--min-free-mb` (8 GiB)** of
   free VRAM via `nvidia-smi --query-gpu=index,memory.free,utilization.gpu`
   and binds the child via `CUDA_VISIBLE_DEVICES`. Override with
   `--gpu N`; pass `--gpu -1` to disable pinning.
2. Downloads `metadata.json` from the Hub, enumerates every subset, and
   calls `datasets.get_dataset_split_names(repo, "default")` to discover
   every split that will land in front of evaluators.
3. For **every** `(subset, split)`: spawns `mmeval/run.py
   --dataset mmeval_hf@<repo> --subset <subset> --split <split>
   --sample_num 50 --sample_order random --sample_seed 42` (or fewer
   rows when the split has < 50, in which case mmeval caps at the split
   size and every row runs). This uses mmeval's own sampler — no derived
   artifact, no row-selection shim — so what gets exercised is exactly
   what a real evaluation would see.
4. After each run, parses `result.json` and requires that **every** row
   passes the same per-row fidelity checks as Step 5b (id present and
   unique, question/options/choices/hint/answer types, non-empty
   rendered prompt, placeholder count == media count, assistant
   response non-empty), AND that the result row count equals
   `min(--rows-per-split, split_size)`. Anything less fails the smoke.
5. Writes everything to a persistent location that **never gets cleaned
   up by default** (see Step 7):

   ```
   <simple-mmeval>/.tmp/smoke_<repo-slug>/
   ├── smoke_metadata.json   # repo, subsets, splits, model, seed
   ├── smoke_summary.json    # per-(subset,split) rc + rows + violations
   └── <subset>/<split>/
       ├── result.json       # rendered prompts + responses
       └── run.log           # full mmeval/run.py stdout/stderr
   ```

Defaults (`Qwen3-VL-2B-Instruct`, sdpa, seed=42, 50 rows per
(subset, split), auto-GPU) are all chosen so a run is reproducible
across machines and sessions; pin them explicitly only when you have a
specific reason.

**Pass/fail discipline.** A dataset is smoke-test passed only when
**every** `(subset, split)` succeeds and **every** sampled row in
**every** result.json clears every fidelity check. A non-zero exit
means at least one pair either crashed mmeval/run.py (dataloader,
prompt-rendering, media-loading, or runtime error) or produced rows
that violated the structural checks. The `.tmp/smoke_<repo-slug>/`
tree is preserved; inspect `run.log` (raw error) and `result.json`
(the rendered prompt the model actually saw) before declaring the
converted dataset healthy. If a failure points at a converter bug, fix
`convert.py` (or the relevant template in `references/`), re-run the
conversion + push, and re-smoke before signing off.

### Step 7 — Clean up intermediate artifacts

After `push_to_hf.py` exits 0 and you've verified the Hub repo, free disk space with `cleanup.py`. Always confirm the push succeeded first.

```bash
# Quick: delete merged artifact after push (built into push_to_hf.py)
python3 scripts/push_to_hf.py \
    --artifact-dir <merged_dir> --repo-id <user>/<repo> --token "$HF_TOKEN" \
    --cleanup-artifact

# Full: explicit control with --dry-run preview
python3 scripts/cleanup.py \
    --artifact-splits <split1_dir> <split2_dir> ... \
    --merged-artifact <merged_dir> \
    --preprocessed <preprocess.json> \
    --work-dir .tmp/<dataset>       \  # removes entire per-dataset work dir (HF cache + temp)
    --dry-run   # preview first, then re-run without --dry-run
```

| What | Safe to delete when |
|------|---------------------|
| Per-split `out/<split>/` dirs | After `merge_splits.py` is verified |
| Merged `out/merged/` dir | After `push_to_hf.py` exits 0 |
| Preprocessed `*.json` files | After `convert.py` writes `hf_dataset/` |
| `.tmp/<dataset>/` work dir | After push succeeds (contains HF cache + temp files) |
| `media/` dir (local mode) | After push succeeds |
| `<simple-mmeval>/.tmp/smoke_<dataset-slug>/` | **Never by default** — Step 5b (local) and Step 6b (HF) smoke artifacts are preserved for human inspection. `cleanup.py` skips any `smoke_*` directory under a work-dir unless `--include-smoke-results` is passed. |

**Never delete** the merged artifact before a successful push.

**Repo layout after push (v2):**

```
<user>/<repo>/
├── data/                   # default config parquet shards
├── metadata.json           # v2 manifest (top-level)
├── README.md               # auto-generated (default config only)
└── .gitattributes
```

See `references/metadata-json.md` for the full v2 manifest schema (top-level `name`/`release_date`/`subsets[name]` with `language`/`modalities`/`task_type`/`prompt_template`/`mapping_from_source`).

## Worked example — CaptionQA (nested-MCQ, 4 domain splits)

```bash
SKILL=<skill-path>
ROOT=<workdir>/captionqa

# 1. Convert each split.
# Run in parallel (&) only if disk has enough free space for N simultaneous
# HF cache downloads; on constrained partitions process sequentially (remove &).
for split in natural document ecommerce embodiedai; do
  python3 -u $SKILL/scripts/convert.py \
    --hf Borise/CaptionQA --split $split \
    --map id=id image=images question=question options=choices answer=answer category=category \
    --explode questions \
    --template '<image>
You are given an image and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation.

Question:
{{ question }}

Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}

Answer:' \
    --mode hf --out $ROOT/$split --workers 16 &
done; wait

# 2. Merge into a single multi-split artifact
python3 $SKILL/scripts/merge_splits.py \
    --inputs $ROOT/natural $ROOT/document $ROOT/ecommerce $ROOT/embodiedai \
    --out $ROOT/merged

# 3. Push (add --cleanup-artifact to delete the merged dir after push)
python3 $SKILL/scripts/push_to_hf.py \
    --artifact-dir $ROOT/merged --repo-id <user>/CaptionQA --token "$HF_TOKEN" \
    --cleanup-artifact

# 4. Clean up per-split dirs and HF cache
python3 $SKILL/scripts/cleanup.py \
    --artifact-splits $ROOT/natural $ROOT/document $ROOT/ecommerce $ROOT/embodiedai \
    --hf-cache Borise/CaptionQA
```

The image-mode prompt is the direct visual analogue of the official `qa.py:build_caption_qa_prompt` from [bronyayang/CaptionQA](https://github.com/bronyayang/CaptionQA): `Caption:\n{caption}` is replaced by `<image>` and the system instruction adapted to "given an image" instead of "given a caption". Choices are *not* shuffled and the "Cannot answer from the caption" option is *not* added — those are caption-mode artifacts.

## Worked example — VizWiz-VQA val

Single command produces both artifacts in ~60 seconds for 4319 rows:

```bash
python3 scripts/convert.py \
    --hf lmms-lab/VizWiz-VQA --split val \
    --map id=question_id question=question image=image answer=answers category=category \
    --template '<image>{{ question }}
Answer the question using a single word or phrase.' \
    --answer-list \
    --mode both \
    --out <workdir>/vizwiz_val \
    --workers 16
```

Then verify both, then smoke-run:

```bash
python3 scripts/validate.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local <workdir>/vizwiz_val \
    --hf <workdir>/vizwiz_val \
    -n 3

python3 scripts/smoke_run.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local <workdir>/vizwiz_val \
    --python /raid/miniconda3/envs/qwenvl/bin/python
```

Run via Simple-MMEval (local artifact, no HF push needed):

```bash
cd <simple-mmeval-checkout>
export PYTHONPATH=./:$PYTHONPATH
python3 mmeval/run.py \
    --model_name_or_path Qwen3-VL-2B-Instruct \
    --dataset local@json \
    --infile <workdir>/vizwiz_val/data.json \
    --img_dir <workdir>/vizwiz_val/media \
    --out_dir work_dirs/vizwiz_val \
    --gpu_per_parallel 1 --parallel_per_task 1
```

Push HF artifact to the user's Hub:

```bash
python3 scripts/push_to_hf.py \
    --artifact-dir <workdir>/vizwiz_val \
    --repo-id <user>/VizWiz-VQA \
    --token "$HF_TOKEN" \
    --cleanup-artifact   # optional: free local disk after push

# If not using --cleanup-artifact, clean up manually afterward:
python3 scripts/cleanup.py \
    --merged-artifact <workdir>/vizwiz_val \
    --hf-cache lmms-lab/VizWiz-VQA
```

## Worked example — MMMU / MMMU-Pro (indexed multi-image MCQ + multi-config)

MMMU-style benchmarks are the **most error-prone shape this skill supports**.
They have three quirks that the generic `convert.py` doesn't handle directly:
indexed `<image N>` references, image references inside option strings, and
multiple subject configs that should land in one repo. Use a custom
preprocessor and feed the result into the HF artifact directly.

The full converter for MMMU and MMMU-Pro lives in this repo at
`scripts/convert_indexed_multimage.py` (template-driven, no baked prompt).
The key per-row logic is:

```python
IMG_REF_RE = re.compile(r"<image (\d+)>")

def process_row(row):
    options = ast.literal_eval(row["options"]) if row["options"] else []
    question = row["question"] or ""

    # 1) Collect refs from question AND every option string, in textual order.
    refs = [int(m) for m in IMG_REF_RE.findall(question)]
    for o in options:
        refs.extend(int(m) for m in IMG_REF_RE.findall(o))

    # 2) Build media list: one entry per reference (the same image may repeat).
    images = [row.get(f"image_{i}") for i in range(1, 8)]
    media = [images[r - 1] for r in refs]   # 1-indexed in the source

    # 3) Normalize <image N> → <image> in question + options BEFORE storing.
    msg_question = IMG_REF_RE.sub("<image>", question)
    msg_options = {chr(ord("A") + i): IMG_REF_RE.sub("<image>", v)
                   for i, v in enumerate(options)}

    # 4) Sanity: total <image> tokens the template will emit must equal media count.
    assert msg_question.count("<image>") + sum(v.count("<image>") for v in msg_options.values()) == len(media)
```

The Jinja template for MMMU mirrors the official paper / eval-code prompt
[byte-for-byte](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/data_utils.py):

```jinja
{{ question }}{% if options %}

{% for k, v in options.items() %}({{ k }}) {{ v }}
{% endfor %}
Answer with the option's letter from the given choices directly.{% else %}

Answer the question using a single word or phrase.{% endif %}
```

Verify it matches the upstream `construct_prompt` output verbatim:

```python
got = env.from_string(TEMPLATE).render(question=q, options=opts)
exp = upstream_construct_prompt(row)
assert got == exp   # byte-for-byte
```

For MMMU-Pro the prompt format is *different* (`A. opt` not `(A) opt`,
single newlines, "option letter" without apostrophe-s) — see
[`mmmu-pro/infer/infer_gpt.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/infer/infer_gpt.py).
Always grep the upstream eval code; do not assume two related benchmarks
share a prompt.

## Common pitfalls (learned the hard way)

- **HF repo name must match the official dataset name exactly — no appended suffixes.** Use the benchmark's official name (e.g., `MMStar`, `ScienceQA`, `VQA-RAD`) as the HF repo name, including exact capitalization, spelling, and punctuation. Never append `-mmeval` or any other suffix; the `mm-eval/` org prefix already namespaces the dataset unambiguously.
- **Don't trust intermediate "convenience" reformats — re-source from the official upstream.** Third-party consolidations of a benchmark sometimes pre-render a `prompt` field whose token count silently disagrees with the unique-image count, or strip metadata (subject, difficulty, explanation) you'd want later. When the upstream is available, source from it.
- **Case-sensitive identifiers — silent wrong results or dropped rows.** HF split names, config names, and dataset column names are all case-sensitive: `val` ≠ `Val`, `question_id` ≠ `Question_ID`. Always take split and config names from `get_dataset_split_names` / `get_dataset_config_names`, and column names from `inspect_source.py` output — never from descriptions, papers, or memory.
- **Missing splits — the silent but critical failure.** HF datasets often expose multiple splits (`test`, `testmini`) or multiple configs (`testmini`, `testmini_text_only`). Converting only one and pushing is wrong even when row counts look right. Always enumerate all configs and splits with `get_dataset_config_names` + `get_dataset_split_names` before starting, and verify against the converted artifact at Step 4.
- **Config name ≠ split name.** For HF datasets that require a config (`load_dataset(repo, config_name, split=...)`), `convert.py --hf` does not support passing a config name. Pre-download such datasets to local JSON first, then use `--json`. See the "Datasets that require a config name" section above for the base64-JPEG pre-download pattern.
- **Multi-config benchmarks → one split per config.** When a source has multiple HF configs that should evaluate as a single benchmark (MMMU-Pro: `standard (4 options)` + `standard (10 options)` + `vision`; MMMU: 30 subjects), run the converter per config into separate output dirs and `merge_splits.py` them. Pick split names without spaces or parens (`standard_4_options`, not `standard (4 options)`) — HF Hub config/split names disallow special chars.
- **Source the prompt template from the published paper and the official eval code — in that order — and cross-check both.** The arxiv / project page often quotes the prompt verbatim in a figure or appendix; the eval code carries the actual format string used to produce reported scores. They should match; when they disagree (rare but it happens) the eval-code version is what produced the reported numbers, so it wins. Cite the file path / line numbers / paper section in the `metadata.metadata` JSON so a future reader can audit the choice.
- **Confirm the chosen template with the user before running the full conversion.** Show the rendered prompt for one MC and one short-answer row, name the source you took it from (paper §X / file L#–L#), and wait for a thumbs-up. Re-running a conversion + re-pushing a multi-GB HF artifact because the prompt was wrong costs minutes you didn't need to spend; a 5-line preview is cheap.
- **Verify your template byte-for-byte against the upstream eval code.** When the source has an official eval script, render your template with a sample row and assert string-equality with `upstream.construct_prompt(row)`. Tiny discrepancies (extra blank line, `(A)` vs `A.`, `option's letter` vs `option letter`, missing trailing instruction) silently change scores — the assert catches them in 5 seconds.
- **Prefer a Jinja template in `metadata` over a pre-rendered `prompt` on the message.** Per the format spec, when the metadata template is non-empty Simple-MMEval *always* re-renders it (the per-row `prompt` becomes a fallback only when rendering throws). A template lets a downstream user fork the repo and change the prompt without re-running the converter. Bake `prompt` only when the rendering is genuinely per-row dynamic in a way Jinja can't express.
- **Multi-image rows.** Use `--map images=<list-field>` and ensure the template emits the right number of `<image>` placeholders. For indexed `<image N>` patterns (MMMU-style, note the space: `<image 1>`) see the worked example above — that shape needs custom preprocessing that replaces each `<image N>` ref with `<image>` and builds a multi-item media list.
- **`<imageN>` tokens (no space) are text labels, not media placeholders.** Some datasets (e.g., MathVision) use `<image1>`, `<image2>`, … (no space before the digit) as textual labels for subfigures *within* a single composite image — each row has exactly **one** media file. These are distinct from MMMU's `<image N>` (space before digit) which reference separate images. Preserve `<imageN>` verbatim in the question text; the template prefix (`<image>{{ question }}\n`) supplies the single real placeholder. Never replace `<image1>` with `<image>` in the question field. The `validate.py --audit` check catches `<image>` embedded in question text as a structural error.
- **Text-only configs in image benchmarks.** Benchmarks like MathVerse have both an image config and a `text_only` config. The text-only config needs a separate `--subset text_only` conversion with a template that has no `<image>` placeholder and `--modalities text`. Merge the resulting DatasetDicts manually (see Step 4 merge snippet).
- **Empty string `""` for absent images silently drops rows.** In pre-downloaded JSON, use `None` (JSON `null`) for rows without images — not `""`. An empty string is a valid string that `_value_to_pil` tries (and fails) to decode as base64, causing `encode_failed` and dropping the row.
- **Video rows — use `--mode local`.** Videos go into `media` as paths or URLs; the template should use `<video>` placeholders. HF mode rejects video: `convert.py` raises `ValueError` if you try `--mode hf` on a video benchmark, because the HF schema uses `Sequence(Image())` and cannot carry video bytes.
- **Missing media is silently skipped.** `convert.py` reports a `skipped` count in `convert_summary.json`. Double-check it isn't unexpectedly large — `skipped > 0` almost always means a reference resolution bug, not genuinely missing data.
- **`--limit N` counts post-explode rows, not source rows.** With `--explode questions`, `--limit 5` keeps 5 sub-rows total, which may be fewer than 5 source rows.
- **Local `data.json` is loaded fully into RAM at eval time.** Simple-MMEval's `LocalJSONDataset` uses `json.load`. For benchmarks beyond ~500k rows or ~500MB JSON, prefer `--mode hf` (Arrow-backed, lazy) or shard the source upfront.
- **Don't pass `--token` on the command line for production pushes.** It lands in shell history. Prefer `export HF_TOKEN=...` and let `push_to_hf.py` pick it up via the env var.
- **HF cache not cleaned up.** After converting a large dataset, its parquet files remain in `~/.cache/huggingface/hub/`. Run `python3 scripts/cleanup.py --hf-cache <repo_id>` after push to reclaim the space.
- **Local machine paths in published metadata.json.** `validate.py --audit` catches these automatically. If authoring metadata by hand, ensure no `source.path` key contains an absolute local path before pushing.
- **Copy-paste `source.url` across subsets.** When a dataset has multiple subsets (e.g., `reasoning` + `descriptive`, `main` + `text_only`), it's easy to copy a subset block and forget to update its `source.url` keys. Each subset's `source.url` must reference **only that subset's** uploaded split names — not a sibling subset's keys. Always cross-check against `get_dataset_split_names` output after push.
- **No-fabrication contract.** Rows are dropped (not patched) when they break this contract:
  - `id` is `None` / missing → counted as `missing_required:id`.
  - Video URL has no recognizable extension → counted as `unknown_video_ext` (the converter refuses to assume `.mp4`).
  - An image fails to decode (broken base64, missing file, HTTP error, …) → counted as `encode_failed`.

  Optional schema fields with `None` source values keep their empty representation (`""` / `{}` / `[]`) — that's a legal "no value", not fabrication. List `options` are normalized to `{A: …, B: …}` per the documented label transform.

## Reference files

- `references/mmeval-format.md` — full schema details, with valid/invalid examples.
- `references/metadata-json.md` — authoring `metadata.json`, `--metadata-json` seed workflow, and what the converter refreshes vs preserves.
- `references/jinja-templates.md` — ready-made templates per benchmark style.

## When *not* to use this skill

- The user wants to add a new **model** to Simple-MMEval — that's a different task (see Simple-MMEval's CONTRIBUTING.md).
- The dataset is already in mm-eval format on the `mm-eval/` HF org — point Simple-MMEval at it directly with `--dataset mmeval_hf@mm-eval/<name>` (ensure it ships a root `metadata.json` in the new layout).
- **Pure text-only benchmarks** can use this skill, but the template must not contain any `<image>` / `<video>` placeholder, and `--map` must omit `image` / `images` / `media`.
- **Circular evaluation** (each MCQ replicated with rotated option labels) is *not* applied here — convert first, then run a second pass.
- **System messages** are not part of the mm-eval message schema (`role` ∈ `{user, assistant}`). Fold any system instruction into the first user message.
- **HF + video benchmarks** — convert with `--mode local` instead; the HF artifact format does not carry video bytes.
