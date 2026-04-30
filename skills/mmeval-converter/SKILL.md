---
name: mmeval-converter
description: Convert any multimodal benchmark (HuggingFace dataset, local JSON, CSV/TSV with images) into Simple-MMEval runnable format. Use this skill whenever the user wants to prepare/normalize/port a vision-language eval dataset for Simple-MMEval, run an existing benchmark through Simple-MMEval, build a `local@json` data file with an `img_dir`, build an `mmeval_hf@`-compatible HF dataset (with `media` / `messages` / `id` + Jinja `metadata` config), or push a converted dataset to HuggingFace Hub. Trigger even if the user does not explicitly say "Simple-MMEval" — phrases like "convert this VQA dataset", "make this runnable in mmeval", "wrap this for evaluation", "turn this into an eval dataset", or "push this benchmark to HF for mmeval" all apply.
---

# mmeval-converter

Convert a multimodal eval dataset into one of two runnable input formats accepted by [Simple-MMEval](https://github.com/mm-evaluation/simple-mmeval):

1. **`local`** — a single JSON file plus a directory of media files (`--dataset local@json --infile <data.json> --img_dir <media_dir>`).
2. **`hf`** — a HuggingFace `DatasetDict` with two configs (`default` and `metadata`) (`--dataset mmeval_hf@<user>/<repo>`).

The converter can emit one mode or **both at once in a single source pass** (`--mode both`), which is the recommended path when the user wants the local artifact for quick smoke runs *and* an HF artifact to push.

## Files in this skill

```
mmeval-converter/
├── SKILL.md                   ← you are here
├── references/
│   ├── mmeval-format.md       ← exact target schema
│   └── jinja-templates.md     ← copy-paste templates
└── scripts/
    ├── inspect_source.py      ← peek source schema (HF or local JSON)
    ├── convert.py             ← do the conversion (one split at a time)
    ├── merge_splits.py        ← merge per-split HF artifacts into one DatasetDict
    ├── validate.py            ← round-trip a converted artifact through Simple-MMEval
    └── push_to_hf.py          ← final HF push (token + repo)
```

> The inspect script is named `inspect_source.py` rather than `inspect.py` because the latter shadows Python's stdlib `inspect` module when run from the same directory and breaks pandas import.

## Workflow

### Step 1 — Inspect the source

```bash
python3 scripts/inspect_source.py --hf <repo_id> --split <split>
python3 scripts/inspect_source.py --json <data.json> --media-dir <dir>
```

Use the printed columns + types + sample row to decide the column mapping with the user.

### Step 2 — Decide the column mapping and template

Ask the user (or infer) which source fields correspond to:

- `id` → unique id (required)
- `image` (or `images` / `media`) → media — single value or list (required for image/video benchmarks)
- `question` → required
- `answer` → optional (often empty for test splits)
- `options` / `choices` / `hint` → only for MCQ-style benchmarks

Then pick a Jinja template. A good default for single-image VQA:

```jinja
<image>{{ question }}
Answer the question using a single word or phrase.
```

Match the prompt style of the original benchmark (terse for VizWiz/OK-VQA; "Please select the correct option" for MMBench). See `references/jinja-templates.md` for ready-made templates per benchmark style.

### Step 3 — Convert

`scripts/convert.py` reads the source, applies the column map and template, and writes the artifact(s).

**Source flags** (pick one):
- `--hf <repo_id> --split <split>`
- `--json <path> --media-dir <dir>`

**Mapping flags:**
- `--map id=<src> question=<src> image=<src> answer=<src> ...` — space-separated `canonical=source` pairs.
- `--template <path-or-string>` — Jinja template (string or file path).
- `--prompt-prefix "<image>"` — auto-prepend the placeholder if the template doesn't include it.
- `--answer-list` — keep `answer` as a list when the source provides a list (default joins with space).
- `--answer-join " "` — join string when reducing list answers to a single string.

**Output flags:**
- `--mode local | hf | both`
- `--out <dir>` — output directory.
- `--workers N` — parallelism for image encoding (default 8; 16 is a good fit on multi-core hosts).
- `--image-format png | jpeg` — PNG (default, lossless) or JPEG (much smaller — 5-10× — at quality 92).
- `--limit N` — cap rows for testing.
- `--explode <field>` — flatten a list-valued field (e.g. `--explode questions` for nested-MCQ datasets like CaptionQA). Each element becomes its own row, with the inner dict's keys merged onto the outer row, and the id auto-suffixed via `--id-template '{id}_q{idx}'` (default).
- `--id-template '{id}_q{idx}'` — id format string for exploded rows.

For datasets with **multiple domain splits** that should land in one HF repo (e.g. CaptionQA with `natural`/`document`/`ecommerce`/`embodiedai`), run `convert.py --mode hf` once per split into separate output dirs, then combine them:

```bash
python3 scripts/merge_splits.py \
    --inputs <out>/natural <out>/document <out>/ecommerce <out>/embodiedai \
    --out <out>/merged
```

The merged directory has one `hf_dataset/` and one `hf_metadata/` ready for `push_to_hf.py`.

**Output layout:**

```
<out>/
├── data.json               # local mode
├── media/                  # local mode (also reused for HF bytes)
├── hf_dataset/             # HF mode — DatasetDict for the `default` config
├── hf_metadata/            # HF mode — DatasetDict for the `metadata` config
└── convert_summary.json
```

### Step 4 — Verify

`scripts/validate.py` round-trips the artifact through Simple-MMEval's actual data loaders (`LocalJSONDataset` and a load-from-disk wrapper around `MMEvalHFDataset`):

```bash
python3 scripts/validate.py \
    --simple-mmeval <path-to-simple-mmeval> \
    --local <out> \
    --hf <out> \
    -n 3
```

It prints rendered prompts and confirms images load as PIL with the right dimensions. Use it before declaring a conversion done.

### Step 5 — Push to HuggingFace (for `hf` mode)

`scripts/push_to_hf.py` is the **final user-facing push script**. It only needs the artifact dir + an HF token + a target repo name:

```bash
python3 scripts/push_to_hf.py \
    --artifact-dir <out> \
    --repo-id <user>/<repo> \
    --token <hf_token> \
    [--private]
```

It pushes the `default` and `metadata` configs as separate configs on the same repo, and prints the exact `simple-mmeval` invocation for the user to copy.

## Worked example — CaptionQA (nested-MCQ, 4 domain splits)

```bash
SKILL=<skill-path>
ROOT=<workdir>/captionqa

# 1. Convert each split (parallel)
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

# 3. Push
python3 $SKILL/scripts/push_to_hf.py \
    --artifact-dir $ROOT/merged --repo-id <user>/CaptionQA-mmeval --token "$HF_TOKEN"
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

Then verify both:

```bash
python3 scripts/validate.py \
    --simple-mmeval <simple-mmeval-checkout> \
    --local <workdir>/vizwiz_val \
    --hf <workdir>/vizwiz_val \
    -n 3
```

Run via Simple-MMEval (local artifact, no HF push needed):

```bash
cd <simple-mmeval-checkout>
export PYTHONPATH=./:$PYTHONPATH
python3 mmeval/run.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
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
    --repo-id <user>/VizWiz-VQA-mmeval \
    --token "$HF_TOKEN"
```

## Common pitfalls (learned the hard way)

- **`messages` is a JSON string in HF mode, a list in local mode.** `convert.py` handles this for you — don't try to make them both lists.
- **Don't pass PIL images to `Dataset.from_dict({"media": ...})` for big datasets.** `datasets<=4.5` has a buggy `embed_storage` path that fails at multi-shard sizes ("Mask must be a pyarrow.Array of type boolean"). The skill writes images to disk first and then feeds raw bytes (`{"path": None, "bytes": <raw>}`) into the HF column — that path is fast (~70 rows/s end-to-end with 16 workers) and avoids the bug.
- **PNG vs JPEG.** PNG (default) is lossless but 5-10× larger on disk and over the wire. For benchmark conversion where the model's eval result is rate-limiting anyway, `--image-format jpeg --jpeg-quality 92` produces much smaller artifacts with no measurable accuracy impact.
- **Multi-image rows.** Use `--map images=<list-field>` and ensure the template emits the right number of `<image>` placeholders.
- **Video rows.** Videos go into `media` as paths (or URLs); the template should use `<video>` placeholders.
- **Missing media is silently skipped.** `convert.py` reports a `skipped` count in `convert_summary.json`. Double-check it isn't unexpectedly large.

## Reference files

- `references/mmeval-format.md` — full schema details, with valid/invalid examples.
- `references/jinja-templates.md` — ready-made templates per benchmark style.

## When *not* to use this skill

- The user wants to add a new **model** to Simple-MMEval — that's a different task (see Simple-MMEval's CONTRIBUTING.md).
- The dataset is already in mm-eval format on the `mm-eval/` HF org — point Simple-MMEval at it directly with `--dataset mmeval_hf@mm-eval/<name>`.
