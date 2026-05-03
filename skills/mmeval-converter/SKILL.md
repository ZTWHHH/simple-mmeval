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
    ├── inspect_source.py            ← peek source schema (HF or local JSON)
    ├── convert.py                   ← generic conversion (one split at a time)
    ├── convert_indexed_multimage.py ← MMMU-style: <image N> refs + per-subject configs
    ├── merge_splits.py              ← merge per-split HF artifacts into one DatasetDict
    ├── validate.py                  ← round-trip a converted artifact through Simple-MMEval
    └── push_to_hf.py                ← final HF push (token + repo)
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

- **`messages` is a JSON string in HF mode, a list in local mode.** `convert.py` handles this for you — don't try to make them both lists.
- **Don't pass PIL images to `Dataset.from_dict({"media": ...})` for big datasets.** `datasets<=4.5` has a buggy `embed_storage` path that fails at multi-shard sizes ("Mask must be a pyarrow.Array of type boolean"). The skill writes images to disk first and then feeds raw bytes (`{"path": None, "bytes": <raw>}`) into the HF column — that path is fast (~70 rows/s end-to-end with 16 workers) and avoids the bug.
- **PNG vs JPEG.** PNG (default) is lossless but 5-10× larger on disk and over the wire. For benchmark conversion where the model's eval result is rate-limiting anyway, `--image-format jpeg --jpeg-quality 92` produces much smaller artifacts with no measurable accuracy impact.
- **Multi-image rows.** Use `--map images=<list-field>` and ensure the template emits the right number of `<image>` placeholders. For indexed `<image N>` patterns (MMMU-style) see the worked example above — that shape needs custom preprocessing.
- **Don't trust intermediate "convenience" reformats — re-source from the official upstream.** Third-party consolidations of a benchmark sometimes pre-render a `prompt` field whose token count silently disagrees with the unique-image count, or strip metadata (subject, difficulty, explanation) you'd want later. When the upstream is available, source from it.
- **Prefer a Jinja template in `metadata` over a pre-rendered `prompt` on the message.** Per the format spec, when the metadata template is non-empty Simple-MMEval *always* re-renders it (the per-row `prompt` becomes a fallback only when rendering throws). A template lets a downstream user fork the repo and change the prompt without re-running the converter. Bake `prompt` only when the rendering is genuinely per-row dynamic in a way Jinja can't express.
- **Source the prompt template from the published paper and the official eval code — in that order — and cross-check both.** The arxiv / project page often quotes the prompt verbatim in a figure or appendix; the eval code carries the actual format string used to produce reported scores. They should match; when they disagree (rare but it happens) the eval-code version is what produced the reported numbers, so it wins. Cite the file path / line numbers / paper section in the `metadata.metadata` JSON so a future reader can audit the choice.
- **Confirm the chosen template with the user before running the full conversion.** Show the rendered prompt for one MC and one short-answer row, name the source you took it from (paper §X / file L#–L#), and wait for a thumbs-up. Re-running a conversion + re-pushing a multi-GB HF artifact because the prompt was wrong costs minutes you didn't need to spend; a 5-line preview is cheap.
- **Verify your template byte-for-byte against the upstream eval code.** When the source has an official eval script, render your template with a sample row and assert string-equality with `upstream.construct_prompt(row)`. Tiny discrepancies (extra blank line, `(A)` vs `A.`, `option's letter` vs `option letter`, missing trailing instruction) silently change scores — the assert catches them in 5 seconds.
- **Multi-config benchmarks → one split per config.** When a source has multiple HF configs that should evaluate as a single benchmark (MMMU-Pro: `standard (4 options)` + `standard (10 options)` + `vision`; MMMU: 30 subjects), run the converter per config into separate output dirs and `merge_splits.py` them. Pick split names without spaces or parens (`standard_4_options`, not `standard (4 options)`) — HF Hub config/split names disallow special chars.
- **Video rows.** Videos go into `media` as paths (or URLs); the template should use `<video>` placeholders.
- **Missing media is silently skipped.** `convert.py` reports a `skipped` count in `convert_summary.json`. Double-check it isn't unexpectedly large — `skipped > 0` almost always means a reference resolution bug, not genuinely missing data.

## Reference files

- `references/mmeval-format.md` — full schema details, with valid/invalid examples.
- `references/jinja-templates.md` — ready-made templates per benchmark style.

## When *not* to use this skill

- The user wants to add a new **model** to Simple-MMEval — that's a different task (see Simple-MMEval's CONTRIBUTING.md).
- The dataset is already in mm-eval format on the `mm-eval/` HF org — point Simple-MMEval at it directly with `--dataset mmeval_hf@mm-eval/<name>`.
