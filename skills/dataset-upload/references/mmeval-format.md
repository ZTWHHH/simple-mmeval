# mm-eval data format reference

Simple-MMEval reads two input shapes; both ultimately produce per-sample dicts with `id`, `media`, and `messages`.

## Local JSON shape (`--dataset local@json --infile <data.json> --img_dir <media_dir>`)

`data.json` is a JSON array. Each entry:

```json
{
  "id": "vw_val_00000000",
  "media": ["vw_val_00000000.png"],
  "messages": [
    {
      "role": "user",
      "question": "<image> What does this say?",
      "answer": "",
      "options": {},
      "choices": [],
      "prompt": "<image> What does this say?\nAnswer the question using a single word or phrase.",
      "hint": ""
    }
  ]
}
```

- `media` entries are **basenames relative to `--img_dir`**.
- `prompt` is the rendered prompt; if present, no template is applied at runtime.
- `answer`/`options`/`choices`/`hint` are optional but conventional — keep them for round-trippability.
- The runner adds `eval-id` (an integer index) when iterating.

## HF dataset shape (`--dataset mmeval_hf@<user>/<repo>` [+ `--subset <name>` when the manifest has multiple subsets])

### `default` config
Arrow table(s) on the Hub. When converted with an explicit `--split`, this is a named `DatasetDict` split (`val`, `test`, …). When converted from local JSON/TSV/CSV **without** `--split`, the on-disk artifact is a flat `Dataset` (Hub push lands on split `train`). Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | string | unique within the split |
| `media` | `Sequence(Image())` | always emitted as a sequence by this converter; the runtime also accepts `Image()` (single-image) but converted artifacts use the sequence form for uniformity. **Video media is not supported in HF mode** — use local mode. |
| `messages` | string (JSON) | `json.dumps([{role, question, ...}], ensure_ascii=False)` — JSON text so each HF row mirrors local `data.json`'s `messages[0]` payload; consumers don't branch on mode. |

### `metadata.json` (repo root)
Not a second dataset config — a single JSON file downloaded at runtime (`hf_hub_download(..., filename="metadata.json")`). It documents subsets, modalities, source provenance, per-field mappings, and the Jinja `prompt_template` used to re-render HF rows. See [`metadata-json.md`](metadata-json.md) for the full schema and authoring workflow.

### How prompts are built

Per row, per message:

1. If `messages[i].prompt` is set, use it as-is.
2. Else render the template with the message dict as context.
3. Replace each `<image>` / `<video>` placeholder with the next media item from `media` (consumed in order across messages).

The number of placeholders rendered must equal the number of media for that message. Mismatches raise immediately.

## Required canonical fields per message

`role` ∈ {`user`, `assistant`}. The base template expects:

- `question` — string
- `options` — dict (`{}` for non-MCQ)
- `hint` — string (`""` if absent)

Any other key is allowed and visible to your custom template (`choices`, `category`, `image_id`, etc.).

## Local vs HF prompt rendering

- **Local mode**: prompts are rendered at conversion time and frozen into `messages[i].prompt`. The runner uses them as-is; the template is *not* re-applied. This keeps local artifacts deterministic and self-contained.
- **HF mode**: only the message dict is shipped; the template lives in `metadata.json → subsets[<subset>].prompt_template` and is re-rendered per row at eval time. This lets you fix a template and re-push without re-encoding images.

## Validation checklist

- [ ] `id` is a string and unique within the split.
- [ ] `media` is a list (single-image still becomes `[img]`).
- [ ] In HF mode, `messages` is a JSON-encoded string serialized with `ensure_ascii=False` (CJK content otherwise inflates 3-5×).
- [ ] In local mode, every `media[i]` is a **basename** relative to `--img_dir` — no `..`, no absolute paths, no URLs.
- [ ] Rendered prompt's placeholder count matches `len(media)` per message.
- [ ] `metadata.json` `mapping_from_source.source.url` is a `{split_name: URL, …}` dict; keys correspond to actual uploaded split names. No `links`, `repo`, or `path` keys in `source`.
- [ ] No video media in `--mode hf` (the converter raises `ValueError` on this; double-check if you assembled the artifact by hand).
