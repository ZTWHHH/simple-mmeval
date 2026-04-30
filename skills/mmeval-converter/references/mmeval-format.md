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

## HF dataset shape (`--dataset mmeval_hf@<user>/<repo>`)

Two configs in the same repo:

### `default` config
Splits as the user expects (`val`, `test`, …). Columns:

| Column | Type | Notes |
|---|---|---|
| `id` | string | unique |
| `media` | `Sequence(Image())` *or* `Image()` | one image per row, or a list when multi-image. Videos are stored as path/URL strings. |
| `messages` | string (JSON) | `json.dumps([{role, question, ...}])` |

### `metadata` config
Same split names as `default`. One row per split. Columns:

| Column | Type | Notes |
|---|---|---|
| `jinja_template` | string | Renders the message dict. Must emit `<image>` / `<video>` placeholders matching `media` count. |
| `version` | string | Free-form (e.g. `"v1"`). |
| `metadata` | string (JSON) | Optional column-mapping snapshot for documentation. |

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

## Validation checklist

- [ ] `id` is a string and unique within the split.
- [ ] `media` is a list (single-image still becomes `[img]`).
- [ ] In HF mode, `messages` is a JSON-encoded string.
- [ ] In local mode, every `media[i]` resolves to an existing file under `--img_dir`.
- [ ] Rendered prompt's placeholder count matches `len(media)` per message.
- [ ] `metadata` config has the same split names as `default`.
