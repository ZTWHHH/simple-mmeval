# Ready-made Jinja templates

The HF layout stores the same template string in `metadata.json → subsets[<subset>].prompt_template`; it must be able to **reconstruct the original benchmark prompt** from the fields preserved in each `messages[0]` dict (see [`metadata-json.md`](metadata-json.md)).

Pick the closest template below and edit. The renderer has access to `zip`, `enumerate`, `len`, `range`, `list`, `dict`, `str`, `int`, `float`, `bool`, `sum`, `max`, `min`. The dict context is the **current message dict** — this converter always emits exactly one user message per row.

## Visible fields

The message dict you can reference inside the template:

- `role` — always `"user"` for converter output.
- `question` — string (`""` if absent in the source).
- `answer` — string, or list when `--answer-list` is set (`""` / `[]` if absent).
- `hint` — string (`""` if absent).
- `options` — dict (`{}` if absent). Source list options like `["cat", "dog"]` are normalized to `{"A": "cat", "B": "dog"}`, so MCQ templates can iterate `options.items()` regardless of source shape.
- `choices` — list (`[]` if absent).
- Any extra `--map foo=bar` pair lands as `foo` (and only when the source value is non-`None`).

## Single-image VQA (free-form answer)

VizWiz-VQA, OK-VQA, GQA, etc.

```jinja
<image>{{ question }}
Answer the question using a single word or phrase.
```

## Single-image with optional hint

```jinja
<image>{% if hint %}Hint: {{ hint }}
{% endif %}Question: {{ question }}
```

## Multiple-choice (MMBench / SEED / MMMU style)

```jinja
<image>{% if hint %}Hint: {{ hint }}
{% endif %}Question: {{ question }}
Options:
{% for k, v in options.items() %}{{ k }}. {{ v }}{% if not loop.last %}
{% endif %}{% endfor %}
Please select the correct answer from the options above.
```

## Multi-image (interleaved)

When the prompt itself names the images:

```jinja
{% for img in media_labels %}<image>{{ img }}{% if not loop.last %} {% endif %}{% endfor %}
{{ question }}
```

Or two fixed images A/B:

```jinja
Given Image A <image> and Image B <image>: {{ question }}
```

## Video QA

```jinja
<video>{{ question }}
Answer briefly.
```

## Text-only (sanity / no media)

```jinja
{{ question }}
```

## Non-English VQA

For benchmarks where the evaluation prompt itself must be in the dataset's native language (e.g. a Chinese benchmark like CCBench or MMBench-CN, a Portuguese benchmark like ENEM), write the template using the target language's instruction strings. Take the exact wording from the benchmark's official evaluation code — do not translate or paraphrase, as any change to instruction phrasing can affect reported scores.

The structure follows the standard MCQ or free-form pattern; only the instruction strings (hint prefix, question prefix, option label, answer instruction) change. For example, a Portuguese MCQ template follows the same `{% if hint %}…{% endif %}{% for k, v in options.items() %}…{% endfor %}` skeleton as the English version, with Portuguese instruction literals taken verbatim from the official eval script.

Always verify the rendered prompt against the official `construct_prompt` output byte-for-byte before running the full conversion.

## How to test a template quickly

```python
import re
from jinja2 import Environment
env = Environment()
env.globals.update({"zip": zip, "enumerate": enumerate, "len": len, "range": range,
                    "list": list, "dict": dict, "str": str, "int": int, "float": float,
                    "bool": bool, "sum": sum, "max": max, "min": min})
rendered = env.from_string(TEMPLATE).render(
    question="What is this?",
    options={"A": "car", "B": "boat"},
    hint="",
)
print(rendered)

# Sanity-check placeholder count vs media count.
media_list = ["img0.png"]
n_placeholders = len(re.findall(r"<(image|video)>", rendered))
assert n_placeholders == len(media_list), \
    f"template has {n_placeholders} placeholders but media has {len(media_list)} items"
```

If your template doesn't include any `<image>` placeholder but the dataset has images, the runner will raise a media/placeholder mismatch — add `<image>` (or `<video>`) explicitly.
