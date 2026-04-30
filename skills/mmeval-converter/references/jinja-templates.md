# Ready-made Jinja templates

Pick the closest one and edit. The renderer has access to `zip`, `enumerate`, `len`, `range`, `list`, `dict`, `str`, `int`, `float`, `bool`, `sum`, `max`, `min`. The dict context is the **first message** of the row.

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

## How to test a template quickly

```python
from jinja2 import Environment
env = Environment()
env.globals.update({"zip": zip, "enumerate": enumerate, "len": len})
print(env.from_string(TEMPLATE).render(question="What is this?", options={"A":"car","B":"boat"}, hint=""))
```

If your template doesn't include any `<image>` placeholder but the dataset has images, the runner will raise a media/placeholder mismatch — add `<image>` (or `<video>`) explicitly.
