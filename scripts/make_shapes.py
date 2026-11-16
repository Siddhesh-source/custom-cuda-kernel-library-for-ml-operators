"""Generate bench/shapes.json, the single source of truth for benchmark shapes.

reference.md §3 asks for shapes extracted from the production models. No
production model has been supplied, so the shapes come from published
architectures that fit the Kaggle T4 (docs/ROADMAP.md §4). Replace MODELS when
real shapes arrive and rerun:

    python scripts/make_shapes.py
"""

from __future__ import annotations

import json
import pathlib

# name: (hidden, ffn, heads, vocab)
MODELS = {
    "gpt2s":   (768, 3072, 12, 50257),
    "gpt2m":   (1024, 4096, 16, 50257),
    "llama1b": (2048, 8192, 32, 32000),
}
BERT_VOCAB = 30522

TOKENS = [1, 16, 128, 512, 2048]        # M = batch * seq; 1 and 16 are decode
LM_HEAD_TOKENS = [1, 16, 128]
SEQS = [128, 256, 512, 1024, 2048]

ODD_MATMUL = [(1, 1, 1), (7, 3079, 1025), (33, 33, 33), (127, 769, 1023),
              (1000, 1000, 1000), (1025, 4097, 33), (2047, 2049, 127)]
ODD_COLS = [1, 7, 33, 1000, 1025, 3079, 4097]


def matmul():
    out = []
    for name, (h, f, _heads, vocab) in MODELS.items():
        layers = {"qkv": (3 * h, h), "attn_out": (h, h),
                  "ffn_up": (f, h), "ffn_down": (h, f)}
        for layer, (n, k) in layers.items():
            for m in TOKENS:
                out.append({"tag": f"{name}.{layer}.m{m}", "M": m, "N": n, "K": k})
        for m in LM_HEAD_TOKENS:
            out.append({"tag": f"{name}.lm_head.m{m}", "M": m, "N": vocab, "K": h})
    for m, n, k in ODD_MATMUL:
        out.append({"tag": f"odd.{m}x{n}x{k}", "M": m, "N": n, "K": k})
    return out


def layernorm():
    out = []
    hiddens = sorted({h for h, _f, _n, _v in MODELS.values()} | {4096})
    for cols in hiddens:
        for rows in [1, 16, 128, 512, 2048, 8192]:
            out.append({"tag": f"h{cols}.rows{rows}", "rows": rows, "cols": cols})
    for cols in ODD_COLS:
        for rows in [128, 4096]:
            out.append({"tag": f"odd.c{cols}.rows{rows}", "rows": rows, "cols": cols})
    return out


def softmax():
    out = []
    # Attention scores: [batch * heads * seq, seq], batch 1.
    for name, (_h, _f, heads, _v) in MODELS.items():
        for s in SEQS:
            out.append({"tag": f"{name}.attn.s{s}", "rows": heads * s, "cols": s})
    out.append({"tag": "gpt2s.attn.b8.s512", "rows": 8 * 12 * 512, "cols": 512})
    # Vocabulary softmax: [tokens, vocab].
    for vocab in sorted({v for *_x, v in MODELS.values()} | {BERT_VOCAB}):
        for rows in [1, 16, 128]:
            out.append({"tag": f"vocab{vocab}.rows{rows}", "rows": rows, "cols": vocab})
    for cols in ODD_COLS:
        out.append({"tag": f"odd.c{cols}.rows1000", "rows": 1000, "cols": cols})
    return out


def main() -> None:
    shapes = {
        "_source": "scripts/make_shapes.py; models: " + ", ".join(MODELS)
                   + "; production shapes not supplied (docs/ROADMAP.md §4)",
        "matmul": matmul(),
        "layernorm": layernorm(),
        "softmax": softmax(),
    }
    path = pathlib.Path(__file__).resolve().parents[1] / "bench" / "shapes.json"
    path.write_text(json.dumps(shapes, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}: " + ", ".join(
        f"{op} {len(v)}" for op, v in shapes.items() if isinstance(v, list)))


if __name__ == "__main__":
    main()
