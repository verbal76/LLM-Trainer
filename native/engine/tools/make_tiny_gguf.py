#!/usr/bin/env python3
"""Build a tiny llama-architecture GGUF with RANDOM weights and a REAL tokenizer, for engine tests.

Not a pretrained model: it exists so the engine's load/generate/score/train/patch pipeline can be exercised for real
(no mocks) on any box.  Deterministic for a given (--seed, shape).

  --vocab bytes : byte-level SentencePiece-style vocab created here (ASCII pieces + 256 byte-fallback tokens; ~370 tokens)
  --vocab spm   : the real 32k SentencePiece vocab from llama.cpp's models/ggml-vocab-llama-spm.gguf (pass --vocab-file)
Sizes: --preset tiny|1m|5m|15m|40m|135m|360m (135m ~ SmolLM2-135M shape) or explicit --n-embd/--n-layer/...
"""
import argparse
import os
import sys

import numpy as np
import gguf

PRESETS = {  # n_embd, n_layer, n_head, n_head_kv, n_ff
    "tiny": (32, 2, 4, 2, 96),
    "1m":   (96, 4, 4, 2, 256),
    "5m":   (192, 6, 6, 2, 512),
    "15m":  (288, 8, 6, 2, 768),
    "40m":  (448, 12, 8, 4, 1152),
    "135m": (576, 30, 9, 3, 1536),
    "360m": (960, 32, 15, 5, 2560),
}


def byte_vocab():
    toks, scores, types = [], [], []

    def add(t, s, ty):
        toks.append(t)
        scores.append(s)
        types.append(int(ty))

    add("<unk>", 0.0, gguf.TokenType.UNKNOWN)
    add("<s>", 0.0, gguf.TokenType.CONTROL)
    add("</s>", 0.0, gguf.TokenType.CONTROL)
    for b in range(256):
        add("<0x%02X>" % b, 0.0, gguf.TokenType.BYTE)
    add("▁", -1.0, gguf.TokenType.NORMAL)  # SentencePiece space marker
    for c in range(33, 127):
        add(chr(c), -1.0, gguf.TokenType.NORMAL)
    for p, s in [("▁the", -0.1), ("▁is", -0.1), ("▁of", -0.1), ("th", -0.2), ("er", -0.2), ("in", -0.2),
                 ("▁The", -0.1), ("N-m", -0.05), ("▁valve", -0.05), ("▁torque", -0.05)]:
        add(p, s, gguf.TokenType.NORMAL)
    return toks, scores, types


def spm_vocab(path):
    r = gguf.GGUFReader(path)
    f = r.fields
    toks = [bytes(f["tokenizer.ggml.tokens"].parts[i]).decode("utf-8", "replace") for i in f["tokenizer.ggml.tokens"].data]
    scores = [float(f["tokenizer.ggml.scores"].parts[i][0]) for i in f["tokenizer.ggml.scores"].data]
    types = [int(f["tokenizer.ggml.token_type"].parts[i][0]) for i in f["tokenizer.ggml.token_type"].data]
    return toks, scores, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    ap.add_argument("--n-embd", type=int)
    ap.add_argument("--n-layer", type=int)
    ap.add_argument("--n-head", type=int)
    ap.add_argument("--n-head-kv", type=int)
    ap.add_argument("--n-ff", type=int)
    ap.add_argument("--n-ctx", type=int, default=512)
    ap.add_argument("--vocab", default="bytes", choices=["bytes", "spm"])
    ap.add_argument("--vocab-file", default=None)
    ap.add_argument("--tied", action="store_true", help="no separate output.weight (output tied to token_embd)")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16"])
    ap.add_argument("--quant", default=None, choices=["q8_0", "q4_0", "q4_1", "q5_0"],
                    help="quantize 2-D weight matrices (norms stay F32), like llama-quantize would")
    ap.add_argument("--from-model", default=None, help="take all tensors from this GGUF instead of random init (same shape flags needed only for metadata)")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init-std", type=float, default=0.02)
    a = ap.parse_args()
    d, L, H, HKV, FF = PRESETS[a.preset]
    d = a.n_embd or d
    L = a.n_layer or L
    H = a.n_head or H
    HKV = a.n_head_kv or HKV
    FF = a.n_ff or FF
    if a.vocab == "spm":
        if not a.vocab_file:
            sys.exit("--vocab spm needs --vocab-file")
        toks, scores, types = spm_vocab(a.vocab_file)
    else:
        toks, scores, types = byte_vocab()
    V = len(toks)
    rng = np.random.default_rng(a.seed)
    dt = np.float32 if a.dtype == "f32" else np.float16
    w = gguf.GGUFWriter(a.out, "llama")
    w.add_name("hag-tiny-random-%s" % a.preset)
    w.add_context_length(a.n_ctx)
    w.add_embedding_length(d)
    w.add_block_count(L)
    w.add_feed_forward_length(FF)
    w.add_head_count(H)
    w.add_head_count_kv(HKV)
    w.add_rope_dimension_count(d // H)
    w.add_layer_norm_rms_eps(1e-5)
    ft = gguf.LlamaFileType.ALL_F32 if a.dtype == "f32" else gguf.LlamaFileType.MOSTLY_F16
    if a.quant:
        ft = {"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
              "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1, "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0}[a.quant]
    w.add_file_type(ft)
    w.add_vocab_size(V)
    w.add_tokenizer_model("llama")
    w.add_tokenizer_pre("default")
    w.add_token_list(toks)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_unk_token_id(0)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    if not a.no_chat_template:
        w.add_chat_template("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    qt = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0,
          "q4_1": gguf.GGMLQuantizationType.Q4_1, "q5_0": gguf.GGMLQuantizationType.Q5_0}.get(a.quant)
    src = None
    if a.from_model:
        rd = gguf.GGUFReader(a.from_model)
        src = {t.name: np.array(t.data, dtype=np.float32) for t in rd.tensors}

    def mat(*shape):
        return (rng.standard_normal(shape) * a.init_std).astype(dt)

    def add(name, arr):
        if src is not None:
            arr = src[name]
        if qt is not None and arr.ndim == 2:
            w.add_tensor(name, gguf.quants.quantize(arr.astype(np.float32), qt), raw_dtype=qt)
        else:
            w.add_tensor(name, arr)

    def ones(n):
        return np.ones(n, dtype=np.float32)

    add("token_embd.weight", mat(V, d))
    add("output_norm.weight", ones(d))
    if not a.tied:
        add("output.weight", mat(V, d))
    dk = d // H
    for i in range(L):
        p = "blk.%d." % i
        add(p + "attn_norm.weight", ones(d))
        add(p + "attn_q.weight", mat(H * dk, d))
        add(p + "attn_k.weight", mat(HKV * dk, d))
        add(p + "attn_v.weight", mat(HKV * dk, d))
        add(p + "attn_output.weight", mat(d, H * dk))
        add(p + "ffn_norm.weight", ones(d))
        add(p + "ffn_gate.weight", mat(FF, d))
        add(p + "ffn_up.weight", mat(FF, d))
        add(p + "ffn_down.weight", mat(d, FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    n = V * d * (1 if a.tied else 2) + L * (2 * d + 2 * H * dk * d + 2 * HKV * dk * d + 3 * FF * d) + d
    print("wrote %s: %s L=%d d=%d V=%d params=%.2fM size=%.1f MB" %
          (a.out, a.preset, L, d, V, n / 1e6, os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
