#!/usr/bin/env python3
"""garas as a Decision Index engine (the kit's `Engine` interface: engine(state, questions) -> (response, raw)), for the kit's runner
(github.com/apolinario/decision-index; tested at commit 9eb2dbe2).

    python -m decision_index run --engine garas_engine:GarasEngine \\
        --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garas \\
        --rows rows.jsonl.gz --out runs/garas

Inference procedure:
  * each request is converted to one multiple-choice item per question (choice: the option keys as given; noul: yes / no);
    a choice with more than 52 options is split into ceil(K / 52) chunks;
  * one prompt per item, using the model's chat template; options lettered A-Z a-z. A choice question's options are shown in
    one fixed non-identity order derived from a hash of (state, question key, option keys); yes / no keeps its order;
  * one forward pass per prompt; the next-token logits of the option letters -> softmax -> mapped back to the options;
  * K > 52: round 1 scores the chunks, round 2 scores the top 52 // C options of every chunk; the final distribution puts
    (1 - 0.001) on the round-2 options and 0.001 on the rest, in proportion to round 1;
  * no truncation: a prompt that does not fit `max_len` tokens without cutting the state raises the kit's Unsupported;
  * the prompts of one request run in right-padded batches of at most `tokens_per_batch` padded tokens.
The adapter is loaded with autocast_adapter_dtype=False, so the LoRA update is computed in the base model's dtype (bf16), as in
training; peft's default would upcast it to fp32 and shift some letter logits by a bf16 step.
Dependencies: torch, transformers, peft, numpy, and the kit on sys.path.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garas_core as core                                          # noqa: E402

from decision_index.engines.base import Engine, Unsupported      # noqa: E402  (the kit must be on sys.path)

TEST_BACKENDS: dict = {}


def request_id(state, questions):
    return "eng-" + hashlib.sha1(json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=False,
                                            default=str).encode("utf-8")).hexdigest()[:16]


def question_seed(state_text, qid_key, keys):
    h = hashlib.sha256(json.dumps([state_text, qid_key, list(keys)], ensure_ascii=False).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def option_order(qtype, k, seed):
    if qtype != "choice" or k < 2:
        return list(range(k))
    rng = np.random.default_rng(seed)
    while True:
        o = [int(x) for x in rng.permutation(k)]
        if o != list(range(k)):
            return o


class GarasEngine(Engine):
    name = "garas"
    latency = ("CUDA-synchronized in-process request wall time: conversion, one prompt per question (the request's prompts in "
               "right-padded batches; K > 52 round 2 after round 1), letter read-out; excludes model loading.")

    def __init__(self, base="google/gemma-4-31B-it", adapter="plantwaterer/garas", device="cuda", max_len=32768,
                 tokens_per_batch=65536, _backend=None, **options):
        super().__init__(**options)
        self.max_len, self.tokens_per_batch = int(max_len), int(tokens_per_batch)
        if _backend is not None:                   # tests: (tok, chat, letter_logits(texts) -> [B, >= k], sync), or its key
            self.tok, self.chat, self._logits, self._sync = TEST_BACKENDS[_backend] if isinstance(_backend, str) else _backend
            self.provenance = {"kind": "test backend"}
            return
        self._load(base, adapter, device)

    def _load(self, base, adapter, device):
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        mcfg = AutoConfig.from_pretrained(base)
        _, model_cls = core.model_loader(mcfg, "auto")
        self.chat = core.chat_preset(mcfg.model_type, False)
        tok = AutoTokenizer.from_pretrained(base)
        tok.padding_side = "right"
        ids = [tok(c, add_special_tokens=False)["input_ids"] for c in core.LABELS]
        if any(len(x) != 1 for x in ids):
            raise SystemExit("a letter is not a single token")
        label_ids = torch.tensor([x[0] for x in ids])
        m = model_cls.from_pretrained(base, dtype=torch.bfloat16, device_map=device).eval()
        pm = PeftModel.from_pretrained(m, adapter, is_trainable=False, autocast_adapter_dtype=False)   # adapter math in bf16
        m = pm.get_base_model().eval()          # the same module tree, LoRA layers injected in place
        self.tok, self.model, self.device = tok, m, device
        lab = label_ids.to(device)

        def letter_logits(texts):
            with torch.no_grad():
                return core.exact_letter_logits(m, tok, texts, lab, device, "pad").float().cpu().numpy()

        self._logits = letter_logits
        self._sync = (lambda: torch.cuda.synchronize()) if str(device).startswith("cuda") else (lambda: None)
        import peft
        import transformers
        self.provenance = {"kind": "in-process", "base": base, "adapter": str(adapter), "dtype": "bfloat16",
                           "peft": peft.__version__, "transformers": transformers.__version__, "torch": torch.__version__,
                           "core_md5": hashlib.md5(open(core.__file__, "rb").read()).hexdigest(),
                           "engine_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest()}

    def synchronize(self):
        self._sync()

    def runtime(self):
        try:
            import torch
            return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        except Exception:                          # noqa: BLE001
            return {}

    def unit(self, rec, qid):
        q = rec["questions"][qid]
        keys, texts, _ = core.question_options(q, rec["labels"][qid])
        return {"state": rec["state"], "instructions": q["instructions"], "qtype": q["type"], "option_keys": keys,
                "option_texts": texts, "noul_criteria": q.get("criteria") if q["type"] == "noul" else None}

    def prompt(self, u, qid):
        order = option_order(u["qtype"], len(u["option_keys"]), question_seed(u["state"], qid.split("|", 1)[1], u["option_keys"]))
        text = core.build_prompt(self.tok, u, order, self.max_len, chat=self.chat)
        if text is None or u["state"] not in text:
            raise Unsupported(f"maximum context length: question {qid.split('|', 1)[1]!r} does not fit {self.max_len} tokens "
                              f"without cutting the state (no truncation)")
        return order, text

    def run_prompts(self, items):
        out = {}
        lens = [len(self.tok(t, add_special_tokens=False)["input_ids"]) for _, _, t in items]
        i = 0
        while i < len(items):
            j, mx = i, 0
            while j < len(items) and max(mx, lens[j]) * (j - i + 1) <= max(self.tokens_per_batch, lens[i]):
                mx = max(mx, lens[j])
                j += 1
            z = np.asarray(self._logits([t for _, _, t in items[i:j]]), dtype=np.float64)
            for b, (qid, order, _) in enumerate(items[i:j]):
                k = len(order)
                zk = z[b, :k].astype(np.float32).astype(np.float64)
                logp = zk - zk.max() - np.log(np.exp(zk - zk.max()).sum())
                p = np.zeros(k)
                p[np.asarray(order)] = np.exp(logp)
                out[qid] = p.tolist()
            i = j
        return out

    def __call__(self, state, questions):
        for k, q in questions.items():
            if q.get("type") not in ("choice", "noul"):
                raise Unsupported(f"question type {q.get('type')!r}")
            if q["type"] == "choice" and len(q.get("criteria") or {}) < 2:
                raise Unsupported("a choice needs at least two options")
        rid = request_id(state, questions)
        stats = {"k_hist": collections.Counter(), "k52_questions": 0, "units": 0}
        row = {"state": state, "questions": questions, "expected": {}, "_evaluation": {"catalog_id": 57, "run_id": rid}}
        rec = core.convert_row(row, None, stats, "text")
        items = []
        for qid in rec["questions"]:
            u = self.unit(rec, qid)
            order, text = self.prompt(u, qid)
            items.append((qid, order, text))
        probs = self.run_prompts(items)
        finals = []
        for qkey in (rec["meta"].get("k52") or {}):
            keys = core.finalists(rec, qkey, probs)
            fq = core.qid_of(rec["id"], qkey, "final")
            q2, lab, kind = core.final_question(rec, qkey, keys)
            r2 = {**rec, "questions": {fq: q2}, "labels": {fq: lab}}
            u = self.unit(r2, fq)
            order, text = self.prompt(u, fq)
            finals.append((fq, order, text))
        if finals:
            probs.update(self.run_prompts(finals))
        st = collections.Counter()
        answers = {}
        for qkey, q in questions.items():
            a = core.answer_for(rec, qkey, q, probs, st)
            if a is None:
                raise RuntimeError(f"no prediction for question {qkey!r}")
            answers[qkey] = a
        return {"model": self.name, "answers": answers}, None
