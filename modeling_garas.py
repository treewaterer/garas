"""garas as a PyTorch model: `GarasForDecision`, a torch.nn.Module that answers multiple-choice decision questions with a probability
for every option.

    from modeling_garas import GarasForDecision
    model = GarasForDecision.from_pretrained()            # google/gemma-4-31B-it + plantwaterer/garas, bf16, on "cuda"
    answers = model.predict_proba(
        "The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
        {"plan": {"type": "choice", "instructions": "What should the organisers do?",
                  "criteria": {"play": "Play as planned", "indoors": "Move it indoors", "postpone": "Postpone it"}}})
    # {"plan": {"type": "choice", "choice": ..., "probabilities": {"play": ..., "indoors": ..., "postpone": ...}}}

Three levels of use:
  * `predict_proba(state, questions)`: a request (a state plus one or more named questions) -> an answer per question. This is
    the same computation as garas_engine.GarasEngine, the engine used for the Decision Index results.
  * `encode` / `encode_request` + `collate` + `forward`: build the exact prompts, batch them, and run the module yourself.
  * `forward(input_ids, attention_mask, option_token_ids, ...)`: option logits / log-probabilities / probabilities from the
    next-token logits at the last real token, and an optional cross-entropy loss over the options.

Questions follow the Decision Index request format: {"type": "choice", "instructions": ..., "criteria": {key: description}} or
{"type": "noul", "instructions": ...} (a yes / no question; the answer is p(yes)).

Inference procedure (as in garas_engine):
  * each question becomes one prompt (the model's chat template; options lettered A-Z a-z, at most 52 per prompt). A choice
    question's options are shown in one fixed non-identity order derived from a hash of (state, question key, option keys);
    yes / no keeps the order Yes, No;
  * one forward pass per prompt; the next-token logits of the option letters -> softmax (computed in float64 from the float32
    logits) -> mapped back to the options;
  * a choice with more than 52 options is split into ceil(K / 52) chunks (round 1); round 2 asks again over the top 52 // C options
    of every chunk; the answer puts (1 - 0.001) on the round-2 options and 0.001 on the rest, in proportion to round 1;
  * no truncation: a prompt that does not fit `max_len` tokens without cutting the state raises `Unsupported`;
  * the prompts of one request run in right-padded batches of at most `tokens_per_batch` padded tokens (default 32768, the
    setting used for the reported results). Batch composition affects bf16 rounding, so keep the defaults to reproduce them.
The adapter is loaded with autocast_adapter_dtype=False, so the LoRA update is computed in the base model's dtype (bf16), as in
training; peft's default would upcast it to fp32 and shift some letter logits by a bf16 step.

nn.Module behaviour: the wrapped transformers model is a registered submodule (`.model`), so `.to()`, `.eval()` / `.train()`,
`parameters()`, `state_dict()` / `load_state_dict()` behave as usual (the state dict holds the base weights plus the lora_A /
lora_B tensors). `from_pretrained` returns the module in eval mode with every parameter frozen. `forward` runs under
`torch.no_grad()` / `torch.inference_mode()`, and under `torch.compile` (its right-padding check is skipped while compiling;
use `dynamic=True` for prompts of varying length). Compiled kernels round bf16 differently, so compiled outputs are close to but
not bit-identical with eager mode; use eager mode (the default) to reproduce the reported results exactly.

Dependencies: torch, transformers, peft, numpy. Tested with torch 2.13, transformers 5.12.1, peft 0.21.2.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import garas_core as core                                          # noqa: E402

__all__ = ["GarasConfig", "GarasOutput", "EncodedQuestion", "GarasForDecision", "Unsupported"]


class Unsupported(ValueError):
    """A request the model does not answer: an unsupported question type, or a prompt that would not fit `max_len` tokens
    without cutting the state (nothing is truncated)."""


@dataclass
class GarasConfig:
    base: str = "google/gemma-4-31B-it"
    adapter: str = "plantwaterer/garas"
    max_len: int = 32768
    tokens_per_batch: int = 32768
    letters: str = core.LABELS                     # A-Z a-z: at most 52 options per prompt


@dataclass
class GarasOutput(ModelOutput):
    """`option_logits` [B, K]: next-token logits at the option letters (float32; -inf where `option_mask` is False);
    `log_probs` / `probs` [B, K]: their (log-)softmax over the options; `loss`: mean cross-entropy over the options when
    `labels` (the gold option's letter index, -100 to ignore) are given."""
    loss: Optional[torch.FloatTensor] = None
    option_logits: Optional[torch.FloatTensor] = None
    log_probs: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None


@dataclass
class EncodedQuestion:
    """One prompt. `order[j]` is the index (into `option_keys`) of the option shown under letter j."""
    qid: str
    text: str
    input_ids: list
    order: list
    option_keys: list
    kind: str
    key: str = field(default="")


def request_id(state, questions):
    return "eng-" + hashlib.sha1(json.dumps({"state": state, "questions": questions}, sort_keys=True, ensure_ascii=False,
                                            default=str).encode("utf-8")).hexdigest()[:16]


def question_seed(state_text, qid_key, keys):
    h = hashlib.sha256(json.dumps([state_text, qid_key, list(keys)], ensure_ascii=False).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def option_order(qtype, k, seed):
    """choice: one non-identity permutation drawn from a generator seeded by `seed`; any other type: the stored order."""
    if qtype != "choice" or k < 2:
        return list(range(k))
    rng = np.random.default_rng(seed)
    while True:
        o = [int(x) for x in rng.permutation(k)]
        if o != list(range(k)):
            return o


def _split_request(request_or_state, questions):
    if questions is None:
        if not isinstance(request_or_state, dict) or "questions" not in request_or_state:
            raise TypeError("pass (state, questions) or a request dict {'state': ..., 'questions': {...}}")
        return request_or_state.get("state"), request_or_state["questions"]
    return request_or_state, questions


class GarasForDecision(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, tokenizer, config: Optional[GarasConfig] = None, chat: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GarasConfig()
        self.chat = chat
        ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in self.config.letters]
        if any(len(x) != 1 for x in ids):
            raise ValueError("every option letter must be a single token")
        if tokenizer.pad_token_id is None:
            raise ValueError("the tokenizer needs a pad token (prompts are right-padded)")
        self.register_buffer("letter_token_ids", torch.tensor([x[0] for x in ids], dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, adapter: str = "plantwaterer/garas", base: str = "google/gemma-4-31B-it", dtype=torch.bfloat16,
                        device_map="cuda", max_len: int = 32768, tokens_per_batch: int = 32768, **model_kwargs):
        """Base model (`dtype`, `device_map`, extra `model_kwargs` go to the transformers loader) + the LoRA adapter, loaded with
        peft's PeftModel.from_pretrained(..., autocast_adapter_dtype=False). Returns the module in eval mode, all parameters frozen."""
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        mcfg = AutoConfig.from_pretrained(base)
        _, model_cls = core.model_loader(mcfg, "auto")
        tok = AutoTokenizer.from_pretrained(base)
        tok.padding_side = "right"
        m = model_cls.from_pretrained(base, dtype=dtype, device_map=device_map, **model_kwargs).eval()
        pm = PeftModel.from_pretrained(m, adapter, is_trainable=False, autocast_adapter_dtype=False)
        inner = pm.get_base_model().eval()          # the same module tree, LoRA layers injected in place
        for p in inner.parameters():
            p.requires_grad_(False)
        cfg = GarasConfig(base=base, adapter=str(adapter), max_len=int(max_len), tokens_per_batch=int(tokens_per_batch))
        return cls(inner, tok, cfg, chat=core.chat_preset(mcfg.model_type, False)).eval()

    @property
    def device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            return next(self.model.parameters()).device

    # ------------------------------------------------------------------------------------------------------------- forward
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, option_token_ids: torch.LongTensor,
                option_mask: Optional[torch.BoolTensor] = None, labels: Optional[torch.LongTensor] = None,
                return_dict: bool = True):
        """input_ids / attention_mask [B, T], RIGHT-padded; option_token_ids [B, K]: the token ids of the letters shown in each
        prompt (`collate` builds all four); option_mask [B, K]: True for real options; labels [B]: gold letter index or -100."""
        dev = self.device
        input_ids, attention_mask = input_ids.to(dev), attention_mask.to(dev)
        lens = attention_mask.sum(1)
        if not torch.compiler.is_compiling():
            pos = torch.arange(attention_mask.shape[1], device=dev)
            if not torch.equal(attention_mask, (pos[None, :] < lens[:, None]).to(attention_mask.dtype)) or bool((lens < 1).any()):
                raise ValueError("attention_mask must be right-padded (ones then zeros), with at least one token per row")
        last = lens - 1
        rows = torch.arange(input_ids.shape[0], device=dev)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=last)
        z = out.logits[rows, rows].float()                     # [B, V]: each row's next-token logits at its last real token
        option_logits = z.gather(1, option_token_ids.to(dev))
        if option_mask is not None:
            option_logits = option_logits.masked_fill(~option_mask.to(dev).bool(), float("-inf"))
        log_probs = torch.log_softmax(option_logits, dim=-1)
        probs = log_probs.exp()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(option_logits, labels.to(dev), ignore_index=-100)
        res = GarasOutput(loss=loss, option_logits=option_logits, log_probs=log_probs, probs=probs)
        return res if return_dict else res.to_tuple()

    # ------------------------------------------------------------------------------------------------------------- prompts
    def _unit(self, rec, qid):
        q = rec["questions"][qid]
        keys, texts, _ = core.question_options(q, rec["labels"][qid])
        return {"state": rec["state"], "instructions": q["instructions"], "qtype": q["type"], "option_keys": keys,
                "option_texts": texts, "noul_criteria": q.get("criteria") if q["type"] == "noul" else None}

    def _prompt(self, u, qid, max_len):
        order = option_order(u["qtype"], len(u["option_keys"]), question_seed(u["state"], qid.split("|", 1)[1], u["option_keys"]))
        text = core.build_prompt(self.tokenizer, u, order, max_len, chat=self.chat)
        if text is None or u["state"] not in text:
            raise Unsupported(f"maximum context length: question {qid.split('|', 1)[1]!r} does not fit {max_len} tokens "
                              f"without cutting the state (no truncation)")
        return order, text

    def _encoded(self, qid, u, order, text):
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return EncodedQuestion(qid=qid, text=text, input_ids=list(ids), order=list(order), option_keys=list(u["option_keys"]),
                               kind=u["qtype"], key=qid.split("|", 1)[1])

    def _prepare(self, state, questions, max_len):
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
            u = self._unit(rec, qid)
            order, text = self._prompt(u, qid, max_len)
            items.append(self._encoded(qid, u, order, text))
        return rec, items

    def encode_request(self, request_or_state, questions=None, max_len: Optional[int] = None) -> list:
        """The round-1 prompts of a request, in the order predict_proba runs them (a choice with more than 52 options gives one
        prompt per chunk; its round-2 prompt depends on round-1 results and is built inside predict_proba)."""
        state, questions = _split_request(request_or_state, questions)
        return self._prepare(state, questions, int(max_len or self.config.max_len))[1]

    def encode(self, state, question, options=None, kind: str = "choice", key: str = "answer",
               max_len: Optional[int] = None) -> EncodedQuestion:
        """One question -> one prompt. kind 'choice': `options` = {key: description} (or a list of option texts, each its own
        key); 'noul': a yes / no question (`options` ignored); 'score': `options` = the levels, shown in their given order (the
        model was evaluated on choice and yes / no questions; predict_proba reports score questions as Unsupported). `key` is the
        question's name; it enters the option-order hash."""
        max_len = int(max_len or self.config.max_len)
        if kind == "score":
            levels = list(options)
            keys, texts, _ = core.question_options({"type": "score", "levels": levels}, {k: 0.0 for k in levels})
            qid = f"score|{key}"
            u = {"state": core.render_state(state, "text"), "instructions": core.as_text(question), "qtype": "score",
                 "option_keys": keys, "option_texts": texts, "noul_criteria": None}
            order, text = self._prompt(u, qid, max_len)
            return self._encoded(qid, u, order, text)
        q = {"type": kind, "instructions": question}
        if kind == "choice":
            if isinstance(options, dict):
                q["criteria"] = options
            else:
                opts = [str(o) for o in options]
                if len(set(opts)) != len(opts):
                    raise ValueError("duplicate option texts: pass {key: description} instead")
                q["criteria"] = {o: o for o in opts}
        elif options is not None:
            q["criteria"] = options
        items = self.encode_request(state, {key: q}, max_len=max_len)
        if len(items) != 1:
            raise ValueError(f"{len(q.get('criteria') or {})} options need the two-round procedure: use predict_proba")
        return items[0]

    def collate(self, encoded: list) -> dict:
        """Right-padded batch: input_ids / attention_mask [B, T], option_token_ids / option_mask [B, K] (K = the most options)."""
        lens = [len(e.input_ids) for e in encoded]
        k_max = max(len(e.order) for e in encoded)
        input_ids = torch.full((len(encoded), max(lens)), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), max(lens)), dtype=torch.long)
        option_token_ids = torch.zeros((len(encoded), k_max), dtype=torch.long)
        option_mask = torch.zeros((len(encoded), k_max), dtype=torch.bool)
        letters = self.letter_token_ids.cpu()
        for b, e in enumerate(encoded):
            input_ids[b, : lens[b]] = torch.tensor(e.input_ids)
            attention_mask[b, : lens[b]] = 1
            k = len(e.order)
            option_token_ids[b, :k] = letters[:k]
            option_mask[b, :k] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "option_token_ids": option_token_ids,
                "option_mask": option_mask}

    # ------------------------------------------------------------------------------------------------------------- inference
    def batches(self, items: list, tokens_per_batch: Optional[int] = None) -> list:
        """Consecutive groups of prompts whose padded size (longest prompt x count) stays within `tokens_per_batch`
        (a single prompt longer than that runs alone)."""
        tpb = int(tokens_per_batch or self.config.tokens_per_batch)
        lens = [len(e.input_ids) for e in items]
        out, i = [], 0
        while i < len(items):
            j, mx = i, 0
            while j < len(items) and max(mx, lens[j]) * (j - i + 1) <= max(tpb, lens[i]):
                mx = max(mx, lens[j])
                j += 1
            out.append(items[i:j])
            i = j
        return out

    @torch.no_grad()
    def _run(self, items, tokens_per_batch, details):
        out = {}
        for group in self.batches(items, tokens_per_batch):
            z = self.forward(**self.collate(group)).option_logits.float().cpu().numpy()
            if details is not None:
                details["batches"].append([e.qid for e in group])
            for b, e in enumerate(group):
                k = len(e.order)
                zk = z[b, :k].astype(np.float32).astype(np.float64)
                logp = zk - zk.max() - np.log(np.exp(zk - zk.max()).sum())
                p = np.zeros(k)
                p[np.asarray(e.order)] = np.exp(logp)
                out[e.qid] = p.tolist()
                if details is not None:
                    details["option_logits"][e.qid] = z[b, :k].copy()
                    details["order"][e.qid] = list(e.order)
        return out

    @torch.no_grad()
    def predict_proba(self, request_or_state, questions=None, *, tokens_per_batch: Optional[int] = None,
                      max_len: Optional[int] = None, return_details: bool = False):
        """{question key: {"type": "choice", "choice": key, "probabilities": {key: p}} | {"type": "noul", "noul": p(yes)}}.
        With return_details=True also returns {"option_logits": {prompt id: float32 logits in letter order}, "order": ...,
        "batches": [[prompt ids], ...]}."""
        state, questions = _split_request(request_or_state, questions)
        max_len = int(max_len or self.config.max_len)
        details = {"option_logits": {}, "order": {}, "batches": []} if return_details else None
        rec, items = self._prepare(state, questions, max_len)
        probs = self._run(items, tokens_per_batch, details)
        finals = []
        for qkey in (rec["meta"].get("k52") or {}):
            keys = core.finalists(rec, qkey, probs)
            fq = core.qid_of(rec["id"], qkey, "final")
            q2, lab, kind = core.final_question(rec, qkey, keys)
            r2 = {**rec, "questions": {fq: q2}, "labels": {fq: lab}}
            u = self._unit(r2, fq)
            order, text = self._prompt(u, fq, max_len)
            finals.append(self._encoded(fq, u, order, text))
        if finals:
            probs.update(self._run(finals, tokens_per_batch, details))
        st = collections.Counter()
        answers = {}
        for qkey, q in questions.items():
            a = core.answer_for(rec, qkey, q, probs, st)
            if a is None:
                raise RuntimeError(f"no prediction for question {qkey!r}")
            answers[qkey] = a
        return (answers, details) if return_details else answers
