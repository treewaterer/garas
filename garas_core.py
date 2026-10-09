"""garas: inference helpers (prompt construction, request conversion, letter read-out, answer assembly).

Generated file; do not edit by hand."""
import contextlib
import hashlib
import json
import math
import random
import string
from collections import Counter
from pathlib import Path
import numpy as np
import torch


def question_options(q, labels):
    if q['type'] == 'choice':
        keys = list(q['criteria'])
        texts = [str(q['criteria'][k]) or str(k) for k in keys]
    elif q['type'] == 'noul':
        keys = ['yes', 'no']
        texts = ['Yes', 'No']
    elif q['type'] == 'score':
        keys = list(q['levels'])
        texts = [str(x) for x in keys]
    else:
        raise ValueError(f"unknown question type {q['type']!r}")
    lab = labels
    vals = []
    for i, k in enumerate(keys):
        if k in lab:
            vals.append(float(lab[k]))
        elif str(i) in lab:
            vals.append(float(lab[str(i)]))
        else:
            raise KeyError(f'no label for option {k!r} (index {i}); available keys: {sorted(lab)[:6]}')
    return (keys, texts, vals)


MAX_K = 52


K52_EPS = 0.001


BENCH = {1: ('01_bfcl', 'BFCL', 'tools', 'Apache-2.0 (gorilla repo)'), 2: ('02_toolret', 'ToolRet', 'tools', 'see ToolRet repo LICENSE; queries/tools aggregate upstream tool datasets with their own terms'), 3: ('03_apibank', 'API-Bank', 'tools', 'see DAMO-ConvAI api-bank/LICENSE'), 4: ('04_banking77', 'BANKING77', 'retrieval', 'CC BY 4.0'), 5: ('05_clinc150', 'CLINC150+OOS', 'retrieval', 'CC BY 3.0'), 6: ('06_routerbench', 'RouterBench', None, 'see withmartian/routerbench LICENSE'), 9: ('09_home_appliances', 'Home appliance simulator', 'tools', 'MIT (decision-index kit)'), 10: ('10_sgd', 'SGD/SGD-X', None, 'CC BY-SA 4.0'), 11: ('11_contractnli', 'ContractNLI', 'language', 'see stanfordnlp/contract-nli LICENSE and dataset terms'), 12: ('12_anli', 'ANLI', 'language', 'CC BY-NC 4.0 (non-commercial)'), 20: ('20_bpomp', 'BPoMP', 'arts', 'per Zenodo record 7299879'), 21: ('21_humicroedit', 'Humicroedit', 'arts', 'SemEval-2020 Task 7 terms'), 22: ('22_pop909', 'POP909-CL', 'arts', 'see POP909-CL repo LICENSE; POP909 underlying data research use only'), 23: ('23_cfcolor', 'cfcolor', 'arts', 'per the cfcolor project page'), 24: ('24_mmlu', 'MMLU', None, 'MIT'), 25: ('25_gpqa_diamond', 'GPQA Diamond', 'knowledge', 'CC BY 4.0; authors ask that questions are never posted in plain text: private copies only'), 26: ('26_arc_easy', 'ARC-Easy', None, 'CC BY-SA 4.0'), 27: ('27_arc_challenge', 'ARC-Challenge', None, 'CC BY-SA 4.0'), 28: ('28_winogrande', 'WinoGrande', 'language', 'see allenai/winogrande card'), 29: ('29_hellaswag', 'HellaSwag', 'language', 'MIT'), 30: ('30_gsm8k', 'GSM8K', 'knowledge', 'MIT'), 31: ('31_chessbench', 'ChessBench', 'knowledge', 'Apache-2.0 (searchless_chess code); see repo LICENSE'), 32: ('32_musr', 'MuSR', 'knowledge', 'MIT'), 33: ('33_sata_bench', 'SATA-Bench', 'knowledge', 'see sata-bench repo LICENSE'), 34: ('34_simplebench', 'SimpleBench', None, 'MIT'), 36: ('36_bright', 'BRIGHT', 'retrieval', 'CC BY 4.0'), 37: ('37_esci', 'Amazon ESCI', 'retrieval', 'Apache-2.0'), 38: ('38_acos', 'ACOS', 'language', 'not stated in the NUSTM/ACOS repository'), 39: ('39_finentity', 'FinEntity', 'language', 'not stated in the FinEntity repository'), 40: ('40_isarcasmeval', 'iSarcasmEval', 'language', 'see iSarcasmEval repo LICENSE'), 41: ('41_vast', 'VAST', 'language', 'not stated in the zero-shot-stance repository'), 42: ('42_nli4ct', 'NLI4CT', 'language', 'SemEval-2024 Task 2 terms'), 43: ('43_cruxeval', 'CRUXEval', 'knowledge', 'MIT'), 44: ('44_cladder', 'CLadder', 'knowledge', 'see causalNLP/cladder LICENSE'), 45: ('45_hle', 'HLE', 'knowledge', 'MIT per dataset card; gated access (terms accepted on the Hub); never in training corpora'), 48: ('48_forecastbench', 'ForecastBench', 'arts', 'see forecastbench-datasets LICENSE'), 50: ('50_habermas', 'Habermas Machine', 'arts', 'see google-deepmind/habermas_machine LICENSE'), 56: ('56_phishnchips', 'PhishNChips', 'retrieval', 'per PhishNChips SOURCE_LICENSES.md (project MIT; URL sources academic use only, with attribution)'), 57: ('57_mmlu_pro', 'MMLU-Pro', 'knowledge', 'MIT'), 58: ('58_bbh', 'BBH', 'knowledge', 'MIT; carries the BIG-bench canary: never in training corpora'), 59: ('59_ragtruth', 'RAGTruth', 'language', 'MIT annotations; contexts include MS MARCO (non-commercial) and the Yelp Open Dataset (no redistribution): evaluation only'), 61: ('61_hover', 'HoVer', 'retrieval', 'CC BY-SA 4.0'), 62: ('62_when2call', 'When2Call', 'tools', 'CC BY 4.0'), 64: ('64_newyorker', 'New Yorker', 'arts', 'CC BY 4.0 annotations')}


def dumps(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def as_text(x) -> str:
    if x is None:
        return ''
    return x if isinstance(x, str) else dumps(x)


def option_keys(q: dict) -> list[str]:
    if q['type'] == 'choice':
        return [str(k) for k in q['criteria']]
    if q['type'] == 'noul':
        return ['yes', 'no']
    raise ValueError(f"unsupported question type {q['type']!r}")


def our_question(q: dict, keys: list[str] | None=None) -> dict:
    t = q['type']
    out = {'type': t, 'instructions': as_text(q.get('instructions'))}
    if t == 'choice':
        crit = {str(k): as_text(v) for k, v in q['criteria'].items()}
        out['criteria'] = crit if keys is None else {k: crit[k] for k in keys}
    elif t == 'noul':
        if q.get('criteria') is not None:
            out['criteria'] = q['criteria']
    else:
        raise ValueError(f'unsupported question type {t!r}')
    return out


def label(q: dict, gold, keys: list[str] | None=None) -> tuple[dict, str]:
    ks = keys if keys is not None else option_keys(q)
    if gold is None:
        return ({k: 0.0 for k in ks}, 'none')
    if q['type'] == 'noul':
        g = 'yes' if gold is True or str(gold).lower() in ('1', 'true', 'yes') else 'no'
    else:
        g = str(gold)
    if keys is None and g not in ks:
        raise ValueError(f'gold {gold!r} is not an option')
    return ({k: 1.0 if k == g else 0.0 for k in ks}, 'hard' if g in ks else 'none')


def qid_of(run_id: str, qkey: str, part: str | None=None) -> str:
    return f'{run_id}|{qkey}' + (f'|{part}' if part else '')


def chunks_for(run_id: str, qkey: str, keys: list[str], max_k: int=MAX_K) -> list[list[str]]:
    c = math.ceil(len(keys) / max_k)
    order = list(keys)
    random.Random(int(hashlib.sha256(f'di-k52:{run_id}|{qkey}'.encode()).hexdigest()[:16], 16)).shuffle(order)
    pos = {k: i for i, k in enumerate(keys)}
    return [sorted(order[j::c], key=pos.__getitem__) for j in range(c)]


def finalists_per_chunk(n_chunks: int, max_k: int=MAX_K) -> int:
    return max_k // n_chunks


def render_state(state, fmt: str='text') -> str:
    if fmt == 'text' and isinstance(state, str):
        return state
    return dumps(state)


def convert_row(row: dict, canary: str | None, stats: dict, state_format: str='text') -> dict:
    e = row['_evaluation']
    n, rid = (int(e['catalog_id']), str(e['run_id']))
    if '|' in rid:
        raise ValueError(f"run id with '|': {rid}")
    slug, name, area, lic = BENCH[n]
    qs, labels, kinds, k52 = ({}, {}, {}, {})
    for qkey, q in row['questions'].items():
        if '|' in qkey:
            raise ValueError(f"question key with '|' in {rid}: {qkey!r}")
        keys = option_keys(q)
        gold = (row.get('expected') or {}).get(qkey)
        stats['k_hist'][len(keys)] += 1
        if q['type'] == 'choice' and len(keys) > MAX_K:
            parts = chunks_for(rid, qkey, keys)
            stats['k52_questions'] += 1
            for j, ck in enumerate(parts):
                qid = qid_of(rid, qkey, f'c{j}of{len(parts)}')
                qs[qid] = our_question(q, ck)
                labels[qid], kinds[qid] = label(q, gold, ck)
            k52[qkey] = {'K': len(keys), 'chunks': len(parts), 'finalists_per_chunk': finalists_per_chunk(len(parts)), 'keys': keys}
            stats['units'] += len(parts)
        else:
            qid = qid_of(rid, qkey)
            qs[qid] = our_question(q)
            labels[qid], kinds[qid] = label(q, gold)
            stats['units'] += 1
    meta = {'catalog_id': n, 'benchmark': name, 'run_id': rid, 'group_id': e.get('group_id'), 'track': e.get('track'), 'payload_sha256': e.get('payload_sha256'), 'area': area, 'in_index_021': area is not None, 'eval_only': True, 'builder': 'decision-index kit 0.2.1 (suite-0.2 rows)'}
    if k52:
        meta['k52'] = k52
    if n == 58 and canary:
        meta['canary'] = canary
    return {'id': rid, 'base_id': f"di/{n}/{e.get('group_id')}", 'source': f'decision_index/{slug}', 'license': f'EVALUATION ONLY. {lic}', 'state': render_state(row['state'], state_format), 'questions': qs, 'labels': labels, 'label_kind': kinds, 'meta': meta, 'split': 'eval'}


def chunk_qids(rec: dict, qkey: str) -> list[str]:
    info = rec['meta']['k52'][qkey]
    out = [qid_of(rec['id'], qkey, f"c{j}of{info['chunks']}") for j in range(info['chunks'])]
    missing = [q for q in out if q not in rec['questions']]
    if missing:
        raise ValueError(f"{rec['id']} {qkey}: chunk questions {missing} not in the record")
    return out


def finalists(rec: dict, qkey: str, probs: dict) -> list[str] | None:
    info, chosen = (rec['meta']['k52'][qkey], set())
    for qid in chunk_qids(rec, qkey):
        keys = list(rec['questions'][qid]['criteria'])
        p = probs.get(qid)
        if p is None or len(p) != len(keys):
            return None
        chosen.update((keys[i] for i in sorted(range(len(keys)), key=lambda i: (-p[i], i))[:info['finalists_per_chunk']]))
    return [k for k in info['keys'] if k in chosen]


def final_question(rec: dict, qkey: str, keys: list[str]) -> tuple[dict, dict, str]:
    parts = chunk_qids(rec, qkey)
    crit = {}
    for qid in parts:
        crit.update(rec['questions'][qid]['criteria'])
    gold = next((k for qid in parts for k, v in rec['labels'][qid].items() if v == 1.0), None)
    q = {'type': 'choice', 'instructions': rec['questions'][parts[0]]['instructions'], 'criteria': {k: crit[k] for k in keys}}
    return (q, {k: 1.0 if k == gold else 0.0 for k in keys}, 'hard' if gold in keys else 'none')


def norm(d: dict) -> dict:
    z = sum(d.values())
    if not z > 0 or not math.isfinite(z):
        return {k: 1.0 / len(d) for k in d}
    return {k: v / z for k, v in d.items()}


def answer_for(rec: dict, qkey: str, q: dict, probs: dict, stats: Counter):
    keys = option_keys(q)
    k52 = (rec.get('meta') or {}).get('k52', {}).get(qkey)
    if k52 is None:
        p = probs.get(qid_of(rec['id'], qkey))
        if p is None or len(p) != len(keys):
            return None
        dist = norm(dict(zip(keys, p)))
    else:
        parts = chunk_qids(rec, qkey)
        p1 = {}
        for qid in parts:
            ck = list(rec['questions'][qid]['criteria'])
            p = probs.get(qid)
            if p is None or len(p) != len(ck):
                return None
            p1.update({k: v / len(parts) for k, v in norm(dict(zip(ck, p))).items()})
        fin = finalists(rec, qkey, probs)
        p2 = probs.get(qid_of(rec['id'], qkey, 'final'))
        if fin is not None and p2 is not None and (len(p2) == len(fin)):
            p2 = norm(dict(zip(fin, p2)))
            rest = {k: v for k, v in p1.items() if k not in p2}
            zr = sum(rest.values()) or 1.0
            dist = {k: (1 - K52_EPS) * p2[k] if k in p2 else K52_EPS * rest[k] / zr for k in keys}
            stats['k52_round2'] += 1
        else:
            dist = {k: p1[k] for k in keys}
            stats['k52_round1_only'] += 1
        dist = norm(dist)
    if q['type'] == 'noul':
        return {'type': 'noul', 'noul': min(1.0, max(0.0, dist['yes']))}
    choice = max(keys, key=lambda k: (dist[k], -keys.index(k)))
    return {'type': 'choice', 'choice': choice, 'probabilities': dist}


LABELS = string.ascii_uppercase + string.ascii_lowercase


PROMPT = '{state}\n\nQuestion: {instructions}\n\nOptions:\n{options}\n\nAnswer with the letter of the single best option and nothing else.'


PROMPT_CASE = '{state}\n\nThis case has {n} questions:\n\n{listing}\n\nAnswer question {k}: {instructions}\n\nOptions:\n{options}\n\nAnswer with the letter of the single best option and nothing else.'


CASE_CRIT_WORDS = (None, 24, 8, 0)


def noul_definitions(crit):
    if not isinstance(crit, dict) or not crit:
        return None
    low = {str(k).strip().lower(): v for k, v in crit.items()}

    def side(names):
        return next((str(low[k]).strip() for k in names if low.get(k) is not None and str(low[k]).strip()), '')
    d = {'yes': side(('yes', 'true')), 'no': side(('no', 'false'))}
    return d if d['yes'] or d['no'] else None


def option_texts(q, noul_defs=False):
    texts = q['option_texts']
    if noul_defs and q['qtype'] == 'noul':
        d = noul_definitions(q.get('noul_criteria'))
        if d:
            texts = [f'{t}: {d[k]}' if d.get(k) else t for t, k in zip(texts, q['option_keys'])]
    return texts


def fixed_case_order(unit, seed=0):
    n = len(unit.get('case_questions') or ())
    order = list(range(n))
    h = int(hashlib.sha1(f"{seed}|{unit.get('case_id', '')}".encode()).hexdigest()[:16], 16)
    random.Random(h).shuffle(order)
    return order


def _clip_words(text, cap):
    w = str(text).split()
    if cap is None or len(w) <= cap:
        return ' '.join(w)
    return ' '.join(w[:cap]) + ' ...'


def case_listing(views, seq, tgt, order, tgt_texts, cap, noul_defs=False):
    lines = []
    for n, vi in enumerate(seq, 1):
        v = views[vi]
        lines.append(f"{n}. {_clip_words(v['instructions'], None)}")
        if cap != 0:
            texts, idx = (tgt_texts, order) if vi == tgt else (option_texts(v, noul_defs), range(len(v['option_texts'])))
            lines += [f'   - {_clip_words(texts[o], cap)}' for o in idx]
    return ('\n'.join(lines), seq.index(tgt) + 1)


def bos_prefix(tok):
    b = getattr(tok, 'bos_token', None)
    if not b or tok.bos_token_id is None:
        return ''
    ids = tok('x', add_special_tokens=True)['input_ids']
    return b if ids and ids[0] == tok.bos_token_id else ''


CHAT_PRESETS = {}


def chat_preset(model_type, plain=False):
    p = None if plain else CHAT_PRESETS.get(model_type)
    return json.loads(json.dumps(p)) if p else None


def model_loader(config, choice='auto'):
    import transformers
    from transformers.models.auto import modeling_auto as ma
    mt = getattr(config, 'model_type', None)
    if choice == 'image-text' or (choice == 'auto' and mt not in ma.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES and (mt in ma.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)):
        return ('image-text', transformers.AutoModelForImageTextToText)
    return ('causal-lm', transformers.AutoModelForCausalLM)


def build_prompt(tok, unit, order, max_len, margin=16, plain=False, whole_case=False, noul_defs=False, case_order=None, stats=None, chat=None):
    texts = option_texts(unit, noul_defs)
    opts = '\n'.join((f'{LABELS[j]}) {texts[o]}' for j, o in enumerate(order)))
    chat = chat or {}

    def render(state, case=None):
        if case is None:
            msg = PROMPT.format(state=state, instructions=unit['instructions'], options=opts)
        else:
            listing, k, n = case
            msg = PROMPT_CASE.format(state=state, n=n, listing=listing, k=k, instructions=unit['instructions'], options=opts)
        if plain:
            return bos_prefix(tok) + msg + '\n\nAnswer:'
        return tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False, **chat.get('chat_kwargs', {})) + chat.get('answer_prefix', '')

    def count(key):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1
    views = unit.get('case_questions') if whole_case else None
    if whole_case and views is None:
        raise ValueError('whole_case needs the units grouped first: attach_case_questions(units)')
    if whole_case and len(views) < 2:
        count('single_question')
    if views and len(views) > 1:
        tgt = unit['case_index']
        seq = list(case_order) if case_order is not None else fixed_case_order(unit)
        if sorted(seq) != list(range(len(views))):
            raise ValueError(f'case_order {seq} is not a permutation of the {len(views)} case questions')
        n_state = len(tok(unit['state'], add_special_tokens=False)['input_ids'])

        def fits(keep, cap):
            listing, k = case_listing(views, keep, tgt, order, texts, cap, noul_defs)
            over = len(tok(render('', (listing, k, len(keep))), add_special_tokens=False)['input_ids'])
            return (listing, k, len(keep)) if n_state <= max_len - over - margin else None
        for cap in CASE_CRIT_WORDS:
            case = fits(seq, cap)
            if case:
                count('case_full' if cap is None else 'case_criteria_cut')
                return render(unit['state'], case)
        keep = list(seq)
        while len(keep) > 2:
            del keep[max((i for i, vi in enumerate(keep) if vi != tgt))]
            case = fits(keep, CASE_CRIT_WORDS[-1])
            if case:
                count('case_questions_dropped')
                return render(unit['state'], case)
        count('case_all_dropped')
    overhead = len(tok(render(''), add_special_tokens=False)['input_ids'])
    budget = max_len - overhead - margin
    if budget <= 0:
        return None
    state = unit['state']
    enc = tok(state, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc['input_ids']) > budget:
        state = state[:int(enc['offset_mapping'][budget - 1][1])]
    return render(state)


def inner_model(model):
    inner = getattr(model, getattr(model, 'base_model_prefix', 'model') or 'model', None)
    if not isinstance(inner, torch.nn.Module) or inner is model:
        raise SystemExit('--exact-head fp32: no inner model under the LM head')
    return inner


def letter_head_fp32(model, h, label_ids):
    head = model.get_output_embeddings()
    z = h.float() @ head.weight[label_ids].float().T
    if getattr(head, 'bias', None) is not None:
        z = z + head.bias[label_ids].float()
    cfg = model.config.get_text_config() if hasattr(model.config, 'get_text_config') else model.config
    mult = getattr(cfg, 'output_multiplier', None)
    if mult:
        z = z * mult
    cap = getattr(cfg, 'final_logit_softcapping', None)
    return cap * torch.tanh(z / cap) if cap else z


def to_dev(data, device, h2d='sync', **kw):
    if h2d != 'async':
        return torch.tensor(data, device=device, **kw)
    return host_to_dev(torch.tensor(data, **kw), device)


def host_to_dev(t, device):
    if torch.device(device).type == 'cuda':
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


@contextlib.contextmanager
def head_input(model, keep, pick):
    if keep is None:
        yield
        return
    hook = model.get_output_embeddings().register_forward_pre_hook(lambda mod, args: keep.append(pick(args[0].detach())))
    try:
        yield
    finally:
        hook.remove()


def exact_letter_logits(model, tok, texts, label_ids, device, mode, head='model', pad_to=0, h2d='sync', keep_h=None):
    ids = [tok(t, add_special_tokens=False)['input_ids'] for t in texts]
    lens = [len(x) for x in ids]
    if mode == 'pack':
        if pad_to or h2d == 'async':
            flat, pos = ([i for x in ids for i in x], [p for n in lens for p in range(n)])
            extra = -len(flat) % pad_to if pad_to else 0
            if extra:
                flat += [tok.pad_token_id if tok.pad_token_id is not None else flat[-1]] * extra
                pos += list(range(extra))
            inp, pos = (to_dev([flat], device, h2d), to_dev([pos], device, h2d))
            last = to_dev(np.cumsum(lens) - 1, device, h2d)
        else:
            inp = torch.tensor([[i for x in ids for i in x]], device=device)
            pos = torch.tensor([[p for n in lens for p in range(n)]], device=device)
            last = torch.tensor(np.cumsum(lens) - 1, device=device)
        if head == 'fp32':
            h = inner_model(model)(input_ids=inp, position_ids=pos, use_cache=False).last_hidden_state[0, last]
            if keep_h is not None:
                keep_h.append(h.detach())
            return letter_head_fp32(model, h, label_ids)
        with head_input(model, keep_h, lambda x: x[0]):
            z = model(input_ids=inp, position_ids=pos, use_cache=False, logits_to_keep=last).logits[0]
        return z.float()[:, label_ids]
    if mode != 'pad':
        raise ValueError(f'exact batch mode {mode!r}')
    if tok.pad_token_id is None:
        raise SystemExit('--exact-batch pad needs a pad token')
    inp = torch.full((len(ids), max(lens)), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(ids), max(lens)), dtype=torch.long)
    for b, x in enumerate(ids):
        inp[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    last = to_dev([n - 1 for n in lens], device, h2d)
    rows = torch.arange(len(ids), device=device)
    if h2d == 'async':
        inp, att = (host_to_dev(inp, device), host_to_dev(att, device))
    if head == 'fp32':
        h = inner_model(model)(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False).last_hidden_state
        hl = h[rows, last]
        if keep_h is not None:
            keep_h.append(hl.detach())
        return letter_head_fp32(model, hl, label_ids)
    with head_input(model, keep_h, lambda x: x[rows, rows]):
        z = model(input_ids=inp.to(device), attention_mask=att.to(device), use_cache=False, logits_to_keep=last).logits
    return z[rows, rows].float()[:, label_ids]
