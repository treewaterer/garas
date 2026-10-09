---
license: apache-2.0
base_model: google/gemma-4-31B-it
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- peft
- decision-making
- multiple-choice
---

# garas

garas is a LoRA adapter for [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) that answers multiple-choice
decision questions with a probability for each option.

## Usage

```python
import string
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "google/gemma-4-31B-it"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cuda")
# autocast_adapter_dtype=False keeps the LoRA update in bf16, the arithmetic the adapter was trained and evaluated with
model = PeftModel.from_pretrained(base, "plantwaterer/garas", autocast_adapter_dtype=False).eval()

LETTERS = string.ascii_uppercase + string.ascii_lowercase      # up to 52 options


def option_probabilities(state, question, options):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    msg = (f"{state}\n\nQuestion: {question}\n\nOptions:\n{opts}\n\n"
           "Answer with the letter of the single best option and nothing else.")
    text = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    letter_ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in LETTERS[: len(options)]]
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    return torch.softmax(logits[letter_ids].float(), dim=-1).tolist()


print(option_probabilities("The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
                           "What should the organisers do?", ["Play as planned", "Move it indoors", "Postpone it"]))
```

Yes / no questions use the options `Yes` and `No` in that order.

### PyTorch model class

`modeling_garas.py` (with `garas_core.py`) defines `GarasForDecision`, a `torch.nn.Module` that wraps the base model and the
adapter. It loads the adapter with `autocast_adapter_dtype=False`, so the LoRA update stays in bf16, the arithmetic the adapter
was trained and evaluated with. `predict_proba` gives the same answers as the Decision Index engine below.

```python
import torch
from modeling_garas import GarasForDecision

model = GarasForDecision.from_pretrained("plantwaterer/garas", base="google/gemma-4-31B-it", device_map="cuda")

# a request: a state plus named questions; each answer has a probability for every option
answers = model.predict_proba(
    "The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
    {"plan": {"type": "choice", "instructions": "What should the organisers do?",
              "criteria": {"play": "Play as planned", "indoors": "Move it indoors", "postpone": "Postpone it"}},
     "rain": {"type": "noul", "instructions": "Is heavy rain likely?"}})
print(answers["plan"]["probabilities"], answers["rain"]["noul"])

# the same prompts, batched and run through forward()
enc = [model.encode("The match is outdoors.", "What should the organisers do?",
                    {"play": "Play", "postpone": "Postpone"}, key="plan"),
       model.encode("The invoice total is $1,240; the purchase order says $1,420.", "Is the invoice consistent?",
                    kind="noul", key="check")]
batch = model.collate(enc)                        # right-padded input_ids / attention_mask, option_token_ids / option_mask
with torch.no_grad():
    out = model(**batch)                          # option_logits, log_probs, probs: [batch, options] in letter order
for e, p in zip(enc, out.probs):
    print({e.option_keys[o]: float(p[j]) for j, o in enumerate(e.order)})   # letter j shows option e.order[j]
```

`forward(..., labels=...)` also returns the cross-entropy over the options, for a gold letter index per row (-100 to skip).
`torch.compile` works (use `dynamic=True`), but compiled kernels round bf16 differently, so use eager mode to reproduce the
reported numbers exactly.

### Decision Index engine

`garas_engine.py` (with `garas_core.py`) runs the adapter through the
[Decision Index kit](https://github.com/apolinario/decision-index)'s runner:

```bash
python -m decision_index run --engine garas_engine:GarasEngine \
  --option base=google/gemma-4-31B-it --option adapter=plantwaterer/garas \
  --option max_len=32768 --option tokens_per_batch=32768 --compact --out runs/garas
```

The inference procedure is as follows:
- **Prompts:** one prompt per question. A choice question's options are shown in one fixed non-identity order derived from a hash of the request.
- **Read-out:** one forward pass per prompt, with the option letters read off the next-token logits.
- **More than 52 options:** handled in two rounds.
- **No truncation:** a request that does not fit 32,768 tokens is reported as unsupported.
- **Batching:** the prompts of one request run in right-padded batches.

Dependencies are torch, transformers, peft and numpy. The engine was tested with transformers 5.12.1 and peft 0.21.2.

## Evaluation

**Decision Index 0.3, public panel (37 benchmarks).** The run used this repository's engine and adapter, with the kit's runner at
commit 9eb2dbe2, on 1 x GPU per shard. The full public suite ran as 8 row shards, merged without changes, followed by
one resume pass of the kit runner. This is our own run of the public kit, not a leaderboard entry.

| | public index |
|---|---|
| board rule (the 19 test items listed below counted as wrong) | 56.57 |
| as scored | 56.59 |

**Latency** was measured on 1 x RTX PRO 6000, one request at a time, on our own 760-row sample of the public suite (750 timed rows).
This is an approximation of the board's latency check, which uses a private sample:

| median | mean | 80th percentile |
|---|---|---|
| 114 ms | 509 ms | 364 ms |

### Overlap with evaluation data

Training used the train splits of BANKING77, CLINC150 and GSM8K. It also used WinoGrande train, Amazon ESCI train queries, and
Lichess puzzles (a different dataset from ChessBench's). No test split of any benchmark was used.

An audit of the training text against the Decision Index public suite found 19 test items that are exact or near-exact
duplicates of utterances in the BANKING77 and CLINC150 train splits. They are counted as wrong in the board-rule number above, as
the board does for train-split duplicates of test items (see the Jet v6.2 and lev entries under Contamination in the
[Decision Index methodology](https://multimodalart-jev-decision-index.static.hf.space/methodology.html)):
`4:BANKING77:BANKING77:test:` 554, 677, 727, 742, 747, 751, 754, 1432, 1735, 1754, 1993, 2081, 2211, 2440, 2644, 2651, 2987,
3070; `5:CLINC150+OOS:CLINC150+OOS:test:1591`.
The same audit found no overlapping items for the other sources named above.

## Licence

The adapter is released under Apache-2.0, and the base model, google/gemma-4-31B-it, is Apache-2.0. The training data come from the
sources below, under the licences and terms listed. Some of these sources carry use conditions that apply to the data itself; they
are repeated here: CourtListener (no Fair Credit Reporting Act uses), US Census PUMS (never use the data to identify any person or
household), NVD / US Department of Labor (not endorsed or certified), California DMHC (modified; not official government data) and
MITRE CVE / CWE (copyright notices below).

## Data attribution

garas was trained on decision questions built from the public data sources below, plus synthetic questions generated by the author
(no third-party data). Thanks to everyone who builds and maintains them. Records were reformatted, and many were modified (fields
selected, text clipped, identifiers masked), so none of the material should be read as an official copy of its source.

### Open data and public records

- **GDELT Project**: event data. Data from The GDELT Project, https://www.gdeltproject.org/.
- **Wikipedia**: article text from the English Wikipedia dump of 2023-11-01, via
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`, by Wikipedia contributors.
  Licensed CC BY-SA 3.0/4.0 and GFDL. https://en.wikipedia.org/
- **Wikipedia Articles for Deletion**: deletion debates by English Wikipedia contributors (CC BY-SA 3.0/4.0, GFDL), via the AfD
  corpus of Mayfield & Black (2019), https://github.com/emayfield/AFD_Decision_Corpus (the repository's code is GPL-3.0; the
  corpus files are distributed separately).
- **Wikidata**: structured data (CC0 1.0), and Requests-for-deletion discussions by the Wikidata contributors (CC BY-SA 4.0).
  https://www.wikidata.org/
- **Swiss Judgment Prediction XL**: [rcds/swiss_judgment_prediction_xl](https://huggingface.co/datasets/rcds/swiss_judgment_prediction_xl)
  (Rasiah et al., 2023), CC BY-SA 4.0. Decisions of the Swiss Federal Supreme Court, anonymised by the court.
- **US federal courts**: the Federal Judicial Center Integrated Database (public domain), and CourtListener bulk data from Free Law
  Project (Public Domain Mark 1.0), https://www.courtlistener.com/. Not for uses covered by the US Fair Credit Reporting Act.
- **State Supreme Court Data Project**: doi:10.7910/DVN/Z80F7P, CC0 1.0.
- **OSHA** Severe Injury Reports and enforcement data, US Department of Labor (public domain), https://www.osha.gov/severe-injury-reports
  and https://data.dol.gov/. Not endorsed or certified by the US Department of Labor.
- **NTSB** aviation accident database (public domain), https://data.ntsb.gov/avdata
- **NHTSA** Office of Defects Investigation investigations and complaints (public domain), https://static.nhtsa.gov/odi/ffdd/
- **openFDA** enforcement reports and device recalls, CC0 1.0, https://open.fda.gov/
- **ClinicalTrials.gov** (US National Library of Medicine), via the ClinicalTrials.gov API v2; data processed by ClinicalTrials.gov on
  2026-10-01. Modified: selected registration fields only, free text clipped, trial ids masked, study-status sentences removed. Some
  ClinicalTrials.gov data may be subject to third-party copyright outside the United States.
- **ClinVar** (NCBI): Landrum et al., Nucleic Acids Research 2018, PMID 29165669.
- **Orphanet** inheritance data, CC BY 4.0, https://www.orpha.net/
- **CIViC** (civicdb.org), CC0 1.0: Griffith et al., Nature Genetics 2017, PMID 28138153.
- **California Department of Managed Health Care**, "DMHC IMR Data, 2001 - Current", CC BY,
  https://data.ca.gov/dataset/independent-medical-review-imr-determinations-trend. Modified; not official government data.
- **NOAA / NWS Storm Prediction Center** text products (public domain), via the Iowa Environmental Mesonet, Iowa State University.
- **LIGO / Virgo / KAGRA** public alerts and the GWOSC event catalogue, CC BY 4.0, by the LIGO Scientific, Virgo and KAGRA
  Collaborations. This model uses data obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO
  Scientific Collaboration, the Virgo Collaboration and KAGRA.
- **USGS** PAGER and ComCat (public domain). Population exposure is from ORNL LandScan (CC BY 4.0).
- **National Vulnerability Database** (NIST, public domain) and **CVE** records (CVE Terms of Use; Copyright The MITRE Corporation).
  **CWE**: Copyright 2006-2026, The MITRE Corporation; CWE is a trademark of The MITRE Corporation. This product uses the NVD API
  but is not endorsed or certified by the NVD.
- **American Community Survey PUMS**, US Census Bureau (CC0 1.0), with task definitions from
  [folktables](https://github.com/socialfoundations/folktables) (Ding et al., 2021; MIT). Never to be used to identify any person or
  household.
- **OpenReview** reviews and scores, CC BY 4.0 (titles and abstracts: CC0 1.0), https://openreview.net/
- **Lichess** open database, CC0 1.0, https://database.lichess.org/
- **Amazon ESCI** shopping queries, [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Reddy et al., 2022),
  Apache-2.0. Copyright Amazon.com, Inc. or its affiliates.
- **AIDev**: [hao-li/AIDev](https://huggingface.co/datasets/hao-li/AIDev) (Li et al., 2025), CC BY 4.0; records from permissively
  licensed repositories only.
- **SWE-rebench**: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
  (Trofimova et al., 2025), CC BY 4.0.
- **Open-Jev**: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), CC0 1.0.

### Research datasets

| dataset | reference | licence |
|---|---|---|
| BANKING77 | Casanueva et al., 2020 | CC BY 4.0 |
| CLINC150 | Larson et al., 2019 | CC BY 3.0 |
| GoEmotions | Demszky et al., 2020 | Apache-2.0 |
| GSM8K | Cobbe et al., 2021 | MIT |
| HelpSteer2 | Wang et al., 2024 | CC BY 4.0 |
| Measuring Hate Speech | Kennedy et al., 2020 | CC BY 4.0 |
| IBM Debater ArgQ-Rank-30kArgs | Gretz et al., 2019 | CC BY 3.0 |
| BBQ | Parrish et al., 2022 | CC BY 4.0 |
| CaSiNo | Chawla et al., 2021 | CC BY 4.0 |
| Circa | Louis et al., 2020 | CC BY 4.0 (dataset card; its README links CC BY-SA 4.0) |
| CodeContests | Li et al., 2022 | CC BY 4.0 |
| CosmosQA | Huang et al., 2019 | CC BY 4.0 |
| CommonsenseQA | Talmor et al., 2019 | MIT |
| e-CARE | Du et al., 2022 | MIT |
| Social IQa | Sap et al., 2019 | CC BY 4.0 |
| WinoGrande | Sakaguchi et al., 2019 | CC BY (code Apache-2.0) |
| MedMCQA | Pal et al., 2022 | Apache-2.0 |
| PubMedQA | Jin et al., 2019 | MIT |
| Stanford Politeness Corpus, via ConvoKit | Danescu-Niculescu-Mizil et al., 2013 | CC BY 4.0 |
| Scruples | Lourie et al., 2021 | Apache-2.0 |
| TabFact | Chen et al., 2020 | CC BY 4.0 |

### Community synthetic datasets

- [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  (tasksource, Sileo, 2024), Apache-2.0
- [frontier-infra/jebadiah-synth-v2](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2), Apache-2.0
- [kaivoss/system-one-270m-data](https://huggingface.co/datasets/kaivoss/system-one-270m-data), Apache-2.0
- [vagmi/jevlite_dataset](https://huggingface.co/datasets/vagmi/jevlite_dataset), CC BY-SA 4.0
- [mghafiri/decision-model-scenarios](https://huggingface.co/datasets/mghafiri/decision-model-scenarios), MIT
