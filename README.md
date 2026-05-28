# Post-Generation Citations — React Edition

A **post-hoc citation system** for Large Language Models, built as a **React + FastAPI** web application with integrated **interpretability tooling** for the NLI component.

Given an LLM response, the system decomposes it into atomic claims, attributes each claim to its supporting passage using either an NLI cross-encoder or an LLM re-ranker, and inserts inline citations — **all after generation**.

This is a full rewrite of the original Streamlit prototype, with a proper REST backend, a reactive frontend, and dedicated pages for inspecting the NLI model's decisions (attention flow, Integrated Gradients, Activation Patching).

> **Bachelor's thesis** · Università di Milano-Bicocca · Post-Generation Citations
> Paradigm: **P-Cite** (post-hoc citation), following [Saxena et al. (2025)](https://arxiv.org/abs/2509.21557). Architecture inspired by CEG ([Li et al., ACL 2024](https://aclanthology.org/2024.acl-long.619/)).

---

## What the thesis measures

The contribution of the thesis is **not** to propose yet another RAG system. The contribution is to **measure how hard claim attribution actually is**, isolating it from retrieval and generation quality. The pipeline serves this measurement goal: it takes the upstream RAG output as given (oracle or near-oracle passages) and stresses only the claim → passage matching step.

Two attribution methods are compared as **judges**:

- **NLI** — `cross-encoder/nli-deberta-v3-large`, sentence-level entailment.
- **LLM** — Claude as a re-ranker + evidence extractor.

The comparison is run on three question types from the ALCE benchmark (ASQA, QAMPARI, ELI5), evaluated through a custom annotated dataset (**ALCE+**, an extension of ALCE with manually annotated *nuggets*) and a set of nugget-based metrics (continuous precision, recall, coverage; split into required vs optional). A complementary **LLM-as-judge** evaluation (DeepSeek) is run alongside Nugget to provide an annotation-free signal.

> Headline finding so far: LLM beats NLI on all three question types, but the gap depends sharply on question type — about +11 points of nugget precision on factoid (ASQA), +10 on long-form (ELI5), and **+29 on multi-answer (QAMPARI)**. The qualitative DeBERTa analysis is consistent with this pattern.

---

## Pipeline

```
Query ──► LLM ──► Raw Response ──► Decompose ──► Attribute ──► Cite ──► Cited Response
         (Claude)                  (atomic       (NLI or LLM   ([1][2])
                                    claims)       as judge)
```

### Steps

1. **Generate** — Claude (or any LLM via the unified client) produces a response to the query, optionally conditioned on a set of passages. No citations at this stage.
2. **Decompose** — The response is broken into atomic claims, FActScore-style ([Min et al., 2023](https://arxiv.org/abs/2305.14627)). The prompt explicitly blacklists meta-claims like *"the passage states X"* to avoid degenerate decompositions.
3. **Attribute** — For each claim, the chosen judge (NLI or LLM) scores the candidate passages and returns those that support the claim. Both methods produce a sentence-level evidence span.
4. **Cite** — Inline markers `[n]` are inserted after the sentences containing the supported claims; a deduplicated reference list is appended. A second LLM call maps each claim to the response sentences it derives from (`claim_source_map`), removing the fuzzy-overlap heuristic that previously made twin sentences steal each other's citations.
5. **Evaluate** — Two evaluation modes are available, switchable from the UI and runnable independently:
   - **Nugget** (vs annotated ground truth) — continuous precision, recall, coverage. Main evaluation surface used for the thesis.
   - **DeepSeek** (LLM-as-judge, three-level) — annotation-free; complementary signal for cases without nugget annotation and for cross-check.

Steps 1 and 2 use the configured LLM. Step 3 runs locally with DeBERTa (NLI mode) or calls the LLM again (LLM mode), parallelised over claims via an async semaphore. Step 4 uses the configured LLM for source mapping. Step 5 is either pure embedding (Nugget) or another LLM call (DeepSeek).

See [`docs/Metriche_Nugget_DeepSeek.pdf`](docs/Metriche_Nugget_DeepSeek.pdf) for the formal definitions of all metrics.

---

## Architecture

### Backend — `backend/` (FastAPI + Uvicorn)

```
backend/
├── main.py                       # FastAPI app + CORS + router registration
├── requirements.txt
├── .env                          # ANTHROPIC_API_KEY etc. (git-ignored)
│
├── core/                         # Pipeline primitives (framework-agnostic)
│   ├── __init__.py
│   ├── generate.py               # Step 1: generation (CLI + functions)
│   ├── decompose.py              # Step 2: atomic claim extraction
│   ├── retrieve.py               # Step 3: NLI and LLM attribution (async)
│   ├── cite.py                   # Step 4: citation insertion + reference list
│   ├── claim_source_map.py       # Step 4 helper: LLM claim→sentence mapping
│   ├── nuggets_evaluate.py       # Step 5a: ALCE+ nugget metrics (continuous)
│   ├── deepseek_evaluate.py      # Step 5b: LLM-as-judge (tri-level), async
│   ├── llm_client.py             # Unified Anthropic / Gemini / Ollama client
│   ├── noise.py                  # Distractor injection for RAG-like setting
│   ├── interpretability.py       # IG + Activation Patching on DeBERTa
│   └── pipeline_runners.py       # High-level orchestration used by routers
│
├── routers/                      # REST endpoints (thin wrappers over core/)
│   ├── pipeline.py               # /api/pipeline/{generate,decompose,retrieve,cite,evaluate_*,...}
│   ├── nli.py                    # /api/nli/predict, /api/nli/batch
│   ├── interpret.py              # /api/interpret/{attention,ig,patching}
│   └── dataset.py                # /api/dataset/{list,load}
│
├── data/
│   └── alce_plus/                # ALCE+ annotated dataset (gitignored)
├── cache/                        # Hugging Face model cache (gitignored)
└── results/                      # Experiment outputs (gitignored)
```

**Key design decisions:**

- **Router/core split.** `core/` is pure Python, no FastAPI imports — can be run as a standalone script, imported from a notebook, or wrapped by any framework. `routers/` only handles request validation (Pydantic), error mapping, and delegation.
- **Lazy model loading with `lru_cache(maxsize=1)`.** First call to an NLI endpoint pays the 30–60s DeBERTa-large loading cost; subsequent calls are fast. Same pattern for the BGE pre-filter and the MiniLM embedder. One instance per model type lives in memory for the lifetime of the process.
- **Stateless endpoints.** Each pipeline step takes all its inputs as JSON and returns all its outputs as JSON. No hidden server state — makes retry, caching, and reproducibility trivial.
- **Async retrieval & judging.** Retrieval (LLM mode) and DeepSeek judging fan out per claim with `asyncio.gather` + a `Semaphore(8)` to bound concurrency, keeping the wall-clock per example close to the slowest claim instead of the sum.
- **Dev server on port `:8000`**, interactive Swagger docs at `/docs`.

### Frontend — `frontend/` (React + Vite)

```
frontend/
├── package.json
├── vite.config.js                # Dev server :5173 + proxy /api → :8000
├── index.html
└── src/
    ├── main.jsx                  # React Router bootstrap
    ├── App.jsx                   # Shell: Sidebar + TopBar + <Outlet/>
    ├── api.js                    # fetch() wrappers — single source of truth
    ├── index.css                 # Global styles (custom, no Tailwind)
    │
    ├── components/
    │   ├── Sidebar.jsx
    │   ├── StepCard.jsx
    │   ├── ScorePill.jsx
    │   ├── MetricCard.jsx
    │   ├── MetricsViews.jsx      # Shared Nugget/DeepSeek detail views
    │   └── Icon.jsx              # Lucide icons wrapper
    │
    └── pages/
        ├── Demo.jsx              # Production-like single-shot query → cited answer
        ├── Pipeline.jsx          # Interactive end-to-end pipeline (6 steps, per-step rerun)
        ├── EvaluateDataset.jsx   # Batch evaluation on a full dataset
        ├── Explore.jsx           # Browse generated / cited results (incl. metrics tab)
        ├── Metrics.jsx           # Aggregated metrics across saved runs
        ├── Attention.jsx         # NLI attention flow inspection
        └── Interpretability.jsx  # IG + Activation Patching
```

**Key design decisions:**

- **Vite, not CRA.** Faster dev server, lighter, modern standard.
- **React Router v6.** Declarative routing, `<NavLink>` for automatic active state.
- **Single `api.js` module.** Every backend call goes through a typed wrapper (`api.pipeline.generate(...)`, `api.nli(...)`). Components never touch `fetch` or URLs directly.
- **Vite proxy in dev.** `/api/*` from the frontend is transparently forwarded to `localhost:8000` — no CORS pain in development. Production uses standard CORS.
- **Shared metric views in `components/MetricsViews.jsx`.** `NuggetMetricsView` and `DeepSeekMetricsView` are exported from a single module and imported by both Pipeline (Step 6) and Explore (Metriche tab). Single source of truth for tri-level styling and continuous-precision formatting.
- **No global state manager.** `useState` + Context API (`AppData`) for the few cross-page concerns (current dataset, saved pipeline results). Redux/Zustand would be overkill at this scope. Persistence is intentionally in-memory: refresh = clean slate (a "Carica JSON" button in Metrics/Explore reloads previously exported results).

---

## Attribution methods

The thesis compares **two** attribution methods exposed as user choices:

### NLI (DeBERTa)

`cross-encoder/nli-deberta-v3-large` via `sentence-transformers`. For each claim:

1. All passages are split into sentences (with abbreviation-aware regex + span recovery).
2. **Optional pre-filter** — when `pre_filter_k > 0`, sentences are scored with a hybrid `0.5 × cosine_sim(BGE) + 0.5 × jaccard_overlap` and only the top-K go through NLI. Useful on large pools, no-op on the typical ALCE+ pool of 5 passages.
3. The cross-encoder scores each `(sentence, claim)` pair. Softmax is computed in a numerically stable way (`exp(x - max)`).
4. **Passage-level aggregation is max-over-sentences**: a passage's score is the highest entailment among its sentences.
5. Passages with score `≥ threshold` (default 0.5) are kept, sorted, and the top-K returned.

### LLM (Claude)

Two-stage pre-filter + re-ranking:

1. A lightweight internal helper (MiniLM cosine similarity) restricts the candidate pool to the top-10 passages most similar to the claim. This is an **implementation detail**, not a user-selectable attribution method.
2. The candidates are passed to Claude with a few-shot prompt asking it to decide, for each passage, whether it `supports` / `contradicts` / is `neutral` to the claim, and to copy the exact supporting sentence as evidence.
3. The output is parsed as JSON; passages labelled `supports` above the threshold are kept.

> **A note on similarity-based matching.** Embedding similarity alone is *not* a standalone attribution method in this system: it measures semantic closeness, not entailment, and is therefore not comparable to NLI/LLM as a citation judge. It still appears inside the codebase as a building block (sentence pre-filter for NLI, passage pre-filter for LLM), but is not exposed in the UI.

---

## ALCE+ — the evaluation dataset

The thesis uses **ALCE+**, an extension of ALCE (Gao et al., EMNLP 2023) built specifically to evaluate claim attribution as a component, not the whole RAG system. The extensions are:

- **Cropping to 5 passages per question** for tractability — the original ALCE provides ~100 passages per question, which is overkill for measuring attribution quality.
- **Manual nugget annotation.** For each question, a list of atomic facts is annotated by hand, each marked `required: true` (essential to a complete answer) or `required: false` (useful but not essential), with a `golden_passage_title`, a `golden_evidence` span, and **keywords** used for the lexical gate.
- **Distractor support.** Each passage is marked `is_gold` or, optionally, `is_noise` if injected from another question's pool by `core/noise.py`.

Three question types, 30 examples each (ASQA, QAMPARI, ELI5), annotated by a single annotator.

### Nugget matching — covering and precision (continuous)

A nugget `g` is **covered** by a claim `c` only if both of these hold (logical AND):

- **Lexical gate** — at least one keyword of `g` appears (case-insensitive substring) in `c`.
- **Hybrid threshold** — the match-score `0.2·lex(g, c) + 0.8·cos(g, c) ≥ 0.6`, where `lex` is content-word overlap and `cos` is cosine similarity on `all-MiniLM-L6-v2` embeddings.

Both gates are necessary, neither sufficient alone. A nugget can be covered by multiple claims; a claim can cover multiple nuggets.

For every covering claim, an **evidence score** `0.2·lex(e*, ê) + 0.8·cos(e*, ê)` compares the claim's extracted evidence `ê` against the gold evidence `e*(g)`. The **continuous precision** of a nugget is then the average evidence score across its covering claims, weighted by how strongly each claim covers the nugget (its match-score). Marginal claims at the threshold pull the average down only weakly; on-topic claims dominate.

See [`docs/Metriche_Nugget_DeepSeek.pdf`](docs/Metriche_Nugget_DeepSeek.pdf) §1 for the full formulas.

> **Why nugget matching avoids NLI.** The covering decision is made by lexical+semantic similarity, not by NLI. This avoids the circularity of *"NLI judging NLI"* — we cannot fairly evaluate the NLI judge using NLI as the matching oracle.

---

## Evaluation

Two evaluation modes are exposed, both per-example (Pipeline page) and batch (EvaluateDataset page). Aggregates across the dataset are **pooled (micro)** — see the PDF §1.6 and §2.4 — so each example contributes proportionally to its number of nuggets / pairs, not equally.

### Nugget (ALCE+) — continuous, vs ground truth

| Metric | What it measures | Formula |
|---|---|---|
| **Nugget Precision** | On the covered nuggets, how well the cited evidence matches the gold. | `Σ_g precision(g) / n_covered` |
| **Nugget Recall** | Same numerator, diluted on the total — penalises uncovered nuggets. | `Σ_g precision(g) / n_total` |
| **Nugget Coverage** | Fraction of nuggets mentioned by at least one claim. | `n_covered / n_total` |

All three are also reported separately on the **required** and **optional** subsets. The precision is **continuous**, not binary: a nugget whose evidence is close but imperfect contributes a partial score rather than a 0/1.

### DeepSeek (LLM-as-judge) — three-level, no ground truth

A separate LLM (DeepSeek) judges each `(claim, evidence)` pair on a three-level scale:

- **supported** — the evidence fully establishes the claim (1.0).
- **partial** — the evidence touches the claim but doesn't fully establish it: covers only part of it, is too generic, or misses a key detail (0.5).
- **not_supported** — the evidence does not support the claim (0.0).

| Metric | What it measures | Formula |
|---|---|---|
| **Citation Precision** | Weighted fraction of pairs the judge accepts. | `(n_full + 0.5·n_partial) / n_pairs` |
| **Citation Recall** | Fraction of claims with at least one full or partial evidence. | `|{c : ∃ê v(c,ê) ∈ {full, partial}}| / n_claims` |
| **Distribution** | How verdicts split across the three levels. | `pct_full`, `pct_partial`, `pct_none` |

Pairs with empty extracted spans count as `not_supported` without being sent to the model; API errors fail safe to `not_supported`. See PDF §2.

### A note on comparison

The two modes are **not** 1:1 comparable. Nugget gives a continuous score against a hand-annotated gold; DeepSeek gives a three-level verdict from a stochastic judge. They are designed as complementary lenses: Nugget is objective but bound to the annotation, DeepSeek is flexible but subject to judge variance. Numbers from binary-DeepSeek runs (pre-tri-level) are not directly comparable to tri-level runs — the recall in particular rises because partial evidence now counts as covering.

---

## The seven pages

### 1. Demo — production-like single-shot

The "front door" of the app. Load a dataset, pick a question, get a cited answer. The page hides the six steps and runs them end-to-end (`generate → decompose → retrieve → cite`) with a static progress indicator. The output is a clickable cited response: clicking a highlighted sentence opens the references that support it, with the exact evidence span highlighted inside each passage. No claim layer is exposed — you go straight from sentence to source. Settings (model, retrieval method, top-k, noise) are accessible from a collapsible panel.

### 2. Pipeline — end-to-end interactive run

The research workbench. Load a dataset (ALCE+, ELI5, QAMPARI), pick an example, run **each step manually**. The page exposes every intermediate artifact: raw response, claims, matched passages with sentence-level evidence, cited output with inline `[n]` markers and a reference panel. Each step (Generate / Decompose / Retrieve / Cite / Evaluate) has its own **rerun button** that re-executes only that step and clears the results of the steps below it. A debug view shows the top-4 sentences per passage with their NLI scores for any selected claim. Step 6 (Evaluate) has a Nugget / DeepSeek toggle.

### 3. EvaluateDataset — batch evaluation

Run the full pipeline on every example of a loaded dataset and aggregate the metrics. Supports both **Nugget** and **DeepSeek** evaluation modes (Standard ALCE-style is no longer in the toggle). For Nugget, includes a nugget-association review table that lists every nugget × claim × evidence triple with status (covered / cited / cited-from-noise) and configurable filters. Per-example top-1 evidence and full top-k are reported side-by-side via a toggle.

### 4. Explore — browse past runs

Load any saved pipeline result or upload an `pipeline_results.json` exported earlier. For each example: tabs for **Risposta grezza**, **Claims**, **Matched** (with click-to-expand passages and entailment scores), **Citata** (cited response + references list), and **Metriche** (full per-nugget / per-claim breakdown via the shared metric views). Each result can be saved into the in-session store from the Pipeline page via "Salva in Esplora".

### 5. Metrics — aggregated metrics across runs

Pools the metrics of every saved result (and any results loaded from a previously exported JSON) and shows them per family. A toggle switches between **Nugget** and **DeepSeek**; the family without data is greyed out. Aggregation is **pooled / micro**: precision and recall sum counts across examples and recompute the ratio (per PDF §1.6 / §2.4). For DeepSeek, the full / partial / none distribution is shown as a separate row of cards. A bar chart and a per-example table let you spot outliers; "Carica risultati" loads a previously downloaded JSON, "Scarica JSON" exports an aggregated snapshot that round-trips with itself.

### 6. Attention — NLI attention flow

For any `(premise, hypothesis)` pair, compute attention flow from `[CLS]` and visualize:

- `hyp_dominance` — how much attention flow ends on `H` vs `P+H`.
- Token-to-token heatmap for any selected layer and head.

### 7. Interpretability — IG + Activation Patching

- **Integrated Gradients** (Captum, 50 steps, baseline = `[PAD]` everywhere except `[CLS]`/`[SEP]`): per-token attribution toward the entailment logit. Optionally layer-wise (24 layers for DeBERTa-large).
- **Activation Patching** on the residual stream: given a clean / corrupt pair of equal length, each `(layer, position)` activation from corrupt is patched into clean and the resulting shift in entailment is recorded. Heatmap over 24 layers × positions.

Both endpoints stream progress for the long-running patching loop (24 × seq_len forward passes).

---

## On the NLI judge and its biases

The thesis surfaced a systematic failure mode of `cross-encoder/nli-deberta-v3-large` when used as a citation judge: on certain premise/hypothesis pairs where `P` does not entail `H`, the model predicts `entailment` with very high confidence. Canonical example:

```
P: Italian tenor Andrea Bocelli performed a stunning rendition of Nessun Dorma
   at the closing ceremony.
H: The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.

→ E = 0.949    (should be neutral)
```

**Status of the explanation.** The **attestation bias** hypothesis (McKenna et al., EMNLP 2023) has been **falsified** as the explanation for this behaviour: attention flow analysis shows `[CLS]` attends to `P` as much as to `H` in biased cases — inconsistent with the classical attestation-bias signature. A **structural / register-sensitivity** hypothesis is under causal validation via Activation Patching, with preliminary evidence localising the effect to a small number of token positions in layers 0–14.

This part of the work feeds into the qualitative chapter of the thesis (Chapter 6 in the current outline). It is **not** a standalone mitigation method — there is no NLI-debiasing module shipped with this system. Mitigation is explicitly future work.

---

## Endpoints (reference)

### Pipeline

- `POST /api/pipeline/generate` — `{query, model, passages?}` → `{response}`
- `POST /api/pipeline/decompose` — `{text, model}` → `{claims: [...]}`
- `POST /api/pipeline/retrieve` — `{claims, passages, method, threshold, top_k, ...}` → `{matched, debug}`. Async fan-out per claim with bounded concurrency.
- `POST /api/pipeline/retrieve-single` — same contract but **one claim per call**; used by the Pipeline UI to drive the per-claim progress bar.
- `POST /api/pipeline/cite` — `{response, matched, model?}` → `{cited_response, references}`. Uses an LLM call internally to map claims to source sentences.
- `POST /api/pipeline/evaluate_nuggets` — `{matched_claims, nuggets, nugget_covering?}` → continuous Nugget metrics + `per_nugget` breakdown.
- `POST /api/pipeline/evaluate_deepseek` — `{matched}` → tri-level DeepSeek metrics + `per_claim` breakdown.
- `POST /api/pipeline/evaluate_example` — full pipeline over one example, used by EvaluateDataset; returns both top-k and top-1 evidence variants.

### NLI

- `POST /api/nli/predict` — `{premise, hypothesis}` → `{entailment, contradiction, neutral}`
- `POST /api/nli/batch` — list of pairs → list of scores.

### Interpretability

- `POST /api/interpret/attention` — `{premise, hypothesis, layer?, head?}` → attention matrix + `hyp_dominance`.
- `POST /api/interpret/ig` — `{premise, hypothesis, target, steps, layerwise?}` → per-token (and per-layer) attribution.
- `POST /api/interpret/patching` — `{clean_premise, clean_hypothesis, corrupt_premise, corrupt_hypothesis}` → `{clean_entailment, corrupt_entailment, patching_effect, num_layers, seq_len}`.

### Dataset

- `GET /api/dataset/list` — available ALCE+ files.
- `POST /api/dataset/load` — `{name, subset?}` → examples.

Full interactive docs at `http://localhost:8000/docs`.

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- An Anthropic API key (for Claude generation and LLM attribution)
- A DeepSeek API key (for the LLM-as-judge evaluation mode) — optional if you only use Nugget mode
- *Optional:* Ollama running locally if you want to use Gemma / Llama / Phi as generator

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
echo "DEEPSEEK_API_KEY=sk-..."     >> .env   # only needed for DeepSeek eval mode

uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to verify.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`.

### Data

```bash
bash scripts/download_data.sh   # Fetches ALCE+ into backend/data/alce_plus/
```

The first call to any NLI endpoint downloads `cross-encoder/nli-deberta-v3-large` (~1.4 GB) into `backend/cache/`.
The first call to the LLM attribution endpoint additionally downloads `BAAI/bge-base-en-v1.5` and `all-MiniLM-L6-v2` for internal pre-filtering.

---

## References

- **Saxena, A. et al.** (2025). *Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution*. — G-Cite / P-Cite taxonomy.
- **Li, W. et al.** (2024). *Citation-Enhanced Generation for LLM-based Chatbots*. ACL 2024. — closest architecture in the literature.
- **Gao, T. et al.** (2023). *Enabling Large Language Models to Generate Text with Citations*. EMNLP 2023. — ALCE benchmark.
- **Min, S. et al.** (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. EMNLP 2023.
- **McKenna, N. et al.** (2023). *Sources of Hallucination by LLMs on Inference Tasks*. Findings of EMNLP 2023. — attestation bias, *falsified in this thesis* for the observed case.
- **McCoy, R. T., Pavlick, E., Linzen, T.** (2019). *Right for the Wrong Reasons*. ACL. — HANS, parallel shortcut-learning result on BERT.
- **Geirhos, R. et al.** (2020). *Shortcut Learning in Deep Neural Networks*. Nature MI. — framing.
- **Sundararajan, M., Taly, A., Yan, Q.** (2017). *Axiomatic Attribution for Deep Networks*. ICML. — Integrated Gradients.
- **Meng, K. et al.** (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS. — Activation Patching.
- **Vig, J. et al.** (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. NeurIPS. — causal mediation template.

Full bibliography in [`docs/bibliografia_tesi.md`](docs/bibliografia_tesi.md). Metric formalization in [`docs/Metriche_Nugget_DeepSeek.pdf`](docs/Metriche_Nugget_DeepSeek.pdf).

---

## License

MIT