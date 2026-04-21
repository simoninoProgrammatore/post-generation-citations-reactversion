# Post-Generation Citations — React Edition

A post-generation attribution system for Large Language Models, rebuilt as a **React + FastAPI** web application with integrated **interpretability tooling** for the NLI component.

Given an LLM response, the system decomposes it into atomic claims, retrieves supporting evidence from a corpus, verifies the claim–evidence link via NLI, and inserts inline citations — **all after generation**.

This is a full rewrite of the original Streamlit prototype, with a proper REST backend, a reactive frontend, and dedicated pages for inspecting the NLI model's decisions (attention flow, Integrated Gradients, Activation Patching).

> **Bachelor's thesis** · Università di Milano-Bicocca · Post-Generation Citations
> Paradigm: **P-Cite** (post-hoc citation), following [Saxena et al. (2025)](https://arxiv.org/abs/2502.10881). Architecture inspired by CEG ([Li et al., ACL 2024](https://aclanthology.org/2024.acl-long.619/)).

---

## Why this rewrite

The Streamlit version worked but had three structural limits:

1. **No separation of concerns** — compute and UI lived in the same process; every interaction re-ran the pipeline.
2. **No real API** — impossible to call the pipeline from outside the app, hard to test, hard to swap frontends.
3. **No room for interpretability** — Streamlit's rerun model fights against the kind of fine-grained NLI inspection the thesis now requires.

The React + FastAPI split fixes all three: backend is a stateless REST service cached per-model, frontend is a proper SPA with client-side routing, and interpretability lives in dedicated pages that call dedicated endpoints.

---

## Pipeline

```
Query ──► LLM ──► Raw Response ──► Decompose ──► Retrieve ──► NLI Verify ──► Cite ──► Cited Response
         (Claude)                  (atomic       (semantic     (DeBERTa      ([1][2])
                                    claims)       match)        entailment)
```

### Steps

1. **Generate** — Claude produces a response to the query (optionally conditioned on ALCE passages).
2. **Decompose** — The response is broken into atomic claims, inspired by [FActScore](https://arxiv.org/abs/2305.14627) (Min et al., 2023).
3. **Retrieve** — For each claim, candidate passages are retrieved from the pool.
4. **Verify (NLI)** — `cross-encoder/nli-deberta-v3-large` scores each (claim, passage) pair. Only pairs above an entailment threshold become citation candidates.
5. **Cite** — Inline markers `[n]` are inserted after the sentences containing the supported claims; a reference list is appended.
6. **Evaluate** (optional) — ALCE metrics: Citation Precision/Recall NLI, Correctness (EM / Claim Recall), Fluency (MAUVE).

Steps 1 and 2 use Claude; steps 3–6 run locally with DeBERTa.

---

## Architecture

### Backend — `backend/` (FastAPI + Uvicorn)

```
backend/
├── main.py                  # FastAPI app + CORS + router registration
├── requirements.txt
├── .env                     # ANTHROPIC_API_KEY (git-ignored)
│
├── core/                    # Pipeline primitives (framework-agnostic)
│   ├── generate.py          # Claude API wrapper
│   ├── decompose.py         # Atomic claim extraction
│   ├── retrieve.py          # Claim ↔ passage matching
│   ├── nli.py               # DeBERTa NLI scoring + caching
│   ├── cite.py              # Citation insertion + reference list
│   └── evaluate.py          # ALCE metrics
│
├── routers/                 # REST endpoints (thin wrappers over core/)
│   ├── pipeline.py          # /api/pipeline/{generate,decompose,retrieve,cite,run}
│   ├── nli.py               # /api/nli/predict, /api/nli/batch
│   ├── interpret.py         # /api/interpret/{attention,ig,patching}
│   └── dataset.py           # /api/dataset/{list,load}
│
├── data/
│   └── alce/                # ALCE benchmark data (gitignored, downloaded)
├── cache/                   # Model weights + response cache (gitignored)
└── results/                 # Experiment outputs
```

**Key design decisions:**

- **Router/core split.** `core/` is pure Python, no FastAPI imports — can be run as a standalone script, imported from a notebook, or wrapped by any framework. `routers/` only does request validation (Pydantic), error mapping, and delegation.
- **Single DeBERTa instance, lazy-loaded.** First call to `/api/nli/predict` pays the 30–60s model-loading cost; subsequent calls are fast. The instance is cached at module level in `core/nli.py`.
- **Stateless endpoints.** Each pipeline step takes all its inputs as JSON and returns all its outputs as JSON. No hidden server state. This makes retry, caching, and reproducibility trivial.
- **Dev server at `:8000`**, Swagger docs auto-generated at `/docs`.

### Frontend — `frontend/` (React + Vite)

```
frontend/
├── package.json
├── vite.config.js           # Dev server :5173 + proxy /api → :8000
├── index.html
└── src/
    ├── main.jsx             # React Router bootstrap
    ├── App.jsx              # Shell: Sidebar + TopBar + <Outlet/>
    ├── api.js               # fetch() wrappers — single source of truth for backend calls
    ├── index.css            # Global styles (custom, no Tailwind)
    │
    ├── components/
    │   ├── Sidebar.jsx
    │   ├── TopBar.jsx
    │   └── Icon.jsx         # Lucide icons wrapper
    │
    └── pages/
        ├── Pipeline.jsx         # Interactive end-to-end pipeline
        ├── Explore.jsx          # Browse generated / cited results
        ├── Metrics.jsx          # ALCE evaluation dashboard
        ├── Attention.jsx        # NLI attention flow inspection
        └── Interpretability.jsx # IG + Activation Patching
```

**Key design decisions:**

- **Vite, not CRA.** Faster dev server, lighter, modern standard.
- **React Router v6.** 5 pages, declarative routing, `<NavLink>` for automatic active state.
- **Single `api.js` module.** Every backend call goes through a typed wrapper (`api.generate({...})`, `api.nli({...})`). Components never touch `fetch` or URLs directly.
- **Vite proxy in dev.** `/api/*` from the frontend is transparently forwarded to `localhost:8000`, so no CORS pain in development. Production uses standard CORS.
- **No global state manager.** `useState` + Context API for the few cross-page concerns (current dataset, current model). Redux/Zustand would be overkill for this scope.
- **Lucide for icons**, custom CSS for everything else (design inherited from the original mockup).

---

## The five pages

### 1. Pipeline — end-to-end interactive run

Input a query, select a model, hit Run. The page streams each step of the pipeline: raw response → claims → retrieved passages → NLI scores → cited output with inline `[n]` markers and a reference panel. Every intermediate artifact is inspectable.

### 2. Explore — browse past runs

Load any ALCE dataset example or saved pipeline run; see the cited response with hover-highlighted evidence (hovering a `[n]` scrolls the corresponding passage into view and highlights the sentence DeBERTa identified as supporting).

### 3. Metrics — ALCE evaluation dashboard

Run the ALCE metrics over a batch of pipeline outputs: Citation Precision/Recall NLI, Citation F1, Correctness, Fluency. Tabular view + per-example drilldown.

### 4. Attention — NLI attention flow

For any `(premise, hypothesis)` pair, compute attention flow from `[CLS]` and visualize:
- `hyp_dominance` — how much attention flow ends on `H` vs `P+H`.
- Token-to-token heatmap for any selected layer and head.

Used during the thesis to falsify the attestation-bias hypothesis on spurious entailment cases.

### 5. Interpretability — IG + Activation Patching

Two tabs:

- **Integrated Gradients** (Captum, 50 steps, PAD baseline with `[CLS]`/`[SEP]` preserved): token-level attribution toward the entailment logit. Presets available for known biased cases.
- **Activation Patching** on the residual stream: given a clean/corrupt pair of equal length, patch each `(layer, position)` activation from corrupt into clean and measure the shift in entailment. Heatmap over the 24 layers × positions.

Both endpoints produce JSON that the page renders; the same JSON is saved to `backend/results/` for reproducibility.

---

## The NLI judge and its biases

The thesis surfaced a systematic failure mode of `cross-encoder/nli-deberta-v3-large` when used as a citation judge: on certain premise/hypothesis pairs where `P` does not entail `H`, the model predicts `entailment` with very high confidence. Example:

```
P: Italian tenor Andrea Bocelli performed a stunning rendition of Nessun Dorma
   at the closing ceremony.
H: The 2006 FIFA World Cup Final was played at the Olympiastadion in Berlin, Germany.

→ E = 0.949    (should be neutral)
```

**Why it matters for this project.** DeBERTa NLI is the final gate in the pipeline: it decides whether a retrieved passage actually supports a claim. If the judge produces spurious entailments, the system inserts wrong citations with high confidence — precisely the failure mode the project is built to prevent.

**Current status.** Attestation bias ([McKenna et al., EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.155/)) has been **falsified** as an explanation: attention flow analysis shows `[CLS]` attends to `P` as much as to `H` in biased cases. A **structural/register-sensitivity hypothesis** is under causal validation via Activation Patching — early results show ~100% of the bias on a test pair transfers causally through a single token position (the presence/absence of an indefinite article), in layers 0–14.

Details and raw data are in a separate findings document; the Interpretability page in the app is the live tool used to produce them.

---

## Endpoints (reference)

### Pipeline

- `POST /api/pipeline/generate` — `{query, model, passages?}` → `{response}`
- `POST /api/pipeline/decompose` — `{text, model}` → `{claims: [...]}`
- `POST /api/pipeline/retrieve` — `{claims, pool}` → `{matches: [...]}`
- `POST /api/pipeline/cite` — `{response, matches}` → `{cited_text, references}`
- `POST /api/pipeline/run` — end-to-end, all of the above chained.

### NLI

- `POST /api/nli/predict` — `{premise, hypothesis}` → `{entailment, contradiction, neutral}`
- `POST /api/nli/batch` — list of pairs → list of scores.

### Interpretability

- `POST /api/interpret/attention` — `{premise, hypothesis, layer?, head?}` → attention matrix + `hyp_dominance`.
- `POST /api/interpret/ig` — `{premise, hypothesis, target, steps}` → per-token attribution.
- `POST /api/interpret/patching` — `{clean, corrupt, hypothesis}` → `{clean_E, corrupt_E, effect: [[layer][pos]]}`.

### Dataset

- `GET /api/dataset/list` — available ALCE files.
- `POST /api/dataset/load` — `{name, subset?}` → examples.

Full interactive docs at `http://localhost:8000/docs`.

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- An Anthropic API key

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

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
bash scripts/download_data.sh   # Fetches ALCE into backend/data/alce/
```

The first call to any NLI endpoint downloads `cross-encoder/nli-deberta-v3-large` (~1.4 GB) into `backend/cache/`.

---

## Evaluation

Following ALCE (Gao et al., EMNLP 2023):

| Metric | What it measures |
|---|---|
| **Citation Precision (NLI)** | Of the passages we cited, how many actually support the claim? |
| **Citation Recall (NLI)** | Of the claims we should have cited, how many did we? |
| **Citation F1** | Harmonic mean. |
| **Correctness** | Exact Match / Claim Recall on gold answers. |
| **Fluency** | MAUVE. |

Run from the Metrics page, or directly:

```bash
cd backend
python -m core.evaluate --input results/cited.json --dataset asqa
```

---

## References

- **Saxena, A. et al.** (2025). *Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution*. — tassonomia G-Cite / P-Cite.
- **Li, W. et al.** (2024). *Citation-Enhanced Generation for LLM-based Chatbots*. ACL 2024. — il sistema più vicino architetturalmente.
- **Gao, T. et al.** (2023). *Enabling Large Language Models to Generate Text with Citations*. EMNLP 2023. — ALCE benchmark.
- **Min, S. et al.** (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. EMNLP 2023.
- **McKenna, N. et al.** (2023). *Sources of Hallucination by LLMs on Inference Tasks*. Findings of EMNLP 2023. — attestation bias (rilevante per la parte NLI-interpretability).
- **Sundararajan, M., Taly, A., Yan, Q.** (2017). *Axiomatic Attribution for Deep Networks*. ICML. — Integrated Gradients.
- **Meng, K. et al.** (2022). *Locating and Editing Factual Associations in GPT (ROME)*. NeurIPS. — Activation Patching.

---

## License

MIT
