# Bibliografia — Tesi triennale

Post-Generation Citation System · Studio sui bias di DeBERTa NLI come giudice di citazione · Valutazione su KILT

Simone Frijio · 914366

---

## Legenda

- **[core]** — citazione obbligatoria, sostiene un'affermazione chiave della tesi.
- **[ctx]** — contesto / inquadramento, non obbligatoria ma rafforza l'argomento.
- **[alt]** — alternativa a un paper della lista, da valutare in base al taglio finale.
- **[preprint]** — non ancora pubblicato in conferenza al momento della verifica, usare con consapevolezza.

---

## 1 — Pipeline di citazione e framework di valutazione

### [core] Gao et al. (2023) — *Enabling Large Language Models to Generate Text with Citations*
EMNLP 2023. Introduce il benchmark ALCE (ASQA, QAMPARI, ELI5) e le metriche di valutazione delle citazioni (Citation Precision/Recall NLI, Fluency, Correctness). Framework di valutazione principale per la parte pipeline della tesi.

### [core] Saxena et al. (2025) — *Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution*
arXiv:2509.21557. **[preprint]** — al momento della verifica è su arXiv, non ancora in conferenza. Introduce la tassonomia **G-Cite** (citazioni durante la generazione) vs **P-Cite** (citazioni post-hoc). Il sistema della tesi è P-Cite: citazione obbligatoria per posizionarlo nel campo. Se il preprint fosse ritirato o modificato, sostituirlo con Schreieder et al. (2025).

### [core] Li et al. (2024) — *Citation-Enhanced Generation for LLM-based Chatbots*
ACL 2024. Sistema CEG: il più vicino architetturalmente a quello della tesi (retrieval + NLI + citation insertion post-hoc). Riferimento principale per discutere cosa è già stato fatto.

### [core] Min et al. (2023) — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
EMNLP 2023. Giustifica la decomposizione in atomic claims come passo della pipeline.

### [ctx] Bohnet et al. (2022) — *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models*
arXiv. Framework parallelo ad ALCE per QA attribuito. Utile per posizionamento.

### [ctx] Rashkin et al. (2023) — *Measuring Attribution in Natural Language Generation Models*
Computational Linguistics. Definizione formale di "attribution" — fornisce il vocabolario rigoroso per parlare di citation verification.

### [ctx] Yue et al. (2023) — *Automatic Evaluation of Attribution by Large Language Models*
Findings of EMNLP 2023. Usa LLM come valutatori di attribuzione, alternativa al NLI. Rilevante nella sezione in cui si giustifica la scelta di DeBERTa-NLI come giudice.

---

## 2 — Dataset e benchmark

### [core] Petroni et al. (2021) — *KILT: a Benchmark for Knowledge Intensive Language Tasks*
NAACL 2021. **Dataset principale della tesi**. 11 dataset su 5 task (fact checking, entity linking, slot filling, open QA, dialogue) ancorati allo stesso snapshot Wikipedia (1 agosto 2019). Formato JSON Lines con campo `provenance` (wikipedia_id + paragrafo + range di caratteri) per ogni output. Usato nella tesi per oracle-retrieval evaluation delle citazioni: si saltano le fasi di retrieval e si misura solo la qualità della citazione data la provenance gold.

### [core] Thorne et al. (2018) — *FEVER: a large-scale dataset for Fact Extraction and VERification*
NAACL 2018. Benchmark di riferimento per fact verification, parte di KILT. Il pipeline di citazione è strutturalmente un sistema di fact verification applicato a claim generati.

### [ctx] Bowman et al. (2015) — *A large annotated corpus for learning natural language inference*
EMNLP 2015. SNLI. Obbligatoria quando si menziona il fine-tuning di DeBERTa (che usa SNLI + MultiNLI + FEVER-NLI + ANLI).

### [ctx] Williams, Nangia & Bowman (2018) — *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*
NAACL 2018. MultiNLI. Stesso ruolo di SNLI.

### [ctx] Nie et al. (2020) — *Adversarial NLI: A New Benchmark for Natural Language Understanding*
ACL 2020. ANLI. Mostra i limiti dei modelli NLI su esempi adversarial — supporta l'argomento che DeBERTa su casi fuori distribuzione (come quelli scoperti nella tesi) è meno robusto.

### [ctx] Parrish et al. (2021) — *Does Putting a Linguist in the Loop Improve NLU Data Collection?*
Findings of EMNLP 2021. Limiti strutturali dei dataset NLI crowdsourced — radice degli shortcut osservati.

---

## 3 — Modello NLI

### [core] He et al. (2023) — *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*
ICLR 2023. Modello usato come giudice di entailment (`cross-encoder/nli-deberta-v3-large`). Obbligatoria in metodologia.

---

## 4 — Attestation bias e conoscenza parametrica (finding principale)

### [core] McKenna et al. (2023) — *Sources of Hallucination by Large Language Models on Inference Tasks*
Findings of EMNLP 2023, pp. 2758–2774. **Paper centrale per il capitolo sui bias**. Introduce l'attestation bias: i modelli NLI etichettano come entailment ogni esempio in cui l'ipotesi è attestata nel training, indipendentemente dalla premessa. La tesi riprende il concetto, lo testa su DeBERTa, e mostra che la spiegazione "classica" (attenzione concentrata su H) **non regge** per il caso osservato — motivo per cui l'analisi si muove verso l'ipotesi di register sensitivity.

### [core] Longpre et al. (2021) — *Entity-Based Knowledge Conflicts in Question Answering*
EMNLP 2021. Conflitto tra conoscenza parametrica e contesto fornito. Quadro concettuale per inquadrare il bias come fallimento nella gestione del conflitto knowledge-vs-context.

### [ctx] Xie et al. (2023) — *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts*
ICLR 2024. Plausibilità contestuale e gestione dei conflitti knowledge/context. **Attenzione**: nella vecchia lista era stato segnalato come forse non strettamente rilevante. Tenere solo se la tesi discute esplicitamente l'interazione tra conoscenza parametrica e contesto recuperato; altrimenti rimuovere.

---

## 5 — Shortcut learning e artifact nei dataset NLI

### [core] Gururangan et al. (2018) — *Annotation Artifacts in Natural Language Inference Data*
NAACL 2018. Risultato fondamentale: classificatori hypothesis-only raggiungono ~67% di accuracy su SNLI. Giustifica perché l'ipotesi "il modello guarda solo H" (attestation bias classico) fosse un punto di partenza ragionevole, prima che l'analisi della tesi la falsificasse.

### [core] Poliak et al. (2018) — *Hypothesis Only Baselines in Natural Language Inference*
*SEM 2018. Estensione del risultato di Gururangan a 10 dataset NLI diversi.

### [core] McCoy, Pavlick & Linzen (2019) — *Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference*
ACL 2019. Dataset HANS. Euristica sintattica (overlap lessicale) — vicina al fenomeno osservato nella tesi ma distinta. Utile per inquadrare: HANS mostra che BERT usa overlap, la tesi mostra che DeBERTa usa feature di registro. Stesso framework (shortcut learning), meccanismi diversi.

### [core] Geirhos et al. (2020) — *Shortcut Learning in Deep Neural Networks*
Nature Machine Intelligence. Framework concettuale di riferimento. Citazione obbligatoria per inquadrare il fenomeno.

### [core] Niven & Kao (2019) — *Probing Neural Network Comprehension of Natural Language Arguments*
ACL 2019. Mostra che BERT classifica argomentazioni sfruttando la parola "not". Esempio canonico di shortcut lessicale — parallelo diretto a quello che la tesi osserva con l'articolo indefinito `a`.

### [ctx] Belinkov et al. (2019) — *Don't Take the Premise for Granted: Mitigating Artifacts in Natural Language Inference*
ACL 2019. Adversarial training per mitigare gli artifact. Rilevante se la tesi discute mitigazioni.

### [ctx] Kaushik, Hovy & Lipton (2020) — *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*
ICLR 2020. Dati controfattuali per ridurre shortcut — collegato concettualmente a mitigazioni come l'entity overlap filter.

### [ctx] Schuster et al. (2019) — *Towards Debiasing Fact Verification Models*
EMNLP 2019. Molto rilevante perché citation verification è essenzialmente fact verification. Mostra che FEVER ha artifact nell'ipotesi e propone debiasing.

---

## 6 — Memorizzazione nei language models

### [core] Carlini et al. (2023) — *Quantifying Memorization Across Neural Language Models*
ICLR 2023. I modelli memorizzano letteralmente parti del training set, e la memorizzazione scala con la dimensione del modello. Supporta la futura osservazione sperimentale che il fenomeno si manifesta su DeBERTa-large ma potrebbe non essere presente su DeBERTa-base (se verificato). **[alt]** sostituisce Xu et al. (2024) sul data contamination della vecchia lista, perché più pertinente: il problema qui non è test leakage, è memorizzazione legittima.

### [ctx] Kandpal, Wallace & Raffel (2022) — *Deduplicating Training Data Mitigates Privacy Risks in Language Models*
ICML 2022. Correla frequenza di duplicazione nel training a memorizzazione. Spiega perché testi Wikipedia-style (alta frequenza) sono memorizzati — rilevante dato che il pool di KILT è proprio Wikipedia.

### [ctx] Zhang et al. (2023) — *Counterfactual Memorization in Neural Language Models*
NeurIPS 2023. Distingue memorizzazione effettiva da apprendimento statistico. Utile per il framing preciso del finding.

### [ctx] Elangovan et al. (2021) — *Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation*
Quantifica il data leakage nella valutazione NLP. Da tenere solo se la tesi discute esplicitamente la possibile contaminazione del benchmark; altrimenti ridondante con Carlini.

---

## 7 — Interpretabilità meccanicistica (metodi usati)

### [core] Sundararajan, Taly & Yan (2017) — *Axiomatic Attribution for Deep Networks*
ICML 2017. Paper originale di Integrated Gradients. Obbligatoria in metodologia dove si descrive IG (50 step, baseline PAD con CLS/SEP preservati, target = logit di entailment).

### [core] Meng et al. (2022) — *Locating and Editing Factual Associations in GPT (ROME)*
NeurIPS 2022. Paper che ha reso famoso l'activation patching come strumento per localizzare memorie parametriche. Giustifica l'uso di activation patching sul residual stream nella tesi.

### [core] Vig et al. (2020) — *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*
NeurIPS 2020. Template metodologico diretto per localizzare un bias in un transformer tramite analisi causale. Citazione forte per giustificare il disegno sperimentale.

### [ctx] Abnar & Zuidema (2020) — *Quantifying Attention Flow in Transformers*
ACL 2020. Giustifica il metodo di attention flow dal `[CLS]` usato nella fase 1 dell'analisi (hyp_dominance), e spiega perché guardare l'attention di un singolo layer non basta.

### [ctx] Hewitt & Liang (2019) — *Designing and Interpreting Probes with Control Tasks*
EMNLP 2019. Metodologia corretta per probing classifier. Da includere solo se la tesi aggiunge probing (proposto come lavoro futuro).

### [ctx] Belinkov (2022) — *Probing Classifiers: Promises, Shortcomings, and Advances*
Computational Linguistics. Survey canonica sul probing. Solo se si fa probing.

---

## 8 — Paper da tenere sotto osservazione o verificare

Riferimenti citati nella vecchia lista che richiedono verifica prima di essere inseriti in tesi:

- **"Neutralizing Bias in LLM Reasoning using Entailment Graphs" (2025)** — verificare autori, sede di pubblicazione, e se propone davvero una mitigazione applicabile al caso della tesi. Se è arXiv-only, marcarlo come **[preprint]**.
- **"LLMs' Reading Comprehension Is Affected by Parametric Knowledge" (2024)** — verificare riferimento completo. L'idea (uso di dati fittizi per valutazione pulita) è in linea con la parte "Invented entities" della batteria sperimentale — se il paper regge, è una citazione molto utile.

---

## 9 — Riferimenti rimossi rispetto alla vecchia lista (con motivazione)

- **Xu et al. (2024) — Benchmark Data Contamination Survey** → sostituito da Carlini et al. (2023). Il problema non è test-set leakage in senso stretto ma memorizzazione di testi pubblici; Carlini è più pertinente.
- **Xie et al. (2023) sulla plausibilità contestuale** → declassato a **[ctx]**. Da tenere solo se la tesi discute esplicitamente il conflitto conoscenza/contesto; altrimenti ridondante con Longpre.

---

## Struttura di citazione suggerita per la tesi

Per orientarsi nel mapping bibliografia → capitoli:

| Capitolo | Citazioni core |
|---|---|
| Introduzione e motivazione | Gao 2023, Saxena 2025, Li 2024 |
| Pipeline del sistema | Gao 2023, Min 2023, Li 2024 |
| Dataset (KILT) | Petroni 2021, Thorne 2018 |
| Modello NLI | He 2023, Bowman 2015, Williams 2018 |
| Valutazione | Gao 2023 (metriche ALCE), Petroni 2021 (metriche KILT), Rashkin 2023 |
| Scoperta del bias | McKenna 2023, Gururangan 2018, McCoy 2019 |
| Framing teorico | Geirhos 2020, Niven 2019, Longpre 2021 |
| Metodologia interpretabilità | Sundararajan 2017, Meng 2022, Vig 2020, Abnar 2020 |
| Discussione memorizzazione | Carlini 2023, Kandpal 2022 |
| Lavoro futuro | Hewitt 2019, Belinkov 2022, Kaushik 2020 |
