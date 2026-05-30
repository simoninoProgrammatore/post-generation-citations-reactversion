# Bibliografia — Tesi triennale

Post-Generation Citation System · Confronto sistematico NLI vs LLM come giudici di citation attribution · Valutazione su **ALCE+** · Studio sui bias di DeBERTa NLI

Simone Frijio · 914366

*Versione aggiornata al 30 maggio 2026. Modifiche rispetto alla versione precedente segnate con **[NEW]** o **[UPDATE]**.*

---

## Legenda

- **[core]** — citazione obbligatoria, sostiene un'affermazione chiave della tesi.
- **[ctx]** — contesto / inquadramento, non obbligatoria ma rafforza l'argomento.
- **[alt]** — alternativa a un paper della lista, da valutare in base al taglio finale.
- **[preprint]** — non ancora pubblicato in conferenza al momento della verifica, usare con consapevolezza.

---

## 1 — Pipeline di citazione e framework di valutazione

### [core] Gao et al. (2023) — *Enabling Large Language Models to Generate Text with Citations*
EMNLP 2023. Introduce il benchmark ALCE (ASQA, QAMPARI, ELI5) e le metriche di valutazione delle citazioni (Citation Precision/Recall NLI, Fluency, Correctness). Framework di valutazione di partenza per la parte pipeline della tesi e dataset di base da cui è derivato ALCE+ (vedi sezione 2).

### [core] **[UPDATE]** Saxena et al. (2025) — *Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution*
**NeurIPS 2025** (39th Conference on Neural Information Processing Systems). arXiv:2509.21557, v2 dicembre 2025. **Il preprint è stato accettato a NeurIPS 2025**, rimosso il tag `[preprint]`. Introduce la tassonomia **G-Cite** (citazioni durante la generazione) vs **P-Cite** (citazioni post-hoc). Il sistema della tesi è P-Cite: citazione obbligatoria per posizionarlo nel campo. Trade-off centrale del paper (coverage vs citation correctness, con il retrieval come driver principale della qualità) molto rilevante per la discussione dei tuoi risultati top-1 vs top-3.

### [core] Li et al. (2024) — *Citation-Enhanced Generation for LLM-based Chatbots*
ACL 2024. Sistema CEG: il più vicino architetturalmente a quello della tesi (retrieval + NLI + citation insertion post-hoc). Riferimento principale per discutere cosa è già stato fatto.

### [core] Min et al. (2023) — *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*
EMNLP 2023. Giustifica la decomposizione in atomic claims come passo della pipeline. Concettualmente affine all'annotazione a livello di *nugget* atomico introdotta in ALCE+: in entrambi i casi l'unità di valutazione è il fatto, non la frase intera.

### [ctx] Bohnet et al. (2022) — *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models*
arXiv:2212.08037. Framework parallelo ad ALCE per QA attribuito. Utile per posizionamento.

### [ctx] Rashkin et al. (2023) — *Measuring Attribution in Natural Language Generation Models*
Computational Linguistics, 49(4). Definizione formale di "attribution" — fornisce il vocabolario rigoroso per parlare di citation verification.

### [ctx] Yue et al. (2023) — *Automatic Evaluation of Attribution by Large Language Models*
Findings of EMNLP 2023, pp. 4615–4635. Usa LLM come valutatori di attribuzione, alternativa al NLI. **Centrale** perché la tesi confronta proprio NLI e LLM come giudici di citation attribution — è il paper di riferimento per il setup del confronto.

### [ctx] **[NEW]** Honovich et al. (2022) — *TRUE: Re-evaluating Factual Consistency Evaluation*
NAACL 2022, pp. 3905–3920. Survey e meta-valutazione delle metriche di factual consistency su 11 dataset; mostra che gli approcci basati su large-scale NLI raggiungono performance forti e complementari rispetto a QG/QA. **Storicamente è il paper che ha consacrato NLI come metrica di attribuzione**, ed è citato da Gao (ALCE) come giustificazione dell'uso di NLI in `citation_precision`/`citation_recall`. Da inserire perché senza di esso il tuo argomento "NLI è lo standard de facto" è in sospeso.

### [ctx] **[NEW]** Liu et al. (2023) — *Evaluating Verifiability in Generative Search Engines*
EACL 2024 (precedentemente arXiv 2304.09848). Valuta la verifiability dei generative search engines (Bing, Perplexity), trovando che solo il 52% degli statement generati è pienamente supportato dalle citazioni. Citato esplicitamente da Yue 2023 come motivazione del problema. Utile in introduzione per quantificare la severità del problema che la tesi affronta.

---

## 2 — Dataset di valutazione: ALCE+

### [core] **ALCE+** (contributo di questa tesi)
ALCE+ è il dataset di valutazione costruito in questa tesi a partire da ALCE (Gao et al. 2023). Le modifiche introdotte sono: (a) **cropping a 5 passaggi per domanda** per trattabilità — l'ALCE originale fornisce ~100 passaggi per domanda, eccessivi per misurare la qualità del solo claim attribution; (b) **annotazione manuale di nugget atomici** per ogni domanda, ciascuno con flag `required: true/false`, `golden_passage_title`, `golden_evidence` span e keywords per il matching; (c) **supporto distrattori** opzionale, con marcatura `is_gold` / `is_noise` a livello passaggio. Tre tipologie di domanda (ASQA factoid-ambigue, QAMPARI multi-answer, ELI5 long-form), 30 esempi annotati per tipologia. Non è un paper esterno: è una contribuzione della tesi e va presentato come tale nel capitolo Metodo/Dataset.

### [core] Gao et al. (2023) — *Enabling Large Language Models to Generate Text with Citations*
Riportato qui per il ruolo di **dataset di partenza** da cui ALCE+ è derivato. Vedi sezione 1 per i dettagli.

### [ctx] Petroni et al. (2021) — *KILT: a Benchmark for Knowledge Intensive Language Tasks*
NAACL 2021. **Alternativa considerata ma non adottata.** KILT copre 11 dataset su 5 task (fact checking, entity linking, slot filling, open QA, dialogue) ancorati allo stesso snapshot Wikipedia, con campo `provenance` (wikipedia_id + paragrafo + range di caratteri) per ogni output. Due ragioni per cui KILT non è adatto come dataset di valutazione per questa tesi: (i) KILT valuta sistemi RAG **end-to-end** mescolando retrieval, generation e provenance, mentre l'obiettivo della tesi è isolare il claim attribution come componente; (ii) la granularità delle annotazioni KILT è **documentale** (wikipedia_id + range di caratteri), non a livello di fatto atomico — non permette di misurare quanti claim *necessari* di una risposta lunga vengono attribuiti correttamente. Da queste due limitazioni nasce la necessità di costruire ALCE+.

### [ctx] Thorne et al. (2018) — *FEVER: a Large-Scale Dataset for Fact Extraction and VERification*
NAACL 2018. Benchmark di riferimento per fact verification, parte di KILT. Citato per chiudere il contesto KILT in sezione related work, e perché il pipeline di citation attribution è strutturalmente affine a un sistema di fact verification applicato a claim generati.

### [ctx] Bowman et al. (2015) — *A Large Annotated Corpus for Learning Natural Language Inference*
EMNLP 2015. SNLI. Obbligatoria quando si menziona il fine-tuning di DeBERTa (che usa SNLI + MultiNLI + FEVER-NLI + ANLI).

### [ctx] Williams, Nangia & Bowman (2018) — *A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference*
NAACL 2018. MultiNLI. Stesso ruolo di SNLI.

### [ctx] Nie et al. (2020) — *Adversarial NLI: A New Benchmark for Natural Language Understanding*
ACL 2020. ANLI. Mostra i limiti dei modelli NLI su esempi adversarial — supporta l'argomento che DeBERTa su casi fuori distribuzione (come quelli scoperti nella tesi) è meno robusto.

### [ctx] Parrish et al. (2021) — *Does Putting a Linguist in the Loop Improve NLU Data Collection?*
Findings of EMNLP 2021. Limiti strutturali dei dataset NLI crowdsourced — radice degli shortcut osservati.

---

## 3 — Modelli e tecniche di base

### [core] He et al. (2023) — *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*
ICLR 2023. Modello usato come giudice di entailment (`cross-encoder/nli-deberta-v3-large`). Obbligatoria in metodologia.

### [ctx] Reimers & Gurevych (2019) — *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
EMNLP 2019. Giustifica l'uso di embedding tramite la libreria `sentence-transformers` e l'architettura cross-encoder per NLI, nonché degli embedding pooled per il pre-filtering. Citazione obbligatoria nel capitolo metodo quando si descrivono i componenti `CrossEncoder` (NLI) e `SentenceTransformer` (pre-filter MiniLM/BGE).

### [core] **[NEW]** Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
NeurIPS 2020. Paper canonico di RAG. La pipeline della tesi è strutturalmente RAG (retrieval + generation + post-hoc citation), e finora questo riferimento mancava completamente. Obbligatorio quando descrivi l'architettura del sistema in metodologia.

### [ctx] **[NEW]** Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain Question Answering*
EMNLP 2020. DPR. Riferimento per il retriever dense usato come componente del sistema. Da citare in metodologia, sezione retrieval.

---

## 4 — LLM-as-a-judge (nuova sezione)

**Motivazione della nuova sezione**: la tesi confronta sistematicamente NLI e LLM come giudici di citation attribution. La vecchia bibliografia non aveva nessuna citazione sull'LLM-as-a-judge come paradigma di valutazione — un buco non sostenibile, perché metà del contributo della tesi è proprio su questo lato.

### [core] **[NEW]** Zheng et al. (2023) — *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
NeurIPS 2023. Paper fondazionale dell'LLM-as-a-judge come paradigma di valutazione automatica. Documenta agreement con valutazione umana >80%, ma anche bias sistematici (position bias, verbosity bias, self-enhancement bias). **Obbligatorio** in metodologia quando descrivi il setup con DeepSeek come judge; obbligatorio anche in discussion quando interpreti differenze NLI vs LLM (i bias del judge potrebbero confondere l'interpretazione).

### [ctx] **[NEW]** Yue et al. (2023) — *Automatic Evaluation of Attribution by Large Language Models*
Findings of EMNLP 2023. Spostato qui da sezione 1 perché è il **link diretto** tra LLM-as-judge e citation attribution: il paper definisce AttrScore (full/partial/no support, le stesse categorie che usi nel tuo modulo DeepSeek) ed esplora sia prompting LLMs sia fine-tuning di small LMs per il task. Il tuo setup è quasi una replica di AttrScore con un altro judge.

### [ctx] **[NEW]** Li et al. (2024) — *AttributionBench: How Hard is Automatic Attribution Evaluation?*
arXiv:2402.15089. **[preprint]** o NAACL 2024 (verificare). Estende Yue 2023 con un benchmark sistematico; mostra che anche GPT-3.5 fine-tuned arriva solo all'80% di macro-F1 sul task. Utile per discutere i limiti del judge LLM nei tuoi risultati.

---

## 5 — Attestation bias e conoscenza parametrica (finding principale)

### [core] McKenna et al. (2023) — *Sources of Hallucination by Large Language Models on Inference Tasks*
Findings of EMNLP 2023, pp. 2758–2774. **Paper centrale per il capitolo sui bias**. Introduce l'attestation bias: i modelli NLI etichettano come entailment ogni esempio in cui l'ipotesi è attestata nel training, indipendentemente dalla premessa. **La tesi testa questa ipotesi su DeBERTa come spiegazione del comportamento osservato (falsi positivi NLI ad alta confidenza), la falsifica via attention flow analysis (CLS attende P quanto H nei casi biased), e propone un meccanismo alternativo di sensibilità strutturale/registro, parzialmente validato via activation patching.**

### [core] **[NEW]** Cheng et al. (2025) — *Neutralizing Bias in LLM Reasoning using Entailment Graphs*
**Findings of ACL 2025**, pp. 13714–13730 (Vienna). arXiv:2503.11614. Confermato e ora rimosso il `[preprint]`. Propone un framework unsupervised per generare dati controfattuali via Entailment Graphs e mitigare l'attestation bias di McKenna. Numeri concreti: AttBias di DeepSeek-R1-Llama-8B scende da 26.04 a 7.58, di Mistral-7B da 32.98 a 13.00. **Rilevanza per la tesi**: (i) conferma indipendentemente l'esistenza dell'attestation bias come fenomeno; (ii) il framework di mitigazione è applicabile a DeBERTa-NLI come "lavoro futuro"; (iii) la metodologia bias-neutralizzata (sostituzione entità con sampling casuale) ricorda la tua batteria "Invented entities" — utile per discutere convergenza metodologica.

### [core] Longpre et al. (2021) — *Entity-Based Knowledge Conflicts in Question Answering*
EMNLP 2021. Conflitto tra conoscenza parametrica e contesto fornito. Quadro concettuale per inquadrare il bias come fallimento nella gestione del conflitto knowledge-vs-context.

### [ctx] Xie et al. (2023) — *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts*
ICLR 2024. Plausibilità contestuale e gestione dei conflitti knowledge/context. **Attenzione**: tenere solo se la tesi discute esplicitamente l'interazione tra conoscenza parametrica e contesto recuperato; altrimenti rimuovere.

### [ctx] **[UPDATE]** Basmov, Goldberg & Tsarfaty (2024) — *LLMs' Reading Comprehension Is Affected by Parametric Knowledge and Struggles with Hypothetical Statements*
arXiv:2404.06283, v2 luglio 2025. **[preprint]** — confermato preprint, non ho trovato pubblicazione in venue. Propone l'uso di "imaginary data" (fatti ed entità fittizie) per testare la reading comprehension degli LLM **senza** che la conoscenza parametrica contamini la valutazione. **Direttamente in linea con la tua batteria "Invented entities"**: stesso disegno sperimentale, applicato però a LLM generativi anziché a NLI cross-encoder. Citato anche da Cheng 2025 ICLR 2025 (Controllable Context Sensitivity, vedi sopra). Da inserire come riferimento metodologico per giustificare la scelta di entità inventate.

---

## 6 — Shortcut learning e artifact nei dataset NLI

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
ACL 2019. Adversarial training per mitigare gli artifact. Rilevante in lavoro futuro (la tesi non implementa mitigazione).

### [ctx] Kaushik, Hovy & Lipton (2020) — *Learning the Difference that Makes a Difference with Counterfactually-Augmented Data*
ICLR 2020. Dati controfattuali per ridurre shortcut. Rilevante in lavoro futuro.

### [ctx] Schuster et al. (2019) — *Towards Debiasing Fact Verification Models*
EMNLP 2019. Molto rilevante perché citation verification è essenzialmente fact verification. Mostra che FEVER ha artifact nell'ipotesi e propone debiasing.

---

## 7 — Memorizzazione nei language models

### [core] Carlini et al. (2023) — *Quantifying Memorization Across Neural Language Models*
ICLR 2023. I modelli memorizzano letteralmente parti del training set, e la memorizzazione scala con la dimensione del modello. Supporta la futura osservazione sperimentale che il fenomeno si manifesta su DeBERTa-large ma potrebbe non essere presente su DeBERTa-base (se verificato). **[alt]** sostituisce Xu et al. (2024) sul data contamination della vecchia lista, perché più pertinente: il problema qui non è test leakage, è memorizzazione legittima.

### [ctx] Kandpal, Wallace & Raffel (2022) — *Deduplicating Training Data Mitigates Privacy Risks in Language Models*
ICML 2022. Correla frequenza di duplicazione nel training a memorizzazione. Spiega perché testi Wikipedia-style (alta frequenza) sono memorizzati — rilevante dato che il pool di ALCE+ è derivato da ALCE, che a sua volta usa passaggi Wikipedia.

### [ctx] Zhang et al. (2023) — *Counterfactual Memorization in Neural Language Models*
NeurIPS 2023. Distingue memorizzazione effettiva da apprendimento statistico. Utile per il framing preciso del finding.

### [ctx] Elangovan et al. (2021) — *Memorization vs. Generalization: Quantifying Data Leakage in NLP Performance Evaluation*
Quantifica il data leakage nella valutazione NLP. Da tenere solo se la tesi discute esplicitamente la possibile contaminazione del benchmark; altrimenti ridondante con Carlini.

---

## 8 — Interpretabilità meccanicistica (metodi usati)

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

## 9 — Riferimenti rimossi o declassati rispetto alla vecchia lista

- **Xu et al. (2024) — Benchmark Data Contamination Survey** → rimosso, sostituito da Carlini et al. (2023). Il problema non è test-set leakage in senso stretto ma memorizzazione di testi pubblici; Carlini è più pertinente.
- **Petroni 2021 (KILT)** → declassato da [core] a [ctx]. KILT non è il dataset della tesi: la tesi usa ALCE+, ALCE+ è derivato da ALCE (Gao 2023). KILT resta come benchmark di confronto e motivazione per la costruzione di ALCE+.
- **Thorne 2018 (FEVER)** → declassato da [core] a [ctx]. Citato solo come parte del contesto KILT.
- **Xie et al. (2023)** → declassato a [ctx]. Da tenere solo se la tesi discute esplicitamente il conflitto conoscenza/contesto; altrimenti ridondante con Longpre.
- **[NEW]** Vecchia sezione 8 "Paper da tenere sotto osservazione" eliminata: i due titoli sono stati verificati e promossi (Cheng 2025 → §5 [core]; Basmov 2024 → §5 [ctx] [preprint]).

---

## Struttura di citazione per la tesi

Per orientarsi nel mapping bibliografia → capitoli:

| Capitolo | Citazioni core |
|---|---|
| Introduzione e motivazione | Gao 2023, Saxena 2025, Li 2024, Liu 2023 (verifiability) |
| Background e Related Work (citation in RAG) | Gao 2023, Saxena 2025, Li 2024, Bohnet 2022, Rashkin 2023, Yue 2023, Honovich 2022 |
| **Background LLM-as-judge** **[NEW]** | Zheng 2023, Yue 2023 |
| Pipeline del sistema | Gao 2023, Min 2023, Li 2024, Saxena 2025, Lewis 2020 (RAG), Karpukhin 2020 (DPR) |
| Dataset di valutazione (**ALCE+**) | Gao 2023 (dataset di partenza), Min 2023 (FActScore, granularità) |
| Dataset alternativi considerati | Petroni 2021 (KILT), Thorne 2018 (FEVER) — entrambi [ctx] |
| Modello NLI e componenti | He 2023, Reimers & Gurevych 2019, Bowman 2015, Williams 2018, Nie 2020 |
| Valutazione | Gao 2023 (metriche ALCE), Rashkin 2023 (definizione attribution), Honovich 2022 (TRUE) |
| Scoperta del bias | McKenna 2023, Cheng 2025, Gururangan 2018, McCoy 2019, Niven 2019 |
| Framing teorico shortcut learning | Geirhos 2020, Niven 2019, Longpre 2021 |
| Metodologia interpretabilità | Sundararajan 2017, Meng 2022, Vig 2020, Abnar 2020 |
| Discussione memorizzazione | Carlini 2023, Kandpal 2022 |
| Lavoro futuro | Hewitt 2019, Belinkov 2022, Kaushik 2020, Belinkov 2019, Schuster 2019, **Cheng 2025** (mitigazione) |

---

## Sintesi dei cambiamenti rispetto alla versione precedente

1. **Saxena 2025**: promosso da `[preprint]` a NeurIPS 2025 pubblicato (verificato su arxiv.org/abs/2509.21557 e neurips.cc/virtual/2025/loc/san-diego/122374).
2. **Cheng et al. 2025** (vecchio §8 "Neutralizing Bias"): verificato e promosso a `[core]` in §5. ACL Findings 2025 (verificato su aclanthology.org/2025.findings-acl.705).
3. **Basmov et al. 2024**: verificato; resta `[preprint]` (arXiv:2404.06283), promosso a `[ctx]` in §5 con motivazione esplicita del legame con la batteria "Invented entities".
4. **Nuova sezione §4 LLM-as-judge** con tre paper (Zheng 2023, Yue 2023 spostato, Li 2024 AttributionBench): copre un buco metodologico significativo.
5. **§3 Modelli base**: aggiunti Lewis 2020 (RAG canonico) e Karpukhin 2020 (DPR), che mancavano del tutto.
6. **§1 Pipeline**: aggiunti Honovich 2022 (TRUE) per giustificare NLI come standard di valutazione, e Liu 2023 (verifiability) per quantificare il problema in introduzione.