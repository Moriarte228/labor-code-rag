# RAG over the Labor Code of Belarus

Question answering over the Labor Code of the Republic of Belarus (Трудовой кодекс РБ, ~500 articles). A user asks in everyday Russian, the system retrieves the relevant articles, and an LLM answers grounded in them.

The project is mostly about **retrieval quality**: every change was measured on a hand-labeled eval set and logged, one change at a time.

**Best result:** Recall@1 = **0.808**, Recall@3 = 0.962, MRR = 0.885 (hybrid retrieval + cross-encoder reranker, 52 questions).

## Pipeline

1. **Parsing** — PyMuPDF at span level. Superscript article numbers (29¹) are recovered via font flags as `29-1`; otherwise they collapse into `291` and break both numbering and labels. Text is split by article headers, then into paragraph chunks, each tagged with its article number.
2. **Retrieval** — dense search (`paraphrase-multilingual-mpnet-base-v2`, cosine) and BM25 (`rank_bm25`), fused with Reciprocal Rank Fusion.
3. **Reranking** — cross-encoder `BAAI/bge-reranker-v2-m3` scores (question, full article text) pairs.
4. **Alternative strategy: structure navigation** — the LLM picks chapters from the table of contents, then articles by their titles (two LLM calls per query).
5. **Generation** — LLM via OpenRouter, instructed to answer only from the retrieved articles and say so if the answer isn't there.

## Evaluation

- **52 questions, one gold article each.** Labeled "from article to question": read an article, then phrase the question the way a non-lawyer would, deliberately colloquial ("за что начальник может уволить?" rather than legal wording).
- Multi-hop questions were converted to single-hop so that partial hits don't inflate the metrics.
- **Metrics:** Recall@k (k = 1, 3, 5, 10) and MRR.
- Every run is logged to `results.csv` by `results_log.py`. The experiment name is the row key, so rerunning an experiment overwrites its row instead of duplicating it.

### Main results (full eval set)

| Strategy | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| Hybrid RRF + reranker (previous model), candidate pool = 2k | 0.451 | 0.686 | 0.843 | 0.941 | 0.752 |
| + switch to `bge-reranker-v2-m3` | 0.490 | 0.725 | 0.863 | 0.941 | 0.859 |
| + larger fixed candidate pool for the reranker | **0.808** | **0.962** | **0.962** | **0.981** | 0.885 |
| Structure navigation (LLM: chapters → articles) | 0.769 | 0.769 | 0.942 | 0.962 | **0.909** |

The first two rows were measured on the 51-question version of the set.

### Early ablation (first 15 questions)

| Strategy | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| Cosine, `paraphrase-multilingual-MiniLM-L12-v2` | 0.533 | 0.600 | 0.600 | 0.667 | 0.578 |
| BM25 | 0.333 | 0.600 | 0.667 | 0.667 | 0.483 |
| Hybrid RRF (MiniLM) | 0.400 | 0.667 | 0.733 | 0.800 | 0.550 |
| Cosine, `paraphrase-multilingual-mpnet-base-v2` | 0.467 | 0.733 | 0.733 | 0.800 | 0.578 |
| Hybrid RRF (mpnet) | 0.467 | 0.600 | 0.733 | 0.933 | 0.571 |
| + cross-encoder reranker | 0.467 | 0.600 | 0.733 | 0.933 | 0.822 |
| + larger candidate pool | 0.533 | 0.733 | 0.867 | 1.000 | 0.850 |

## Findings

- **The candidate pool was the bottleneck, not the reranker.** A reranker can only promote what it is shown. Widening its candidate pool raised R@1 from 0.49 to 0.81 with no model change.
- **Terminology gap.** Everyday wording ("уволить") doesn't match legal wording ("расторжение трудового договора по инициативе нанимателя"), so the gold article often lands in the top 10 but not at position 1.
- **BM25 can hurt the hybrid.** On terminology-gap queries BM25 adds lexical noise that pushes the gold article out of the reranker's window.
- **The RRF constant didn't matter** at this scale: k = 10, 30 and 100 gave identical metrics.
- **Structure navigation** has the best MRR (0.909), but lower R@1, is nondeterministic and costs two LLM calls per query.
- **HyDE failed silently.** When the API returned a 401 error, the error text itself was embedded as the "hypothetical article". Lesson: validate LLM output before passing it downstream.
- **Data pitfalls:**
  - Zero-width spaces (`\u200b`) in the document and narrow no-break spaces (`\u202f`) in LLM responses silently broke regex parsing.
  - Moving to a newer, cleaned edition of the Code first dropped R@1 to 0.08. That was a label/document numbering mismatch, not a model problem; once labels were re-aligned, the metrics recovered.

## Limitations and next steps

- The eval set is small: one question is about 2 percentage points of Recall. All questions are single-hop.
- The CLI demo (`main.py`) currently uses plain dense retrieval rather than the best pipeline.
- Next: agentic RAG — the LLM reformulates the query and calls retrieval as a tool, evaluated by whether the gold article is among the cited ones.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OpenRouter key
python main.py                # ask a question in the console
python evaluation.py          # run the eval (strategy is chosen in the evaluate() call)
```

On the first run the embedding and reranker models are downloaded from Hugging Face (a few GB).

| File | Purpose |
|---|---|
| `load_document.py` | parsing, chunking, all retrieval strategies, reranking |
| `eval_set.py` | labeled questions |
| `evaluation.py` | Recall@k / MRR for any strategy `(query, k) → articles` |
| `results_log.py`, `results.csv` | experiment journal |
| `main.py` | console Q&A |
| `tk_rb_cleaned.pdf` | source document (annotations removed) |
