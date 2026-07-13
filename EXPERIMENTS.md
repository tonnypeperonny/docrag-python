# Retrieval experiments

Scenario log for tuning retrieval on the sample corpus. Metrics come from
`python main.py eval` (hit@k + MRR against `evalset.jsonl`, split into
keyword vs paraphrase questions). Re-run `ingest` between chunking scenarios —
the index only holds one chunking configuration at a time.

Results below: 2026-07-13, Elasticsearch 9.1.4, all-MiniLM-L6-v2, 14-question
evalset (8 keyword / 6 paraphrase), answers via llama3.2:3b on Ollama.
Everything runs locally — no hosted APIs.

## 1. Retrieval mode: hybrid vs BM25-only vs kNN-only

Does RRF fusion actually beat either branch alone, and where does each
branch fail? Expectation: BM25 wins keyword questions (e.g. "ADRs"),
kNN wins paraphrase questions, hybrid is the best overall.

```powershell
.\.venv\Scripts\python main.py ingest sample-docs
.\.venv\Scripts\python main.py eval
```

| mode   | hit@3 | MRR   | keyword MRR | paraphrase MRR | notes |
|--------|-------|-------|-------------|----------------|-------|
| hybrid | 1.00  | 0.964 | 1.000       | 0.917          | inherits bm25's one paraphrase rank-2 |
| bm25   | 1.00  | 0.964 | 1.000       | 0.917          | drops on paraphrase, as expected |
| knn    | 1.00  | 0.964 | 0.938       | 1.000          | drops on keyword, as expected |

At default chunking the corpus is only 3 chunks (each doc fits in one chunk),
so hit@3 is saturated and only MRR differentiates. The branches fail exactly
where predicted — bm25 on paraphrase, knn on keyword — and mirror each other.
The mode differences get much clearer at 400-char chunking (11 chunks): there
bm25 paraphrase MRR falls to 0.806 while knn holds 1.000, and hybrid stays at
1.000 everywhere — fusion genuinely rescues bm25's paraphrase misses.
Real conclusion: 3 docs is too small; need distractor docs (see backlog).

## 2. Chunk size sweep

Smaller chunks = more precise retrieval but less context per chunk for the
LLM; larger chunks = the opposite. Where is the sweet spot for these docs?

```powershell
.\.venv\Scripts\python main.py ingest sample-docs --chunk 400
.\.venv\Scripts\python main.py eval --mode hybrid
# repeat for 800, 1200 (default), 2400
```

| chunk size | chunks indexed | hit@3 | MRR   | notes |
|------------|----------------|-------|-------|-------|
| 400        | 11             | 1.00  | 1.000 | best — one section ≈ one chunk |
| 800        | 6              | 1.00  | 1.000 |       |
| 1200       | 3              | 1.00  | 0.964 | whole doc = one chunk |
| 2400       | 3              | 1.00  | 0.964 | identical to 1200 — docs already fit |

Smaller chunks win here (hybrid mode): at 400 chars each policy section is
its own chunk, so the right chunk ranks #1 instead of competing inside a
whole-document blob. 2400 changes nothing vs 1200 because every sample doc is
already under 1200 chars of paragraphs — the sweep needs longer docs to say
anything about the upper end.

## 3. Overlap on vs off

Overlap exists to keep boundary-straddling facts retrievable. Does removing
it measurably hurt on this corpus, or is it only visible on bigger docs?

```powershell
.\.venv\Scripts\python main.py ingest sample-docs --chunk 400 --no-overlap
.\.venv\Scripts\python main.py eval --mode hybrid
```

| config              | chunks indexed | hit@3 | MRR   | notes |
|---------------------|----------------|-------|-------|-------|
| 400 + overlap       | 11             | 1.00  | 1.000 |       |
| 400 no overlap      | 9              | 1.00  | 0.929 | paraphrase MRR 1.000 → 0.833 |

Measurable even on this tiny corpus: without the carried-over paragraph, two
paraphrase questions slip from rank 1 to rank 2 (the answering fact sits at a
section boundary and its chunk loses context that helped the embedding).
Overlap costs 2 extra chunks and buys a cleaner top rank — keep it on.

## 4. Retrieval depth for answering (`--top`)

More retrieved chunks = more chances the answer is in context, but more
noise and tokens. Compare answer quality by eye on a few questions:

```powershell
.\.venv\Scripts\python main.py ask --top 2 what is the training budget and does it roll over
.\.venv\Scripts\python main.py ask --top 8 what is the training budget and does it roll over
```

Notes (llama3.2:3b, 400-char index, 11 chunks):

- `--top 2`: both retrieved chunks were from expenses-policy.md; answer
  correct (4,000 PLN, no rollover) with citations [1][2].
- `--top 8`: 6 of the 8 chunks were noise from the other two docs; the model
  still ignored them, cited only [1], and answered correctly.
- On this corpus extra depth is pure token cost with no quality gain — but
  that's with one distractor doc pair; retest after adding distractors.

## 5. Answer model: small vs bigger local model

Same retrieval, different generator — does a 3B model respect "answer only
from the sources, cite them", and does stepping up to a 7B improve grounding?

```powershell
.\.venv\Scripts\python main.py ask which purchases need pre-approval
ollama pull qwen2.5:7b
$env:DOCRAG_LLM_MODEL = "qwen2.5:7b"; .\.venv\Scripts\python main.py ask which purchases need pre-approval
```

Notes:

- llama3.2:3b: correct answer ("above 1,000 PLN, department head"), named
  the source file and cited [1]. On these short, unambiguous policy
  questions the 3B model has been reliable so far — no hallucination
  observed yet.
- qwen2.5:7b: also correct and cited; noticeably more concise (one sentence
  vs llama's restated bullet lists on other questions).
- Answer-not-present test ("what is the parental leave policy" — not in any
  doc): BOTH models correctly said the sources don't contain it, no
  invention. llama padded with a summary of what the sources do cover; qwen
  answered in one line.
- Verdict so far: on this small, clean corpus the 3B model is not the weak
  link — retrieval quality dominates. Revisit after adding distractor docs
  and multi-source questions; that's where the 7B should separate.

## Backlog / ideas

- Add distractor documents to the corpus so retrieval has real competition —
  metrics on 3 files are too easy.
- Add answer-not-present questions to test the "say so, don't invent" rule.
- Sweep RRF k (currently 60) — does it matter at this corpus size?
- Mirror the eval harness in the .NET version and confirm rankings match.
