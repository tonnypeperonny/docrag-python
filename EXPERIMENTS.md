# Retrieval experiments

Scenario log for tuning retrieval on the sample corpus. Metrics come from
`python main.py eval` (hit@k + MRR against `evalset.jsonl`, split into
keyword vs paraphrase questions). Re-run `ingest` between chunking scenarios —
the index only holds one chunking configuration at a time.

Results below: 2026-07-23, Elasticsearch 9.1.4, all-MiniLM-L6-v2, 23-question
evalset (13 keyword / 10 paraphrase), answers via llama3.2:3b on Ollama.
Everything runs locally — no hosted APIs.

## Corpus

Eight documents. Three carry the labelled answers (expenses-policy,
remote-work-policy, project-atlas-brief); the other five are distractors
added on purpose so retrieval has real competition instead of three docs on
three disjoint topics:

- `travel-booking-policy` — shares travel / hotel / meal / PLN / approval
  vocabulary with the expense policy.
- `procurement-policy` — shares purchase / pre-approval / software-license
  vocabulary with the expense policy.
- `project-beacon-brief` — a second cloud-migration brief (AWS/Kinesis vs
  Atlas's Azure/Service Bus); competes with every Atlas question.
- `information-security-policy` — shares confidential-data / device wording
  with the remote-work security section.
- `it-equipment-policy` — shares laptop / monitor / service-desk wording with
  the remote-work equipment section.

Each distractor is written to be lexically close but to *not* contain the
specific fact any labelled question asks for, so the labels stay
unambiguous while the retriever has to discriminate.

## 1. Retrieval mode: hybrid vs BM25-only vs kNN-only

Does RRF fusion actually beat either branch alone, and where does each
branch fail? Expectation: BM25 wins keyword questions, kNN wins paraphrase
questions, hybrid is the best overall.

```powershell
.\.venv\Scripts\python main.py ingest sample-docs
.\.venv\Scripts\python main.py eval --mode hybrid   # then --mode bm25, --mode knn
```

Default chunking (1200 chars → 8 chunks, one per doc):

| mode   | hit@3 | MRR   | keyword MRR | paraphrase MRR | misses |
|--------|-------|-------|-------------|----------------|--------|
| hybrid | 1.00  | 0.906 | 0.962       | 0.833          | none |
| bm25   | 0.96  | 0.891 | 0.923       | 0.850          | "nightly hotel limit" (keyword) |
| knn    | 0.91  | 0.841 | 0.885       | 0.783          | "receipt hand-in", "purchases pre-approval" |

With the distractors in, the branches now genuinely diverge — hit@3 is no
longer pinned at 1.00. Hybrid is the only mode that retrieves every answer in
the top 3; bm25 loses "nightly hotel limit" (the expense policy's meal/travel
lines out-score the travel policy on the shared tokens), and kNN drops two
where the wording is close across docs. Fusion is worth it precisely because
the two branches fail on different questions.

## 2. Chunk size sweep

Smaller chunks = more precise retrieval but less context per chunk for the
LLM; larger chunks = the opposite. Where is the sweet spot? (hybrid mode)

```powershell
.\.venv\Scripts\python main.py ingest sample-docs --chunk 400
.\.venv\Scripts\python main.py eval --mode hybrid
# repeat for 800, 1200 (default), 2400
```

| chunk size | chunks indexed | hit@3 | MRR   | notes |
|------------|----------------|-------|-------|-------|
| 400        | 28             | 1.00  | 0.928 | best — one section ≈ one chunk |
| 800        | 16             | 0.96  | 0.906 | loses "nightly hotel limit" |
| 1200       | 8              | 1.00  | 0.906 | whole doc = one chunk |
| 2400       | 8              | 1.00  | 0.906 | identical to 1200 — docs already fit |

400 is still the best config, and the reason is clearer now: at 400 chars
each policy section is its own chunk, so a distractor doc only competes on
the one section that overlaps rather than pulling its whole-document blob into
contention. 800 is the odd one out — merging two sections per chunk is enough
to let the expense policy's travel line beat the travel policy on "hotel
limit". 2400 changes nothing vs 1200 because every sample doc is already under
1200 chars of paragraphs.

## 3. Overlap on vs off

Overlap exists to keep boundary-straddling facts retrievable. Does removing
it measurably hurt? (400 chars, hybrid)

```powershell
.\.venv\Scripts\python main.py ingest sample-docs --chunk 400 --no-overlap
.\.venv\Scripts\python main.py eval --mode hybrid
```

| config          | chunks indexed | hit@3 | MRR   | paraphrase MRR |
|-----------------|----------------|-------|-------|----------------|
| 400 + overlap   | 28             | 1.00  | 0.928 | 0.900 |
| 400 no overlap  | 26             | 1.00  | 0.906 | 0.850 |

hit@3 is unchanged, but overlap still buys a cleaner top rank on paraphrase
questions (0.900 vs 0.850): without the carried-over paragraph, a couple of
answers whose fact sits at a section boundary slip from rank 1 to rank 2.
Two extra chunks for a measurably better MRR — keep it on.

## 4. Retrieval depth for answering (`--top`)

More retrieved chunks = more chances the answer is in context, but more
noise and tokens. Compare answer quality by eye (llama3.2:3b, 400-char index):

```powershell
.\.venv\Scripts\python main.py ask --top 2 what is the training budget and does it roll over
.\.venv\Scripts\python main.py ask --top 8 what is the training budget and does it roll over
```

- `--top 2`: chunks were expenses#2 and a travel-policy distractor; answer
  correct (4,000 PLN, no rollover), cited [1] and [2].
- `--top 8`: 6 of 8 chunks were distractors (travel, procurement, IT
  equipment, remote-work); the model ignored all of them, cited only [1], and
  answered correctly.
- Even with real noise in the window now, extra depth is still pure token cost
  on this corpus — the answer lives in one chunk and the model finds it. Depth
  would only pay off on questions that need to combine facts from two docs.

## 5. Answer model: small vs bigger local model

Same retrieval, different generator — does a 3B model respect "answer only
from the sources, cite them", and does a 7B improve grounding once distractor
chunks are actually landing in the context window?

```powershell
.\.venv\Scripts\python main.py ask which purchases need pre-approval
$env:DOCRAG_LLM_MODEL = "qwen2.5:7b"; .\.venv\Scripts\python main.py ask which purchases need pre-approval
```

Single-fact questions, with distractor chunks present in the retrieved set:

- "Atlas messaging technology" — retrieved set included the Beacon brief
  (Kinesis) at [3]. Both llama3.2:3b and qwen2.5:7b answered Azure Service Bus
  and cited [1]; neither was pulled toward the wrong project.
- "which purchases need pre-approval" — retrieved set included the procurement
  policy at [2]. Both models gave the expense-policy answer (above 1,000 PLN,
  department head) and cited [1]. qwen was a touch more concise, same content.

Where they both fall short — a genuinely ambiguous question spanning two docs:

- "how much can I spend on meals" — retrieval pulled *both* the expense
  policy's client-meal limit (200 PLN, [1]) and the travel policy's meal per
  diem (150 PLN, [3]). Both models answered only the top-ranked one (200 PLN
  client meals) and silently dropped the per diem. Neither conflated the two
  numbers (no "150–200 PLN" hallucination), but neither surfaced that there
  are two different meal allowances.

Verdict: with distractors landing in context, the 3B model still holds its
grounding on single-fact questions — it doesn't get distracted, so the 7B has
nothing to fix there. The real gap is completeness on questions with more than
one valid answer, and the 7B doesn't close it either — that's a
retrieval-presentation problem (surface and label both facts) more than a
model-size one. Next lever to try is multi-source questions where the answer
*must* combine two chunks.

## 6. Automated answer accuracy

Scenarios 4 and 5 judged answers by eye. This measures them: `answer-eval`
runs the full pipeline per question and checks the generated answer against
gold facts in `answerset.jsonl` (13 questions). A question passes only if the
answer contains every required fact (numbers matched comma-insensitively, so
"4,000" == "4000"); the parental-leave question passes only if the model
declines instead of inventing an answer. Matching is normalized substring —
crude, so read the printed failures, not just the number.

```powershell
.\.venv\Scripts\python main.py answer-eval
$env:DOCRAG_LLM_MODEL = "qwen2.5:7b"; .\.venv\Scripts\python main.py answer-eval
```

| model       | overall     | single-fact | multi-source |
|-------------|-------------|-------------|--------------|
| llama3.2:3b | 0.92 (12/13)| 1.00 (11/11)| 0.50 (1/2)   |
| qwen2.5:7b  | 0.92 (12/13)| 1.00 (11/11)| 0.50 (1/2)   |

The two models are identical, including *which* question they miss and *how*:
"how much can I spend on meals" — both answer only the 200 PLN client-meal
limit and drop the 150 PLN travel per diem. Crucially the per-diem chunk *was*
retrieved (rank 3 at top 5), so this is a generation-completeness gap, not a
retrieval miss, and the 7B doesn't close it. The refusal question passes for
both — the automated check confirms neither invents a parental-leave policy.

This puts a number on the scenario-5 finding: on single-fact questions the 3B
is already at ceiling (so the 7B has nothing to fix), and on the one question
with two valid answers, size doesn't help. The lever that would move
multi-source is prompting/formatting the model to enumerate every relevant
source, not a bigger model.

## Backlog / ideas

- Add more multi-source questions (the answer set has only two) and test
  whether a "list every applicable rule" system prompt lifts multi from 0.50.
- Sweep RRF k (currently 60) now that the branches disagree — does it move the
  hybrid rankings at all?
- Mirror both eval harnesses in the .NET version and confirm rankings match.
- Add a few more near-duplicate distractors (e.g. a second expense-style
  policy) to push bm25/kNN hit@3 further apart.
