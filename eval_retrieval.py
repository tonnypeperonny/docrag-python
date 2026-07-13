"""Retrieval quality evaluation.

Scores each retrieval mode against a small labelled question set
(`evalset.jsonl`: question -> the file that contains the answer). Metrics:

- hit@k: fraction of questions where the expected file appears in the top k
- MRR:   mean reciprocal rank of the expected file (1/rank, 0 if missed)

Questions are tagged `keyword` (exact terms appear in the doc) or
`paraphrase` (asks in different words than the doc uses) so the results
show WHERE each mode wins, not just an overall average.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from search_index import SearchIndex

EVALSET_PATH = Path(__file__).parent / "evalset.jsonl"


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_file: str
    kind: str  # "keyword" | "paraphrase"


@dataclass(frozen=True)
class EvalResult:
    mode: str
    k: int
    hit_rate: float
    mrr: float
    by_kind: dict[str, tuple[float, float]]  # kind -> (hit_rate, mrr)
    misses: list[EvalCase]


def load_evalset(path: Path = EVALSET_PATH) -> list[EvalCase]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        cases.append(EvalCase(raw["question"], raw["expected_file"], raw.get("kind", "")))
    return cases


def evaluate(index: SearchIndex, mode: str, k: int = 3) -> EvalResult:
    cases = load_evalset()
    ranks: dict[EvalCase, int | None] = {}

    for case in cases:
        results = index.search(case.question, mode=mode, top_n=k)
        ranks[case] = next(
            (i for i, r in enumerate(results, start=1) if r.source_file == case.expected_file),
            None,
        )

    def metrics(subset: list[EvalCase]) -> tuple[float, float]:
        if not subset:
            return 0.0, 0.0
        hit = sum(1 for c in subset if ranks[c] is not None) / len(subset)
        mrr = sum(1.0 / ranks[c] for c in subset if ranks[c] is not None) / len(subset)
        return hit, mrr

    hit_rate, mrr = metrics(cases)
    kinds = sorted({c.kind for c in cases if c.kind})
    by_kind = {kind: metrics([c for c in cases if c.kind == kind]) for kind in kinds}
    misses = [c for c in cases if ranks[c] is None]

    return EvalResult(mode=mode, k=k, hit_rate=hit_rate, mrr=mrr, by_kind=by_kind, misses=misses)


def print_report(results: list[EvalResult]) -> None:
    k = results[0].k
    kinds = sorted({kind for r in results for kind in r.by_kind})

    header = f"{'mode':<8} {'hit@' + str(k):>7} {'MRR':>7}"
    for kind in kinds:
        header += f"   {kind + ' hit@' + str(k):>16} {kind + ' MRR':>14}"
    print(header)
    print("-" * len(header))

    for r in results:
        row = f"{r.mode:<8} {r.hit_rate:>7.2f} {r.mrr:>7.3f}"
        for kind in kinds:
            hit, mrr = r.by_kind.get(kind, (0.0, 0.0))
            row += f"   {hit:>16.2f} {mrr:>14.3f}"
        print(row)

    for r in results:
        if r.misses:
            print(f"\n{r.mode} missed:")
            for case in r.misses:
                print(f"  [{case.kind}] {case.question!r} (expected {case.expected_file})")
