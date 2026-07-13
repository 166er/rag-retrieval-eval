from pathlib import Path
from eval.io import read_runs, read_queries
from eval.negative_rejection import is_abstain, compute_negative_rejection

FIX = Path(__file__).parent / "fixtures"
PHRASE = "Dazu liegen mir keine Informationen vor."


def test_is_abstain_robust():
	assert is_abstain("Dazu liegen mir keine Informationen vor.", PHRASE)
	assert is_abstain("  dazu LIEGEN mir keine informationen vor  ", PHRASE)
	assert not is_abstain("Den Parameter WARTUNGSMODUS auf AUTO setzen.", PHRASE)


def test_compute_negative_rejection_quote():
	runs = read_runs(FIX / "mini_runs_p1.jsonl")
	queries = read_queries(FIX / "mini_queries.jsonl")
	out = compute_negative_rejection(runs, runs, queries, PHRASE)
	# q3 ist die einzige unanswerable, P1 enthält sich korrekt -> rate 1.0
	assert out["p1"]["unanswerable_ids"] == ["q3"]
	assert out["p1"]["rejection_rate"] == 1.0
	# q1/q2 sind beantwortbar und enthalten sich NICHT -> keine false abstains
	assert out["p1"]["false_abstain_answerable"] == []
