import math
from eval.iteration_analysis import (
	iteration_distribution, bucket_quality, subset_compare, analyze, render_markdown, _norm_reason,
)

# Mini-Fixtures: 3 Läufe, zwei Typen, Iterationen 1/1/3.
RUNS = {
	"q1": {"query_id": "q1", "raw": {"iterations": 1, "stopReason": "Kontext ausreichend nach Iteration 1"}},
	"q2": {"query_id": "q2", "raw": {"iterations": 3, "stopReason": "Iterationslimit erreicht"}},
	"q3": {"query_id": "q3", "raw": {"iterations": 1, "stopReason": "Kontext ausreichend nach Iteration 1"}},
}
QUERIES = {"q1": {"type": "factual"}, "q2": {"type": "multihop"}, "q3": {"type": "factual"}}
RAGAS_P1 = [
	{"query_id": "q1", "type": "factual", "faithfulness": 0.9, "answer_correctness": 0.6},
	{"query_id": "q2", "type": "multihop", "faithfulness": 0.8, "answer_correctness": 0.5},
	{"query_id": "q3", "type": "factual", "faithfulness": 0.7, "answer_correctness": 0.4},
]
RAGAS_P2 = [
	{"query_id": "q1", "type": "factual", "faithfulness": 1.0, "answer_correctness": 0.7},
	{"query_id": "q2", "type": "multihop", "faithfulness": 0.6, "answer_correctness": 0.3},
	{"query_id": "q3", "type": "factual", "faithfulness": 0.8, "answer_correctness": 0.5},
]


def test_norm_reason_entfernt_iterationsnummer():
	assert _norm_reason("Kontext ausreichend nach Iteration 2") == "Kontext ausreichend nach Iteration N"


def test_iteration_distribution_zaehlt_und_gruppiert_nach_typ():
	d = iteration_distribution(RUNS, QUERIES)
	assert d["total"] == 3
	assert d["by_count"][1]["n"] == 2
	assert d["by_count"][3]["n"] == 1
	assert d["by_count"][3]["reasons"]["Iterationslimit erreicht"] == 1
	assert d["by_type"]["factual"] == {1: 2}
	assert d["by_type"]["multihop"] == {3: 1}


def test_bucket_quality_mittelt_nur_beantwortbare():
	r2 = {r["query_id"]: r for r in RAGAS_P2}
	bq = bucket_quality(RUNS, r2, ["faithfulness"])
	assert bq[1]["n"] == 2
	assert bq[1]["faithfulness"] == 0.9          # (1.0 + 0.8) / 2
	assert bq[3]["faithfulness"] == 0.6


def test_subset_compare_liefert_delta_auf_gemeinsamer_menge():
	r1 = {r["query_id"]: r for r in RAGAS_P1}
	r2 = {r["query_id"]: r for r in RAGAS_P2}
	sc = subset_compare(["q2"], r1, r2, ["faithfulness", "answer_correctness"])
	assert sc["n"] == 1
	assert math.isclose(sc["metrics"]["faithfulness"]["delta"], 0.6 - 0.8)
	assert math.isclose(sc["metrics"]["answer_correctness"]["delta"], 0.3 - 0.5)


def test_analyze_und_render_erzeugen_vollstaendigen_block():
	a = analyze(RUNS, QUERIES, RAGAS_P1, RAGAS_P2)
	assert a["max_iteration"] == 3
	assert a["max_subset_compare"]["n"] == 1          # nur q2 lief 3 Iterationen
	md = render_markdown(a)
	assert "## Iterations-Analyse (Agentic P2)" in md
	assert "Iterationslimit erreicht" in md
	assert "Δ (P2−P1)" in md
