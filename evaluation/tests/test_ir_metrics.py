import math
from pathlib import Path
import pytest
from eval.io import read_qrels, read_runs, read_queries
from eval.ir_metrics import run_dict_from_rows, evaluate_subset, compute_ir

FIX = Path(__file__).parent / "fixtures"
METRICS = ["precision@5", "recall@5", "ndcg@5"]


def test_evaluate_subset_gegen_handrechnung():
	qrels = read_qrels(FIX / "mini_qrels.txt")
	run = run_dict_from_rows(read_runs(FIX / "mini_runs_p1.jsonl"))
	# Nur beantwortbare Queries (q1, q2); q3 hat keine qrels -> ausgeschlossen.
	res = evaluate_subset(qrels, run, ["q1", "q2", "q3"], METRICS)
	# P@5: q1=2/5=0.4, q2=1/5=0.2 -> Mittel 0.3
	assert res["precision@5"] == pytest.approx(0.3, abs=1e-6)
	# R@5: q1=2/2=1.0, q2=1/1=1.0 -> 1.0
	assert res["recall@5"] == pytest.approx(1.0, abs=1e-6)
	# nDCG@5: q1 = (1 + 1/log2(4)) / (1 + 1/log2(3)) = 1.5/1.63093 = 0.91972; q2 = 1.0
	ndcg_q1 = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
	assert res["ndcg@5"] == pytest.approx((ndcg_q1 + 1.0) / 2, abs=1e-4)


def test_compute_ir_gruppiert_und_schliesst_unanswerable_aus():
	qrels = read_qrels(FIX / "mini_qrels.txt")
	runs = read_runs(FIX / "mini_runs_p1.jsonl")
	queries = read_queries(FIX / "mini_queries.jsonl")
	out = compute_ir(qrels, runs, runs, queries, k_values=(5, 10))
	# q3 (unanswerable) darf nicht in die Mittelwerte eingehen -> n der factual-Gruppe = 2
	assert out["n_per_group"]["by_type"]["factual"] == 2
	assert "unanswerable" not in out["n_per_group"]["by_type"]
	# identische Runs -> Signifikanz p ~ 1.0 (kein Unterschied)
	assert out["overall"]["p1"]["precision@5"] == pytest.approx(out["overall"]["p2"]["precision@5"])
