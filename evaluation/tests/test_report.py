from eval.report import build_report

IR = {
	"metrics": ["precision@5", "recall@5", "ndcg@5"],
	"overall": {
		"p1": {"precision@5": 0.4, "recall@5": 0.8, "ndcg@5": 0.7},
		"p2": {"precision@5": 0.5, "recall@5": 0.9, "ndcg@5": 0.75},
		"significance": {"precision@5": 0.02, "recall@5": 0.20, "ndcg@5": 0.04},
	},
	"by_type": {}, "by_source": {},
	"n_per_group": {"answerable_total": 63, "by_type": {}, "by_source": {}},
}
NEG = {"p1": {"rejection_rate": 0.5, "unanswerable_ids": ["q61", "q62"], "false_abstain_answerable": []},
       "p2": {"rejection_rate": 1.0, "unanswerable_ids": ["q61", "q62"], "false_abstain_answerable": []}}


def test_build_report_ohne_ragas_zeigt_hinweis():
	md, rows = build_report(IR, None, NEG, None)
	assert "RAGAS ausstehend" in md
	assert "precision@5" in md
	# Signifikanz-Marker bei p<0.05
	assert "*" in md
	assert any(r["metric"] == "precision@5" and r["prototype"] == "p2" for r in rows)


def test_build_report_mit_ragas():
	ragas = {"p1": {"overall": {"faithfulness": 0.8, "answer_relevancy": 0.7}, "coverage": {"faithfulness": 63}},
	         "p2": {"overall": {"faithfulness": 0.82, "answer_relevancy": 0.72}, "coverage": {"faithfulness": 63}}}
	md, rows = build_report(IR, ragas, NEG, None)
	assert "RAGAS ausstehend" not in md
	assert "faithfulness" in md.lower()
