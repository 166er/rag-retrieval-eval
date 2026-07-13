"""IR-Metriken (ranx) Precision@k, Recall@k, nDCG@k gesamt/pro Typ/pro Quelle + Signifikanz P1 vs P2."""
from __future__ import annotations
import json
from pathlib import Path
from ranx import Qrels, Run, evaluate, compare
from eval.config import EvalConfig
from eval.io import read_qrels, read_runs, read_queries


def metric_names(k_values: tuple[int, int]) -> list[str]:
	# ranx-Namensschema "precision@5", "recall@10", "ndcg@5", ...
	return [f"{m}@{k}" for m in ("precision", "recall", "ndcg") for k in k_values]


def run_dict_from_rows(runs: dict[str, dict]) -> dict[str, dict[str, float]]:
	# retrieved -> {query_id: {chunk_id: rerank_score}}, ranx ordnet nach Score absteigend.
	out: dict[str, dict[str, float]] = {}
	for qid, row in runs.items():
		scored = {}
		for c in row.get("retrieved", []):
			# fehlt rerank_score, über inversen Rang ordnen.
			scored[c["chunk_id"]] = float(c.get("rerank_score") if c.get("rerank_score") is not None else 1.0 / c["rank"])
		if scored:
			out[qid] = scored
	return out


def evaluate_subset(qrels: dict, run: dict, qids: list[str], metrics: list[str]) -> dict[str, float]:
	# Auf qids filtern, Queries ohne qrels-Eintrag (nicht beantwortbar) entfallen automatisch.
	q_sub = {q: qrels[q] for q in qids if q in qrels}
	r_sub = {q: run[q] for q in q_sub if q in run}
	if not q_sub or not r_sub:
		return {m: float("nan") for m in metrics}
	return evaluate(Qrels.from_dict(q_sub), Run.from_dict(r_sub), metrics)


def _significance(qrels: dict, run1: dict, run2: dict, metrics: list[str]) -> dict[str, float]:
	# p-Werte P1 vs P2 je Metrik via ranx.compare (Fisher-Randomisierung). Leere Eingabe -> NaN.
	if not qrels:
		return {m: float("nan") for m in metrics}
	r1 = Run.from_dict({q: run1[q] for q in qrels if q in run1}, name="p1")
	r2 = Run.from_dict({q: run2[q] for q in qrels if q in run2}, name="p2")
	rep = compare(Qrels.from_dict(qrels), runs=[r1, r2], metrics=metrics, max_p=0.05, stat_test="fisher")
	# ranx 0.3.20 hat kein get_pvalue(), die p-Werte stehen in rep.to_dict()["p1"]["comparisons"]["p2"][metric].
	comp = rep.to_dict()["p1"]["comparisons"]["p2"]
	return {m: float(comp[m]) for m in metrics}


def _groups(queries: dict, key: str) -> dict[str, list[str]]:
	# Gruppiert query_ids nach 'type' bzw. nach gejointem 'source_target'.
	g: dict[str, list[str]] = {}
	for qid, q in queries.items():
		if key == "type":
			val = q.get("type", "unbekannt")
		else:
			st = q.get("source_target") or ["unbekannt"]
			val = ",".join(st)
		g.setdefault(val, []).append(qid)
	return g


def compute_ir(qrels: dict, runs_p1: dict, runs_p2: dict, queries: dict, k_values=(5, 10)) -> dict:
	metrics = metric_names(k_values)
	run1, run2 = run_dict_from_rows(runs_p1), run_dict_from_rows(runs_p2)
	# Beantwortbare Menge = Queries mit nicht-leeren qrels.
	answerable = [q for q in queries if q in qrels]

	def both(qids: list[str]) -> dict:
		return {
			"p1": evaluate_subset(qrels, run1, qids, metrics),
			"p2": evaluate_subset(qrels, run2, qids, metrics),
		}

	# Signifikanz nur auf der Gesamt-(beantwortbaren)-Menge.
	sig = _significance({q: qrels[q] for q in answerable}, run1, run2, metrics)

	out = {
		"metrics": metrics,
		"overall": {**both(answerable), "significance": sig},
		"by_type": {t: both(qids) for t, qids in _groups(queries, "type").items() if any(q in qrels for q in qids)},
		"by_source": {s: both(qids) for s, qids in _groups(queries, "source").items() if any(q in qrels for q in qids)},
		"n_per_group": {
			"answerable_total": len(answerable),
			"by_type": {t: sum(q in qrels for q in qids) for t, qids in _groups(queries, "type").items() if any(q in qrels for q in qids)},
			"by_source": {s: sum(q in qrels for q in qids) for s, qids in _groups(queries, "source").items() if any(q in qrels for q in qids)},
		},
	}
	return out


def main() -> None:
	cfg = EvalConfig.load()
	qrels = read_qrels(cfg.qrels_path)
	queries = read_queries(cfg.queries_path)
	runs_p1 = read_runs(cfg.runs_dir / "runs_p1.jsonl")
	runs_p2 = read_runs(cfg.runs_dir / "runs_p2.jsonl")
	out = compute_ir(qrels, runs_p1, runs_p2, queries, cfg.k_values)
	cfg.results_dir.mkdir(parents=True, exist_ok=True)
	(cfg.results_dir / "ir.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"IR-Metriken geschrieben: {cfg.results_dir / 'ir.json'}")


if __name__ == "__main__":
	main()
