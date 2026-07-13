"""Iterations-Analyse des Agentic-Prototyps (P2), wie oft lief die Retrieve-Reflect-Schleife,
mit welchem Abbruchgrund, getrennt nach Query-Typ 
Datenquelle runs_p2.jsonl, Feld raw.iterations / raw.stopReason.
der Qualitaetsvergleich zieht die per-query RAGAS-Scores aus ragas_raw_p{1,2}.json heran.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from statistics import fmean
from eval.config import EvalConfig
from eval.io import read_queries, read_runs

# Bevorzugte Metrik-Reihenfolge in den Tabellen (vorhandene zuerst, Rest alphabetisch angehängt).
_METRIC_ORDER = ("faithfulness", "answer_correctness", "answer_relevancy", "llm_context_precision_with_reference")
_NON_SCORE = ("query_id", "type")


def _norm_reason(reason: str) -> str:
	# Konkrete Iterationsnummer aus dem Abbruchgrund entfernen ("...nach Iteration 2" -> "...nach Iteration N"),
	# damit gleichartige Gruende über Queries hinweg zusammenfallen.
	return re.sub(r"\d+", "N", reason or "?")


def _nanmean(werte: list) -> float:
	gut = [w for w in werte if isinstance(w, (int, float)) and not math.isnan(w)]
	return fmean(gut) if gut else float("nan")


def _iterations(run: dict):
	# Iterationszahl aus dem Workflow-Output, None, wenn der Lauf sie nicht protokolliert hat.
	return (run.get("raw") or {}).get("iterations")


def iteration_distribution(runs_p2: dict, queries: dict) -> dict:
	"""Verteilung der Iterationszahlen über alle P2-Läufe, mit Abbruchgründen und Aufschlüsselung nach Typ."""
	by_count: dict = {}
	by_type: dict = {}
	for qid, run in runs_p2.items():
		n = _iterations(run)
		if n is None:
			continue
		reason = _norm_reason((run.get("raw") or {}).get("stopReason", "?"))
		bucket = by_count.setdefault(n, {"n": 0, "reasons": {}})
		bucket["n"] += 1
		bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
		typ = queries.get(qid, {}).get("type", "unbekannt")
		by_type.setdefault(typ, {})
		by_type[typ][n] = by_type[typ].get(n, 0) + 1
	return {"by_count": by_count, "by_type": by_type, "total": sum(b["n"] for b in by_count.values())}


def _metric_keys(ragas_per_query: list[dict]) -> list[str]:
	if not ragas_per_query:
		return []
	vorhanden = [k for k in ragas_per_query[0] if k not in _NON_SCORE]
	geordnet = [m for m in _METRIC_ORDER if m in vorhanden]
	return geordnet + [m for m in vorhanden if m not in geordnet]


def bucket_quality(runs_p2: dict, ragas_p2: dict, metric_keys: list[str]) -> dict:
	"""Mittlere P2-RAGAS-Scores je Iterations-Bucket (nur beantwortbare Queries, d.h. in ragas_p2 vorhanden)."""
	ids_by_n: dict = {}
	for qid, run in runs_p2.items():
		n = _iterations(run)
		if n is not None and qid in ragas_p2:
			ids_by_n.setdefault(n, []).append(qid)
	return {n: {"n": len(ids), **{m: _nanmean([ragas_p2[i].get(m) for i in ids]) for m in metric_keys}}
	        for n, ids in sorted(ids_by_n.items())}


def subset_compare(ids: list[str], ragas_p1: dict, ragas_p2: dict, metric_keys: list[str]) -> dict:
	"""P1-vs-P2-Vergleich auf identischer Query-Menge (Schwierigkeit konstant) -> Mittel + Delta je Metrik."""
	gem = [i for i in ids if i in ragas_p1 and i in ragas_p2]
	out = {"n": len(gem), "metrics": {}}
	for m in metric_keys:
		a = _nanmean([ragas_p1[i].get(m) for i in gem])
		b = _nanmean([ragas_p2[i].get(m) for i in gem])
		out["metrics"][m] = {"p1": a, "p2": b, "delta": b - a}
	return out


def analyze(runs_p2: dict, queries: dict, ragas_p1: list[dict], ragas_p2: list[dict]) -> dict:
	"""Komplette Iterations-Analyse aus bereits geladenen Daten."""
	mk = _metric_keys(ragas_p2)
	r1 = {r["query_id"]: r for r in ragas_p1}
	r2 = {r["query_id"]: r for r in ragas_p2}
	dist = iteration_distribution(runs_p2, queries)
	# Subset = Läufe mit der höchsten beobachteten Iterationszahl (dort lief die Schleife voll aus).
	max_n = max(dist["by_count"], default=None)
	max_ids = [qid for qid, run in runs_p2.items() if _iterations(run) == max_n] if max_n is not None else []
	return {
		"metric_keys": mk,
		"distribution": dist,
		"bucket_quality": bucket_quality(runs_p2, r2, mk),
		"max_iteration": max_n,
		"max_subset_compare": subset_compare(max_ids, r1, r2, mk),
	}


def _fmt(v) -> str:
	return f"{v:.3f}" if isinstance(v, (int, float)) and not math.isnan(v) else "n/a"


def render_markdown(a: dict) -> str:
	mk = a["metric_keys"]
	z = ["## Iterations-Analyse (Agentic P2)", ""]

	# 1) Verteilung + Abbruchgrund
	z += ["**Verteilung der Iterationen & Abbruchgrund**", "",
	      "| Iterationen | Queries | Abbruchgrund |", "|---|---|---|"]
	for n in sorted(a["distribution"]["by_count"]):
		b = a["distribution"]["by_count"][n]
		gruende = ", ".join(f"{r} ×{c}" for r, c in sorted(b["reasons"].items(), key=lambda x: -x[1]))
		z.append(f"| {n} | {b['n']} | {gruende} |")
	z.append("")

	# 2) Iterationen nach Query-Typ
	ns = sorted({n for d in a["distribution"]["by_type"].values() for n in d})
	z += ["**Iterationen nach Query-Typ**", "",
	      "| Query-Typ | " + " | ".join(f"iter {n}" for n in ns) + " |",
	      "|---|" + "---|" * len(ns)]
	for typ in sorted(a["distribution"]["by_type"]):
		d = a["distribution"]["by_type"][typ]
		z.append(f"| {typ} | " + " | ".join(str(d.get(n, 0)) for n in ns) + " |")
	z.append("")

	# 3) Qualität je Iterations-Bucket (P2, beantwortbar)
	z += ["**Qualität je Iterations-Bucket (P2, beantwortbar)**", "",
	      "| Iterationen | n | " + " | ".join(mk) + " |",
	      "|---|---|" + "---|" * len(mk)]
	for n, row in a["bucket_quality"].items():
		z.append(f"| {n} | {row['n']} | " + " | ".join(_fmt(row.get(m)) for m in mk) + " |")
	z.append("")

	# 4) Fairer Vergleich auf dem Subset mit maximaler Iterationszahl
	sc = a["max_subset_compare"]
	z += [f"**P1 vs. P2 auf dem Subset mit {a['max_iteration']} Iterationen (n={sc['n']}, Schwierigkeit konstant)**", "",
	      "| Metrik | Advanced (P1) | Agentic (P2) | Δ (P2−P1) |", "|---|---|---|---|"]
	for m in mk:
		c = sc["metrics"].get(m, {})
		d = c.get("delta")
		dtxt = f"{d:+.3f}" if isinstance(d, (int, float)) and not math.isnan(d) else "n/a"
		z.append(f"| {m} | {_fmt(c.get('p1'))} | {_fmt(c.get('p2'))} | {dtxt} |")
	z += ["", "*Hinweis: Das Subset mit maximaler Iterationszahl besteht aus den schwersten Queries "
	      "(Selektionsbias), niedrige Absolutwerte sind erwartbar. Aussagekräftig ist der P1-vs-P2-Δ "
	      "bei identischer Query-Menge, nicht der Vergleich zwischen den Buckets.*", ""]
	return "\n".join(z)


def section_from_cfg(cfg: EvalConfig) -> str | None:
	"""Lädt die nötigen Dateien (best effort) und rendert den Report-Block; None, wenn Daten fehlen."""
	runs_path = cfg.runs_dir / "runs_p2.jsonl"
	r1_path = cfg.results_dir / "ragas_raw_p1.json"
	r2_path = cfg.results_dir / "ragas_raw_p2.json"
	if not (runs_path.exists() and r1_path.exists() and r2_path.exists()):
		return None
	runs_p2 = read_runs(runs_path)
	queries = read_queries(cfg.queries_path)
	ragas_p1 = json.loads(r1_path.read_text(encoding="utf-8"))
	ragas_p2 = json.loads(r2_path.read_text(encoding="utf-8"))
	if not any(_iterations(r) is not None for r in runs_p2.values()):
		return None
	return render_markdown(analyze(runs_p2, queries, ragas_p1, ragas_p2))


def main() -> None:
	cfg = EvalConfig.load()
	md = section_from_cfg(cfg)
	if md is None:
		raise SystemExit("Iterations-Daten fehlen (runs_p2.jsonl mit raw.iterations + ragas_raw_p{1,2}.json).")
	out = cfg.results_dir / "iteration_analysis.md"
	out.write_text(md, encoding="utf-8")
	print(f"Iterations-Analyse geschrieben: {out}")


if __name__ == "__main__":
	main()
