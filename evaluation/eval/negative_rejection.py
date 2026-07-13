"""Negative-Rejection-Quote, enthalten sich die Prototypen bei unbeantwortbaren Queries korrekt?"""
from __future__ import annotations
import json
from pathlib import Path
from eval.config import EvalConfig
from eval.io import read_runs, read_queries


def _norm(s: str) -> str:
	# Lowercase, Whitespace kollabieren, Satzzeichen am Rand entfernen.
	return " ".join((s or "").lower().split()).strip(" .!?,;:")


def is_abstain(answer: str, abstain_phrase: str) -> bool:
	# Normalisierter Teilstring-Match, toleriert Gross/Klein, Rand-Whitespace und Rand-Satzzeichen.
	return _norm(abstain_phrase) in _norm(answer)


def _one(runs: dict, queries: dict, phrase: str) -> dict:
	unanswerable = [q for q, v in queries.items() if v.get("type") == "unanswerable"]
	answerable = [q for q, v in queries.items() if v.get("type") != "unanswerable"]
	korrekt = [q for q in unanswerable if q in runs and is_abstain(runs[q]["answer"], phrase)]
	false_abstain = [q for q in answerable if q in runs and is_abstain(runs[q]["answer"], phrase)]
	rate = (len(korrekt) / len(unanswerable)) if unanswerable else float("nan")
	return {
		"unanswerable_ids": sorted(unanswerable),
		"correct_abstain_ids": sorted(korrekt),
		"rejection_rate": rate,
		"false_abstain_answerable": sorted(false_abstain),
	}


def compute_negative_rejection(runs_p1: dict, runs_p2: dict, queries: dict, abstain_phrase: str) -> dict:
	return {"p1": _one(runs_p1, queries, abstain_phrase), "p2": _one(runs_p2, queries, abstain_phrase)}


def main() -> None:
	cfg = EvalConfig.load()
	queries = read_queries(cfg.queries_path)
	runs_p1 = read_runs(cfg.runs_dir / "runs_p1.jsonl")
	runs_p2 = read_runs(cfg.runs_dir / "runs_p2.jsonl")
	out = compute_negative_rejection(runs_p1, runs_p2, queries, cfg.abstain_phrase)
	cfg.results_dir.mkdir(parents=True, exist_ok=True)
	(cfg.results_dir / "neg_rejection.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"Negative-Rejection geschrieben: {cfg.results_dir / 'neg_rejection.json'}")


if __name__ == "__main__":
	main()
