"""Offline-Smoke-Test, ir -> neg -> report auf einem Mini-Fixture, ohne Dienste."""
from __future__ import annotations
import csv
from pathlib import Path
from dataclasses import replace
from eval.io import read_qrels, read_runs, read_queries
from eval.ir_metrics import compute_ir
from eval.negative_rejection import compute_negative_rejection
from eval import report as report_mod


class _Cfg:
	# Config-Stub für die Offline-Kette (kein .env nötig).
	def __init__(self, results_dir: Path, runs_dir: Path):
		self.results_dir = results_dir
		self.runs_dir = runs_dir
		self.k_values = (5, 10)


def run_offline_smoke(fixtures_dir: Path, out_dir: Path) -> Path:
	qrels = read_qrels(fixtures_dir / "mini_qrels.txt")
	queries = read_queries(fixtures_dir / "mini_queries.jsonl")
	runs = read_runs(fixtures_dir / "mini_runs_p1.jsonl")
	ir = compute_ir(qrels, runs, runs, queries, (5, 10))
	neg = compute_negative_rejection(runs, runs, queries, "Dazu liegen mir keine Informationen vor.")
	cfg = _Cfg(out_dir, out_dir)
	report_mod.write_outputs(cfg, ir, None, neg, None)
	return out_dir / "report.md"


def main() -> None:
	fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
	out = Path(__file__).resolve().parents[1] / "results" / "smoke"
	out.mkdir(parents=True, exist_ok=True)
	print(f"Smoke-Report: {run_offline_smoke(fixtures, out)}")


if __name__ == "__main__":
	main()
