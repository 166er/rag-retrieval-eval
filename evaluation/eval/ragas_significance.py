"""RAGAS-Signifikanz-Stufe: gepaarter Randomisierungstest + Streuung P1 vs P2 auf den Roh-Scores."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from statistics import fmean
from eval.config import EvalConfig

# Score-Spalten der ragas_raw_p{1,2}.json
METRIC_KEYS = ("faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "answer_correctness")
# Monte-Carlo-Parameter: fester Seed + hohe Rundenzahl -> deterministisch reproduzierbare p-Werte.
SEED = 42
N_PERMUTATIONS = 100_000


def _is_score(wert) -> bool:
	# Gültiger Score = Zahl und nicht NaN (vgl. Coverage-Logik in eval.ragas_eval).
	return isinstance(wert, (int, float)) and not math.isnan(wert)


def read_raw(pfad: Path) -> dict[str, dict]:
	# Roh-Scores je Query laden, nicht beantwortbare Anfragen ausklammern 
    daten = json.loads(pfad.read_text(encoding="utf-8"))
	return {r["query_id"]: r for r in daten if r.get("type") != "unanswerable"}


def describe(werte: list[float]) -> dict:
	# Mittelwert, Stichproben-Standardabweichung (n-1) und 95%-Konfidenzintervall (Normalapproximation).
	n = len(werte)
	if n < 2:
		return {"mean": fmean(werte) if werte else float("nan"), "sd": float("nan"), "ci95": [float("nan"), float("nan")]}
	m = fmean(werte)
	sd = math.sqrt(sum((w - m) ** 2 for w in werte) / (n - 1))
	halbweite = 1.96 * sd / math.sqrt(n)
	return {"mean": m, "sd": sd, "ci95": [m - halbweite, m + halbweite]}


def paired_permutation_test(a: list[float], b: list[float], n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> float:
	# gepaarter Randomisierungstest (Fisher-Prinzip wie beim IR-Test, sihe eval.ir_metrics)
	# Vorzeichen der Query-Differenzen zufällig tauschen und zählen, wie oft eine mindestens so grosse mittlere Differenz wie die beobachtete entsteht.
	if not a or len(a) != len(b):
		return float("nan")
	rng = random.Random(seed)
	diffs = [x - y for x, y in zip(a, b)]
	beobachtet = abs(fmean(diffs))
	treffer = 0
	for _ in range(n_permutations):
		summe = sum(d if rng.random() < 0.5 else -d for d in diffs)
		if abs(summe / len(diffs)) >= beobachtet:
			treffer += 1
	return treffer / n_permutations


def compute_significance(raw_p1: dict[str, dict], raw_p2: dict[str, dict], metric_keys=METRIC_KEYS,
						 n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> dict:
	# Nur Queries, die in beiden Läufen vorliegen. je Metrik fallen Paare mit NaN/fehlendem Wert raus.
	ids = sorted(set(raw_p1) & set(raw_p2))
	out = {"n_queries": len(ids), "seed": seed, "n_permutations": n_permutations, "metrics": {}}
	for mk in metric_keys:
		paare = [(raw_p1[q].get(mk), raw_p2[q].get(mk)) for q in ids]
		gueltig = [(float(x), float(y)) for x, y in paare if _is_score(x) and _is_score(y)]
		a = [x for x, _ in gueltig]		# P1 (Advanced RAG)
		b = [y for _, y in gueltig]		# P2 (Agentic RAG)
		out["metrics"][mk] = {
			"n": len(gueltig),
			"p1": describe(a),
			"p2": describe(b),
			"delta_mean_p2_minus_p1": (fmean(b) - fmean(a)) if gueltig else float("nan"),
			"p_value": paired_permutation_test(a, b, n_permutations, seed),
		}
	return out


def main() -> None:
	cfg = EvalConfig.load()
	raw_p1 = read_raw(cfg.results_dir / "ragas_raw_p1.json")
	raw_p2 = read_raw(cfg.results_dir / "ragas_raw_p2.json")
	out = compute_significance(raw_p1, raw_p2)
	ziel = cfg.results_dir / "ragas_significance.json"
	ziel.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"RAGAS-Signifikanz geschrieben: {ziel}")


if __name__ == "__main__":
	main()
