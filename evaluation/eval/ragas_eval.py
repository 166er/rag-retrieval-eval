"""RAGAS-Stufe"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from statistics import fmean
from eval.config import EvalConfig
from eval.io import read_runs, read_queries, read_jsonl

# Die zwei teuren Mehrfach-Call-Metriken. Auf dem GB10-Judge brauchen sie auf langen Inputs ~900-1100 s -> 
# nur diese laufen in Timeouts und sind im --only-missing-Modus nachzurechnen.
HEAVY_METRICS = ("faithfulness", "answer_correctness")
# Felder, die KEIN Score sind und bei Coverage/Update ausgeklammert werden.
_NON_SCORE = ("query_id", "type")
_INPUT_FIELDS = ("user_input", "retrieved_contexts", "response", "reference")
# Pro-Job-Timeout, faithfulness = ~1093 s, unter max_workers=4 dehnt sich ein Job auf das
# 1,5-3-fache -> 5400 s (90 min) gibt sicheren Puffer, damit auch der längste aggregate-Job durchläuft.
RUN_TIMEOUT = 5400
RUN_MAX_WORKERS = 4


def build_dataset_rows(runs: dict, queries: dict, references: dict, answerable_only: bool = True) -> list[dict]:
	rows = []
	for qid, q in queries.items():
		if answerable_only and q.get("type") == "unanswerable":
			continue
		if qid not in runs:
			continue
		row = runs[qid]
		rows.append({
			"query_id": qid,
			"type": q.get("type", "unbekannt"),
			"user_input": q["text"],
			"retrieved_contexts": [c["text"] for c in row.get("contexts", [])],
			"response": row.get("answer", ""),
			"reference": references.get(qid, ""),
		})
	return rows


def _nanmean(werte: list[float]) -> float:
	gut = [w for w in werte if isinstance(w, (int, float)) and not math.isnan(w)]
	return fmean(gut) if gut else float("nan")


def aggregate_scores(per_query: list[dict], metric_keys: list[str]) -> dict:
	overall = {m: _nanmean([r.get(m, float("nan")) for r in per_query]) for m in metric_keys}
	coverage = {m: sum(1 for r in per_query if isinstance(r.get(m), (int, float)) and not math.isnan(r.get(m))) for m in metric_keys}
	typen = sorted({r["type"] for r in per_query})
	by_type = {t: {m: _nanmean([r.get(m, float("nan")) for r in per_query if r["type"] == t]) for m in metric_keys} for t in typen}
	return {"overall": overall, "coverage": coverage, "by_type": by_type}


# Manche Judge-Modelle escapen Nicht-ASCII-Zeichen Python-Style als \xHH statt JSON-konform.
# \x ist in JSON KEIN gueltiges Escape -> json.loads scheitert -> RAGAS wirft OutputParserException
# z.B. q016: "Verschluesselung" -> ...Verschl\xfcsselung...). \xHH
# kodiert den Latin-1-Codepoint U+00HH, also ist \xHH -> \u00HH verlustfrei (\xfc == ü == ü).
# Negative Lookbehind + Paar-Gruppe schützen bereits escapte Backslashes (\\xHH = Backslash + xHH).
_INVALID_HEX_ESCAPE = re.compile(r'(?<!\\)((?:\\\\)*)\\x([0-9a-fA-F]{2})')


def _repair_json_escapes(text: str) -> str:
	"""Wandelt ungültige \\xHH-Escapes in JSON-konforme \\u00HH um."""
	return _INVALID_HEX_ESCAPE.sub(lambda m: m.group(1) + r'\u00' + m.group(2), text)


def _json_safe_wrapper_cls():
	# Lazy: erst hier RAGAS importieren (Modul-Import bleibt RAGAS-frei -> damit Offline-Tests laufen).
	from ragas.llms import LangchainLLMWrapper

	class _JsonSafeLLMWrapper(LangchainLLMWrapper):
		"""Wrapper, der den rohen LLM-Text repariert, bevor RAGAS ihn an json.loads gibt.
		RAGAS ab 0.2 liest generations[0][i].text -> genau dort die \\xHH-Escapes verbessern."""

		@staticmethod
		def _repair_result(result):
			for gens in result.generations:
				for g in gens:
					if getattr(g, "text", None):
						g.text = _repair_json_escapes(g.text)
					msg = getattr(g, "message", None)
					if msg is not None and isinstance(getattr(msg, "content", None), str):
						msg.content = _repair_json_escapes(msg.content)
			return result

		def generate_text(self, *args, **kwargs):
			return self._repair_result(super().generate_text(*args, **kwargs))

		async def agenerate_text(self, *args, **kwargs):
			return self._repair_result(await super().agenerate_text(*args, **kwargs))

	return _JsonSafeLLMWrapper


def _build_judge_llm(cfg: EvalConfig):
	# Qwen3-32B via vLLM. temperature=0 + seed -> Determinismus auf Request-Ebene.
	from langchain_openai import ChatOpenAI
	# Qwen3 hat Reasoning per Default AN -> generiert vor jeder Antwort eine lange <think>-Kette
	# (~3-4 min/Job -> TimeoutErrors). Per chat_template_kwargs serverseitig abschalten; vLLM
	# reicht extra_body unverändert an das Qwen3-Chat-Template weiter.
	llm = ChatOpenAI(model=cfg.judge_model, base_url=cfg.judge_base_url, api_key=cfg.judge_api_key,
	                 temperature=0, model_kwargs={"seed": 42},
	                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
	# JSON-Safe-Wrapper: verbessert ungültige \xHH-Escapes des Judge-Outputs.
	return _json_safe_wrapper_cls()(llm)


def _build_embeddings(cfg: EvalConfig):
	from ragas.embeddings import LangchainEmbeddingsWrapper
	from eval.adapters.bge_m3_embeddings import BgeM3Embeddings
	return LangchainEmbeddingsWrapper(BgeM3Embeddings(cfg.embed_base_url))


def _preflight(cfg: EvalConfig) -> None:
	# Sichert den manuellen Modell-Swap ab, Judge + Embeddings müssen erreichbar sein.
	import requests
	for name, url in [("Judge", f"{cfg.judge_base_url.rstrip('/')}/models"), ("Embeddings", f"{cfg.embed_base_url.rstrip('/')}/health")]:
		try:
			requests.get(url, timeout=10).raise_for_status()
		except Exception as e:
			raise SystemExit(f"Preflight fehlgeschlagen: {name} ({url}) nicht erreichbar: {e}")
	# Warnung, falls der Generator noch läuft (Speicherkonflikt, R02).
	try:
		requests.get("http://" + cfg.p1_webhook_url.split("//", 1)[1].split("/", 1)[0], timeout=3)
		print("ACHTUNG, Prüfe, ob der Generator (GPU-Host:5678) heruntergefahren ist, sonst Speicherkonflikt mit dem Judge.")
	except Exception:
		pass


def _evaluate_prototype(rows: list[dict], cfg: EvalConfig) -> list[dict]:
	from ragas import evaluate
	from ragas.run_config import RunConfig
	from ragas.dataset_schema import EvaluationDataset
	from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, AnswerCorrectness
	llm, emb = _build_judge_llm(cfg), _build_embeddings(cfg)
	metrics = [Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=emb),
	           LLMContextPrecisionWithReference(llm=llm), AnswerCorrectness(llm=llm, embeddings=emb)]
	ds = EvaluationDataset.from_list([{k: r[k] for k in _INPUT_FIELDS} for r in rows])
	# Bandbreitenlimitierter GB10-Judge, lange Mehrfach-Call-Jobs brauchen bis ~2700 s unter Last.
	# Timeout grosszügig (RUN_TIMEOUT), damit kein langer Job abgeschnitten wird (= NaN). 
    # Parallelität für Durchsatz (vLLM batcht, amortisiert das Gewichts-Laden).
	run_config = RunConfig(timeout=RUN_TIMEOUT, max_workers=RUN_MAX_WORKERS)
	result = evaluate(dataset=ds, metrics=metrics, run_config=run_config)
	df = result.to_pandas()
	# Pro-Query-Scores zurück an query_id/type hängen (Reihenfolge bleibt erhalten).
	per_query = []
	for r, (_, drow) in zip(rows, df.iterrows()):
		entry = {"query_id": r["query_id"], "type": r["type"]}
		for col in df.columns:
			if col not in _INPUT_FIELDS:
				entry[col] = float(drow[col]) if drow[col] == drow[col] else float("nan")
		per_query.append(entry)
	return per_query


def run_ragas(cfg: EvalConfig) -> dict:
	_preflight(cfg)
	queries = read_queries(cfg.queries_path)
	references = {r["query_id"]: r["answer"] for r in read_jsonl(cfg.reference_answers_path)}
	out = {}
	for proto, fname in (("p1", "runs_p1.jsonl"), ("p2", "runs_p2.jsonl")):
		runs = read_runs(cfg.runs_dir / fname)
		rows = build_dataset_rows(runs, queries, references, answerable_only=True)
		per_query = _evaluate_prototype(rows, cfg)
		metric_keys = [k for k in per_query[0] if k not in _NON_SCORE] if per_query else []
		# Judge-Outputs per-query persistieren.
		(cfg.results_dir).mkdir(parents=True, exist_ok=True)
		(cfg.results_dir / f"ragas_raw_{proto}.json").write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
		out[proto] = aggregate_scores(per_query, metric_keys)
	return out


def _is_missing(value) -> bool:
	# Zelle gilt als fehlend, wenn kein Zahlenwert ODER NaN (RAGAS schreibt für abgebrochene Jobs NaN).
	return not (isinstance(value, (int, float)) and not math.isnan(value))


def find_missing(per_query: list[dict], metric_keys: list[str]) -> dict:
	"""Pro Metrik die query_ids mit fehlendem/NaN-Wert (Reihenfolge wie in per_query)."""
	return {m: [r["query_id"] for r in per_query if _is_missing(r.get(m))] for m in metric_keys}


def apply_updates(per_query: list[dict], updates: dict) -> list[dict]:
	"""Setzt nachgerechnete Werte (updates: {query_id: {metric: value}}) in per_query."""
	by_id = {r["query_id"]: r for r in per_query}
	for qid, vals in updates.items():
		if qid in by_id:
			by_id[qid].update(vals)
	return per_query


def _recompute_metric(rows_subset: list[dict], metric_name: str, cfg: EvalConfig) -> dict:
	# Eine einzelne teure Metrik auf einer Teilmenge von Rows nachrechnen -> {query_id: value}.
	from ragas import evaluate
	from ragas.run_config import RunConfig
	from ragas.dataset_schema import EvaluationDataset
	from ragas.metrics import Faithfulness, AnswerCorrectness
	llm = _build_judge_llm(cfg)
	if metric_name == "faithfulness":
		metric = Faithfulness(llm=llm)
	elif metric_name == "answer_correctness":
		metric = AnswerCorrectness(llm=llm, embeddings=_build_embeddings(cfg))
	else:
		raise ValueError(f"Nicht unterstützte Metrik für Nachlauf: {metric_name}")
	ds = EvaluationDataset.from_list([{k: r[k] for k in _INPUT_FIELDS} for r in rows_subset])
	result = evaluate(dataset=ds, metrics=[metric],
	                  run_config=RunConfig(timeout=RUN_TIMEOUT, max_workers=RUN_MAX_WORKERS))
	df = result.to_pandas()
	out = {}
	for r, (_, drow) in zip(rows_subset, df.iterrows()):
		val = drow[metric_name]
		out[r["query_id"]] = float(val) if val == val else float("nan")
	return out


def rerun_missing(cfg: EvalConfig, metrics: tuple = HEAVY_METRICS) -> dict:
	"""Füllt nur NaN-Zellen der teuren Metriken nach (gleiche Judge-Config, deterministisch ->
	konsistent mit den vorhandenen Werten). Schreibt vorher ein .bak je Prototyp."""
	_preflight(cfg)
	queries = read_queries(cfg.queries_path)
	references = {r["query_id"]: r["answer"] for r in read_jsonl(cfg.reference_answers_path)}
	out = {}
	for proto, fname in (("p1", "runs_p1.jsonl"), ("p2", "runs_p2.jsonl")):
		raw_path = cfg.results_dir / f"ragas_raw_{proto}.json"
		raw_text = raw_path.read_text(encoding="utf-8")
		per_query = json.loads(raw_text)   # json akzeptiert NaN-Literale -> float('nan')
		runs = read_runs(cfg.runs_dir / fname)
		rows_by_id = {r["query_id"]: r for r in build_dataset_rows(runs, queries, references, answerable_only=True)}
		missing = find_missing(per_query, list(metrics))
		updates: dict = {}
		for metric_name, ids in missing.items():
			subset = [rows_by_id[i] for i in ids if i in rows_by_id]
			print(f"[{proto}] {metric_name}: {len(subset)} fehlende Zellen werden nachgerechnet")
			if not subset:
				continue
			for qid, v in _recompute_metric(subset, metric_name, cfg).items():
				updates.setdefault(qid, {})[metric_name] = v
		# Backup VOR dem Überschreiben (die guten Werte dürfen nie verloren gehen).
		(raw_path.parent / f"ragas_raw_{proto}.bak.json").write_text(raw_text, encoding="utf-8")
		apply_updates(per_query, updates)
		raw_path.write_text(json.dumps(per_query, ensure_ascii=False, indent=2), encoding="utf-8")
		metric_keys = [k for k in per_query[0] if k not in _NON_SCORE] if per_query else []
		out[proto] = aggregate_scores(per_query, metric_keys)
	return out


def main() -> None:
	import sys
	cfg = EvalConfig.load()
	only_missing = "--only-missing" in sys.argv
	out = rerun_missing(cfg) if only_missing else run_ragas(cfg)
	(cfg.results_dir / "ragas.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
	print(f"RAGAS geschrieben: {cfg.results_dir / 'ragas.json'}{' (nur Lücken nachgefüllt)' if only_missing else ''}")


if __name__ == "__main__":
	main()
