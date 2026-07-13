import json
import math
from pathlib import Path
import pytest
from eval.io import read_runs, read_queries, read_jsonl
from eval.ragas_eval import build_dataset_rows, aggregate_scores, find_missing, apply_updates, _repair_json_escapes

FIX = Path(__file__).parent / "fixtures"


def test_build_dataset_rows_schliesst_unanswerable_aus_und_mappt_felder():
	runs = read_runs(FIX / "mini_runs_p1.jsonl")
	queries = read_queries(FIX / "mini_queries.jsonl")
	refs = {r["query_id"]: r["answer"] for r in read_jsonl(FIX / "mini_reference_answers.jsonl")}
	rows = build_dataset_rows(runs, queries, refs, answerable_only=True)
	ids = {r["query_id"] for r in rows}
	assert ids == {"q1", "q2"}                       # q3 (unanswerable) raus
	r1 = next(r for r in rows if r["query_id"] == "q1")
	assert r1["user_input"].startswith("Wie aktiviert")
	assert r1["retrieved_contexts"] == ["a", "x", "b"]   # Texte aus contexts
	assert r1["reference"].startswith("Den Konfigurationsparameter")


def test_aggregate_scores_nanmean_und_coverage():
	per_query = [
		{"type": "factual", "faithfulness": 0.8, "answer_relevancy": 0.6},
		{"type": "factual", "faithfulness": float("nan"), "answer_relevancy": 0.4},
		{"type": "multihop", "faithfulness": 1.0, "answer_relevancy": 1.0},
	]
	agg = aggregate_scores(per_query, ["faithfulness", "answer_relevancy"])
	assert agg["overall"]["faithfulness"] == 0.9          # (0.8+1.0)/2, NaN ignoriert
	assert agg["coverage"]["faithfulness"] == 2           # 2 von 3 bewertet
	assert agg["by_type"]["factual"]["answer_relevancy"] == 0.5


def test_find_missing_listet_nan_und_fehlende_zellen():
	per_query = [
		{"query_id": "q1", "type": "factual", "faithfulness": 0.8, "answer_correctness": float("nan")},
		{"query_id": "q2", "type": "multihop", "faithfulness": float("nan"), "answer_correctness": 0.5},
		{"query_id": "q3", "type": "aggregate", "faithfulness": 1.0},   # answer_correctness fehlt ganz
	]
	missing = find_missing(per_query, ["faithfulness", "answer_correctness"])
	assert missing["faithfulness"] == ["q2"]
	assert missing["answer_correctness"] == ["q1", "q3"]   # NaN UND fehlender Schlüssel


def test_apply_updates_fuellt_nur_luecken_und_laesst_gute_werte_unangetastet():
	per_query = [
		{"query_id": "q1", "type": "factual", "faithfulness": 0.8, "answer_correctness": float("nan")},
		{"query_id": "q2", "type": "multihop", "faithfulness": float("nan"), "answer_correctness": 0.5},
	]
	apply_updates(per_query, {"q1": {"answer_correctness": 0.42}, "q2": {"faithfulness": 0.9}})
	assert per_query[0]["answer_correctness"] == 0.42
	assert per_query[0]["faithfulness"] == 0.8            # unverändert
	assert per_query[1]["faithfulness"] == 0.9
	assert per_query[1]["answer_correctness"] == 0.5      # unverändert


# Exakter q016-Judge-Output: der Judge escapt das "ue" in "Verschluesselung" Python-Style als \xfc.
# In JSON ist \x KEIN gueltiges Escape -> json.loads scheitert -> RAGAS OutputParserException (NaN).
_Q016_BAD = r'''{
    "statements": [
        "To enable TLS in OpenSearch, the two options must be set to true.",
        "The source is 11 TLS-Verschl\xfcsselung zwischen allen Komponenten."
    ]
}'''


def test_q016_output_ist_ohne_reparatur_ungueltiges_json():
	# Belegt den Root Cause: der rohe Judge-Output ist genau wegen \xfc nicht parsebar.
	with pytest.raises(json.JSONDecodeError):
		json.loads(_Q016_BAD)


def test_repair_json_escapes_macht_q016_parsebar_und_erhaelt_umlaut():
	obj = json.loads(_repair_json_escapes(_Q016_BAD))   # darf nicht mehr werfen
	# \xfc wurde verlustfrei zu ü (U+00FC) -> Bedeutung bleibt erhalten.
	assert "Verschlüsselung" in obj["statements"][1]


def test_repair_laesst_escaped_backslash_unangetastet():
	# Windows-Pfad: \\x ist escaped-Backslash + literal x, KEIN ungültiges Escape -> nicht anfassen.
	ok = r'{"p": "C:\\x64\\bin"}'
	assert _repair_json_escapes(ok) == ok
	assert json.loads(ok)["p"] == r"C:\x64\bin"
