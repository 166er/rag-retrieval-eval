import json
from pathlib import Path
from eval.runner import parse_response
from eval.io import validate_run_row

FIX = Path(__file__).parent / "fixtures"


def test_parse_response_normalisiert_schema():
	body = json.loads((FIX / "webhook_response_p1.json").read_text(encoding="utf-8"))
	parsed = parse_response(body)
	assert parsed["retrieved"][0]["chunk_id"] == "A"
	assert parsed["retrieved"][0]["rerank_score"] == 0.99
	assert parsed["contexts"][0]["chunk_id"] == "A"
	assert parsed["answer"] == "Antwort A."
	# Schema-Vertrag (mit query_id) muss gültig sein
	parsed["query_id"] = "q1"
	validate_run_row(parsed)


def test_parse_response_leitet_contexts_ab_wenn_fehlend():
	# Ohne explizites contexts-Feld: aus retrieved mit Score>=0.2, max 8.
	body = {"retrieved": [{"chunkId":"A","rerankScore":0.99,"text":"a"},
	                      {"chunkId":"B","rerankScore":0.10,"text":"b"}], "answer": "x"}
	parsed = parse_response(body)
	assert [c["chunk_id"] for c in parsed["contexts"]] == ["A"]   # B < 0.2 gefiltert
