from pathlib import Path
import pytest
from eval.io import read_jsonl, read_qrels, read_runs, validate_run_row

FIX = Path(__file__).parent / "fixtures"


def test_read_qrels_ignoriert_spalte2_und_filtert_nullen():
	q = read_qrels(FIX / "mini_qrels.txt")
	assert q["q1"] == {"A": 1, "B": 1}      # X (rel 0) ist nicht enthalten
	assert q["q2"] == {"C": 1}
	assert "q3" not in q                       # nicht beantwortbar = kein Eintrag


def test_read_runs_indiziert_nach_query_id():
	r = read_runs(FIX / "mini_runs_p1.jsonl")
	assert set(r) == {"q1", "q2", "q3"}
	assert r["q1"]["retrieved"][0]["chunk_id"] == "A"


def test_validate_run_row_wirft_bei_fehlendem_feld():
	with pytest.raises(ValueError, match="answer"):
		validate_run_row({"query_id": "q1", "retrieved": [], "contexts": []})
