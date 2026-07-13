import os
import pytest
from pathlib import Path
from eval.config import EvalConfig


def test_load_liest_pflichtfelder(monkeypatch, tmp_path):
	# Minimale gültige Umgebung setzen
	env = {
		"P1_WEBHOOK_URL": "http://vm:5678/webhook/p1",
		"P2_WEBHOOK_URL": "http://vm:5678/webhook/p2",
		"JUDGE_BASE_URL": "http://GPU-HOST:8001/v1",
		"JUDGE_MODEL": "Qwen/Qwen3-32B",
		"JUDGE_API_KEY": "dummy",
		"EMBED_BASE_URL": "http://GPU-HOST:8082",
		"QUERIES_PATH": "../Testdaten/queries.jsonl",
		"QRELS_PATH": "../Testdaten/qrels.txt",
		"REFERENCE_ANSWERS_PATH": "../Testdaten/reference_answers.jsonl",
		"RUNS_DIR": str(tmp_path / "runs"),
		"RESULTS_DIR": str(tmp_path / "results"),
		"ABSTAIN_PHRASE": "Dazu liegen mir keine Informationen vor.",
		"EVAL_MODULE_VERSION": "1.0.0",
	}
	for k, v in env.items():
		monkeypatch.setenv(k, v)
	cfg = EvalConfig.load(use_dotenv=False)
	assert cfg.p1_webhook_url.endswith("/p1")
	assert cfg.judge_model == "Qwen/Qwen3-32B"
	assert cfg.k_values == (5, 10)
	assert cfg.abstain_phrase.startswith("Dazu liegen")


def test_load_fehlende_pflichtvariable_meldet_klar(monkeypatch):
	monkeypatch.delenv("P1_WEBHOOK_URL", raising=False)
	with pytest.raises(ValueError, match="P1_WEBHOOK_URL"):
		EvalConfig.load(use_dotenv=False)
