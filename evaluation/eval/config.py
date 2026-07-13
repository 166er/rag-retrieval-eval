"""Konfiguration der Evaluations-Pipeline, lädt Evaluation/.env, validiert Pflichtfelder."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Pflicht-Variablen, ohne die keine Stufe sinnvoll laufen kann.
_REQUIRED = [
	"P1_WEBHOOK_URL", "P2_WEBHOOK_URL", "JUDGE_BASE_URL", "JUDGE_MODEL",
	"EMBED_BASE_URL", "QUERIES_PATH", "QRELS_PATH", "REFERENCE_ANSWERS_PATH",
	"RUNS_DIR", "RESULTS_DIR", "ABSTAIN_PHRASE", "EVAL_MODULE_VERSION",
]


@dataclass(frozen=True)
class EvalConfig:
	p1_webhook_url: str
	p2_webhook_url: str
	judge_base_url: str
	judge_model: str
	judge_api_key: str
	embed_base_url: str
	queries_path: Path
	qrels_path: Path
	reference_answers_path: Path
	runs_dir: Path
	results_dir: Path
	abstain_phrase: str
	module_version: str
	k_values: tuple[int, int] = (5, 10)

	@classmethod
	def load(cls, use_dotenv: bool = True) -> "EvalConfig":
		if use_dotenv:
			load_dotenv(Path(__file__).resolve().parents[1] / ".env")
		fehlend = [k for k in _REQUIRED if not os.getenv(k)]
		if fehlend:
			raise ValueError(f"Fehlende Pflicht-Umgebungsvariablen: {', '.join(fehlend)}")
		# Pfade relativ zum Evaluation-Ordner aufloesen (Forward-Slashes in .env).
		base = Path(__file__).resolve().parents[1]
		def p(key: str) -> Path:
			raw = Path(os.environ[key])
			return raw if raw.is_absolute() else (base / raw)
		return cls(
			p1_webhook_url=os.environ["P1_WEBHOOK_URL"],
			p2_webhook_url=os.environ["P2_WEBHOOK_URL"],
			judge_base_url=os.environ["JUDGE_BASE_URL"],
			judge_model=os.environ["JUDGE_MODEL"],
			judge_api_key=os.getenv("JUDGE_API_KEY", "dummy"),
			embed_base_url=os.environ["EMBED_BASE_URL"],
			queries_path=p("QUERIES_PATH"),
			qrels_path=p("QRELS_PATH"),
			reference_answers_path=p("REFERENCE_ANSWERS_PATH"),
			runs_dir=p("RUNS_DIR"),
			results_dir=p("RESULTS_DIR"),
			abstain_phrase=os.environ["ABSTAIN_PHRASE"],
			module_version=os.environ["EVAL_MODULE_VERSION"],
		)
