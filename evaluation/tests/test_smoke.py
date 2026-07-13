from pathlib import Path
from eval.smoke import run_offline_smoke

FIX = Path(__file__).parent / "fixtures"


def test_offline_smoke_erzeugt_report(tmp_path):
	report = run_offline_smoke(FIX, tmp_path)
	assert report.exists()
	text = report.read_text(encoding="utf-8")
	assert "IR-Metriken" in text
	assert "RAGAS ausstehend" in text   # Smoke läuft ohne Judge
