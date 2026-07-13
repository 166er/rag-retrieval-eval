"""Report-Stufe führt IR + RAGAS + Negative-Rejection zu metrics.{json,csv} + report.md zusammen."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from eval.config import EvalConfig

SIG = 0.05


def _ir_table(ir: dict) -> str:
	# Tabelle: Metrik | P1 | P2 | p-Wert (Marker * bei Signifikanz).
	sig = ir["overall"].get("significance", {})
	zeilen = ["| Metrik | Advanced (P1) | Agentic (P2) | p |", "|---|---|---|---|"]
	for m in ir["metrics"]:
		p1 = ir["overall"]["p1"].get(m, float("nan"))
		p2 = ir["overall"]["p2"].get(m, float("nan"))
		p = sig.get(m)
		marker = " *" if (isinstance(p, (int, float)) and p < SIG) else ""
		p_txt = f"{p:.3f}{marker}" if isinstance(p, (int, float)) else "n/a"
		zeilen.append(f"| {m} | {p1:.3f} | {p2:.3f} | {p_txt} |")
	return "\n".join(zeilen)


def _flat_rows(ir: dict, ragas: dict | None, neg: dict) -> list[dict]:
	rows: list[dict] = []
	for proto in ("p1", "p2"):
		for m, v in ir["overall"][proto].items():
			rows.append({"prototype": proto, "group": "overall", "scope": "ir", "metric": m, "value": v})
		rows.append({"prototype": proto, "group": "overall", "scope": "neg", "metric": "rejection_rate",
		             "value": neg[proto]["rejection_rate"]})
		if ragas:
			for m, v in ragas[proto]["overall"].items():
				rows.append({"prototype": proto, "group": "overall", "scope": "ragas", "metric": m, "value": v})
	return rows


def build_report(ir: dict, ragas: dict | None, neg: dict, meta: dict | None, iteration_md: str | None = None) -> tuple[str, list[dict]]:
	teile = ["# Evaluations-Report", ""]
	if meta:
		teile += [f"- Datum: {meta.get('timestamp', 'n/a')}", f"- Eval-Modul: {meta.get('module_version', 'n/a')}",
		          f"- Queries (beantwortbar): {ir['n_per_group'].get('answerable_total', 'n/a')}", ""]
	teile += ["## IR-Metriken (gesamt)", "", _ir_table(ir), "", "*\\* p < 0.05 (Fisher-Randomisierung)*", ""]
	teile += ["## Negative Rejection (unbeantwortbar)", "",
	          f"- Advanced (P1): {neg['p1']['rejection_rate']}", f"- Agentic (P2): {neg['p2']['rejection_rate']}", ""]
	if ragas:
		teile += ["## RAGAS (gesamt, beantwortbar)", ""]
		metr = sorted(ragas["p1"]["overall"])
		teile += ["| Metrik | Advanced (P1) | Agentic (P2) |", "|---|---|---|"]
		for m in metr:
			teile.append(f"| {m} | {ragas['p1']['overall'][m]:.3f} | {ragas['p2']['overall'][m]:.3f} |")
		teile.append("")
	else:
		teile += ["## RAGAS", "", "_RAGAS ausstehend (Judge-Phase noch nicht gelaufen)._", ""]
	if iteration_md:
		teile += [iteration_md]
	return "\n".join(teile), _flat_rows(ir, ragas, neg)


def write_outputs(cfg: EvalConfig, ir: dict, ragas: dict | None, neg: dict, meta: dict | None,
                  iteration_md: str | None = None) -> None:
	md, rows = build_report(ir, ragas, neg, meta, iteration_md)
	cfg.results_dir.mkdir(parents=True, exist_ok=True)
	(cfg.results_dir / "report.md").write_text(md, encoding="utf-8")
	(cfg.results_dir / "metrics.json").write_text(
		json.dumps({"ir": ir, "ragas": ragas, "neg": neg, "meta": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
	with (cfg.results_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
		w = csv.DictWriter(f, fieldnames=["prototype", "group", "scope", "metric", "value"])
		w.writeheader()
		w.writerows(rows)


def _load(path: Path):
	return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
	cfg = EvalConfig.load()
	ir = _load(cfg.results_dir / "ir.json")
	if ir is None:
		raise SystemExit("results/ir.json fehlt, zuerst eval.ir_metrics laufen lassen.")
	neg = _load(cfg.results_dir / "neg_rejection.json") or {"p1": {"rejection_rate": float("nan")}, "p2": {"rejection_rate": float("nan")}}
	ragas = _load(cfg.results_dir / "ragas.json")
	meta = _load(cfg.runs_dir / "run_meta.json")
	from eval.iteration_analysis import section_from_cfg
	iteration_md = section_from_cfg(cfg)   # P2-Iterations-Block, sofern runs_p2 + ragas_raw vorhanden
	write_outputs(cfg, ir, ragas, neg, meta, iteration_md)
	print(f"Report geschrieben: {cfg.results_dir / 'report.md'}")


if __name__ == "__main__":
	main()
