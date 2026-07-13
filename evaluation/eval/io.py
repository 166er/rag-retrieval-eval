"""JSONL-/qrels-IO und runs-Schema-Validierung für die Evaluation (lokal, kein Testdaten-Import)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

_RUN_PFLICHT = ("query_id", "retrieved", "contexts", "answer")


def read_jsonl(path: Path) -> list[dict]:
	return [json.loads(z) for z in Path(path).read_text(encoding="utf-8").splitlines() if z.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	with Path(path).open("w", encoding="utf-8") as f:
		for r in rows:
			f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
	# TREC 4-spaltig: query_id  iter(0, ignoriert)  chunk_id  relevance. Nur rel>0 behalten.
	out: dict[str, dict[str, int]] = {}
	for z in Path(path).read_text(encoding="utf-8").splitlines():
		if not z.strip():
			continue
		qid, _iter, cid, rel = z.split()
		if int(rel) > 0:
			out.setdefault(qid, {})[cid] = int(rel)
	return out


def read_queries(path: Path) -> dict[str, dict]:
	return {r["query_id"]: r for r in read_jsonl(path)}


def read_runs(path: Path) -> dict[str, dict]:
	return {r["query_id"]: r for r in read_jsonl(path)}


def validate_run_row(row: dict) -> None:
	fehlend = [k for k in _RUN_PFLICHT if k not in row]
	if fehlend:
		raise ValueError(f"runs-Zeile unvollständig, fehlend: {', '.join(fehlend)} (query={row.get('query_id')})")
	if not isinstance(row["retrieved"], list) or not isinstance(row["contexts"], list):
		raise ValueError(f"retrieved/contexts müssen Listen sein (query={row.get('query_id')})")
