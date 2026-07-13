"""Runner schickt jede Query an beide Webhooks, protokolliert die Ranglisten + run_meta."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests
from tqdm import tqdm
from eval.config import EvalConfig
from eval.io import read_queries, read_jsonl, write_jsonl, validate_run_row

CONTEXT_SCORE_MIN = 0.2
CONTEXT_TOP_K = 8


def _normalize_chunk(c: dict, rank: int) -> dict:
	return {
		"chunk_id": c["chunkId"],
		"rank": rank,
		"rerank_score": c.get("rerankScore"),
		"text": c.get("text", ""),
		"source": c.get("source"),
		"doc_id": c.get("docId"),
		"title": c.get("title"),
		"heading_path": c.get("headingPath"),
		"page_number": c.get("pageNumber"),
	}


def parse_response(body: dict) -> dict:
	retrieved = [_normalize_chunk(c, i + 1) for i, c in enumerate(body.get("retrieved", []))]
	if body.get("contexts"):
		contexts = [_normalize_chunk(c, i + 1) for i, c in enumerate(body["contexts"])]
	else:
		gefiltert = [c for c in retrieved if (c["rerank_score"] or 0) >= CONTEXT_SCORE_MIN]
		contexts = gefiltert[:CONTEXT_TOP_K]
	return {"retrieved": retrieved, "contexts": contexts, "answer": body.get("answer", ""), "raw": body}


def _post(url: str, query: str, timeout: int) -> dict:
	resp = requests.post(url, json={"query": query}, timeout=timeout)
	resp.raise_for_status()
	return resp.json()


def _preflight(url: str) -> None:
	try:
		requests.options(url, timeout=10)
	except Exception as e:
		raise SystemExit(f"Preflight: Webhook {url} nicht erreichbar: {e}")


def build_run_meta(cfg: EvalConfig) -> dict:
	# git-unabhängiger Reproduzierbarkeits-Stempel.
	def file_hash(p: Path) -> str:
		return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "n/a"
	wf = Path(__file__).resolve().parents[2] / "Workflows"
	return {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"module_version": cfg.module_version,
		"workflow_p1_sha": file_hash(wf / "Advanced RAG.json"),
		"workflow_p2_sha": file_hash(wf / "Agentic RAG.json"),
		"endpoints": {"p1": cfg.p1_webhook_url, "p2": cfg.p2_webhook_url},
		"k_values": list(cfg.k_values),
	}


def run_all(cfg: EvalConfig, prototype: str, only_ids: set[str] | None, timeout: int = 180) -> None:
	queries = list(read_queries(cfg.queries_path).values())
	if only_ids:
		queries = [q for q in queries if q["query_id"] in only_ids]
	targets = {"p1": (cfg.p1_webhook_url, "runs_p1.jsonl"), "p2": (cfg.p2_webhook_url, "runs_p2.jsonl")}
	protos = ["p1", "p2"] if prototype == "both" else [prototype]
	cfg.runs_dir.mkdir(parents=True, exist_ok=True)
	for proto in protos:
		url, fname = targets[proto]
		_preflight(url)
		out_path = cfg.runs_dir / fname
		bestehend = {r["query_id"]: r for r in read_jsonl(out_path)} if (only_ids and out_path.exists()) else {}
		for q in tqdm(queries, desc=f"Runner {proto}"):
			try:
				parsed = parse_response(_post(url, q["text"], timeout))
			except Exception as e:
				parsed = {"retrieved": [], "contexts": [], "answer": "", "raw": {}, "error": str(e)}
				tqdm.write(f"FEHLER bei {q['query_id']} ({proto}): {e}")
			parsed["query_id"] = q["query_id"]
			if "error" not in parsed:
				validate_run_row(parsed)
			bestehend[q["query_id"]] = parsed
		# Atomar schreiben (temp -> replace).
		tmp = out_path.with_suffix(".jsonl.tmp")
		write_jsonl(tmp, [bestehend[k] for k in sorted(bestehend)])
		tmp.replace(out_path)
	(cfg.runs_dir / "run_meta.json").write_text(json.dumps(build_run_meta(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--prototype", choices=["p1", "p2", "both"], default="both")
	ap.add_argument("--only", default="")
	ap.add_argument("--timeout", type=int, default=180)
	args = ap.parse_args()
	only = {s.strip() for s in args.only.split(",") if s.strip()} or None
	run_all(EvalConfig.load(), args.prototype, only, args.timeout)


if __name__ == "__main__":
	main()
