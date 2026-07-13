# Evaluations-Pipeline



Berechnet auf einer Testkollektion (Queries, qrels, Referenzantworten) und frischen Prototypen-Runs IR-Metriken per ranx und RAGAS-Scores.
Erstellt einen kompletten Markdown-Report.
Fünf unabhängig lauffähige Stufen mit JSONL/JSON-Artefakten dazwischen.



# Bestandteile



|Stufe|Modul|Eingaben|Ausgaben|
|-|-|-|-|
|Runner|eval.runner|queries.jsonl, Webhooks|runs/runs\_p{1,2}.jsonl, runs/run\_meta.json|
|IR-Metriken|eval.ir\_metrics|qrels.txt, runs, queries|results/ir.json|
|RAGAS|eval.ragas\_eval|runs, queries, reference\_answers.jsonl, Judge + Embeddings|results/ragas.json, results/ragas\_raw\_p{1,2}.json|
|RAGAS-Signifikanz|eval.ragas\_significance|results/ragas\_raw\_p{1,2}.json|results/ragas\_significance.json|
|Negative Rejection|eval.negative\_rejection|runs, queries|results/neg\_rejection.json|
|Report|eval.report|ir.json, ragas.json, neg\_rejection.json, run\_meta.json|results/metrics.{csv,json}, results/report.md|
|Smoke-Test|eval.smoke|Mini-Fixture (tests/fixtures/)|results/smoke/report.md|

 

# Befehls-Reihenfolge



## 1. Generierungs-Phase



python -m eval.runner --prototype both
python -m eval.ir\_metrics



## 2. Modell-Swap auf dem GPU-Host



Generator stoppen, dann docker compose --profile judge up -d vllm-qwen3-judge



## 3. Bewertungs-Phase



python -m eval.ragas\_eval
python -m eval.ragas\_significance
python -m eval.negative\_rejection
python -m eval.report



## Offline-Funktionscheck, ohne Dienste



python -m eval.smoke



## Alle Tests

pytest


eval.report läuft auch ohne ragas.json (IR-only-Fallback).
Der Report weist RAGAS dann als ausstehend aus. So lässt sich nach der Generierungs-Phase bereits ein vollständiger IR-Report erzeugen.

