## Repository zur Bachelorarbeit



Repository zur Bachelorarbeit **„Evaluierung der Retrieval-Qualität von RAG-Workflows für lokal betriebenes, KI-gestütztes Wissensmanagement"**, 2026.
Es enthält die beiden untersuchten RAG-Prototypen als n8n-Workflow-Exporte, den entwickelten BGE-M3-Embedding-Microservice, die Evaluationspipeline sowie die Docker-Compose-Definitionen der eingesetzten Dienste.



## Inhalt



**workflows/:**
Die zwei Prototypen als n8n-Exporte, Advanced RAG und Agentic RAG, inkl. Systemprompts.


**embedding-service:**
FastAPI-Microservice, der FlagEmbedding kapselt und für BGE-M3 Dense- und Sparse-Vektoren aus einem Forward-Pass liefert.


**evaluation/:**
Evaluationspipeline mit Runner (Webhook-Aufrufe), IR-Metriken (ranx), RAGAS-Bewertung inkl. Signifikanztest, Negative Rejection, Report-Generierung, Tests.


**evaluation/sample\_data/:**
Drei Beispiel-Einträge im Format der echten Testkollektion, siehe Formatdoku unten.


**docker-compose.yml:**
vLLM-Generator GPT-OSS-120B, BGE-M3-Embeddings, TEI-Reranker bge-reanker-v2-m3, Judge-Modell Qwen3-32B.


**docker-compose\_n8n.yml:**
n8n 2.23.4 + PostgreSQL.


**docker-compose\_qdrant.yml:**
Qdrant v1.13.4.


**.env.example:**
Platzhalter für Hosts, Endpunkte und Keys.



## Was fehlt und warum



* **Der Dokumentenkorpus** ist Unternehmenswissen und in keiner Form enthalten.
* **Die Testkollektion** basiert auf diesem Korpus und ist ebenfalls nicht enthalten, evaluation/sample\_data/ demonstriert das Format.
* **Die Datenaufbereitungspipeline** ist explizit aus der Untersuchung ausgegrenzt und ohne den Korpus nicht demonstrierbar.





## Systemkontext



Zwei Hosts, ein GPU-Host (NVIDIA DGX Spark) für Generator, Embeddings, Reranker und Judge sowie eine VM für n8n und Qdrant.
Die n8n-Workflows erreichen die Dienste über die Umgebungsvariablen, die Evaluationspipeline über evaluation/.env.



## Format der Eval-Eingabedateien



**queries.jsonl, ein Query pro Zeile:**
{"query\_id": "q001", "text": "…", "type": "factual|multihop|aggregate|unanswerable", "source\_target": \["pdf"], "intended\_meaning": "", "notes": "…"}


**qrels.txt, Relevanzurteile im TREC-qrels-Format**:
query\_id 0 chunk\_id relevanz, Chunk-IDs aus Qdrant, nicht beantwortbare Queries haben keine qrels-Zeilen.
q001 0 4c8f2a1e-0001-5a2b-9c3d-1f6e8b2d4a01 1


**reference\_answers.jsonl, Referenzantwort pro Query:**
bei nicht beantwortbaren Queries die Enthaltungs-Formulierung.
{"query\_id": "q001", "answer": "…"}


**Runner-Ausgabe runs/runs\_p{1,2}.jsonl:**
Ein Run pro Query mit voller Rangliste (retrieved, für IR-Metriken),
den tatsächlich in die Synthese gegebenen Chunks (contexts, für RAGAS) und der generierten Antwort.
{"query\_id": "q001", "retrieved": \[{"chunk\_id": "…", "rank": 1, "rerank\_score": 0.99, "text": "…"}], "contexts": \[{"chunk\_id": "…", "rerank\_score": 0.99, "text": "…"}], "answer": "…", "raw": {}}



## Rechtliches



Dieses Repository dient ausschließlich der Begutachtung und Nachvollziehbarkeit der Bachelorarbeit.
Es steht bewusst unter keiner Open-Source-Lizenz.
Alle Rechte vorbehalten.

