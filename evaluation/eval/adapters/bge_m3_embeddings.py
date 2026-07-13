"""RAGAS-kompatibler Embeddings-Adapter über den BGE-M3 FastAPI-Embeddings-Microservice."""
from __future__ import annotations
import requests
from langchain_core.embeddings import Embeddings


class BgeM3Embeddings(Embeddings):
	# Implementiert die LangChain-Embeddings-Schnittstelle (embed_query/embed_documents),
	# damit RAGAS sie via LangchainEmbeddingsWrapper verwenden kann. Nur der dense-Vektor
	# wird genutzt (Answer Relevance / Answer Correctness brauchen semantische Nähe).
	# Erbt von langchain_core.embeddings.Embeddings -> liefert aembed_query/aembed_documents
	# als Defaults (sync via run_in_executor in einen Thread), die RAGAS im async-Lauf aufruft.
	# Request-/Response-Form gegen docker/embedding-service/app.py abgeglichen:
	# POST /embed {"inputs": [...]} -> {"dense": [[...]], "sparse": [...]}.
	def __init__(self, base_url: str, timeout: int = 60):
		self.base_url = base_url.rstrip("/")
		self.timeout = timeout

	def _post(self, texts: list[str]) -> list[list[float]]:
		resp = requests.post(f"{self.base_url}/embed", json={"inputs": texts}, timeout=self.timeout)
		resp.raise_for_status()
		return resp.json()["dense"]

	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return self._post(texts)

	def embed_query(self, text: str) -> list[float]:
		return self._post([text])[0]
