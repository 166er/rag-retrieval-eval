import json
import eval.adapters.bge_m3_embeddings as mod
from eval.adapters.bge_m3_embeddings import BgeM3Embeddings


class _Resp:
	def __init__(self, payload):
		self._p = payload
	def raise_for_status(self):
		pass
	def json(self):
		return self._p


def test_embed_query_ruft_dense_endpoint(monkeypatch):
	calls = {}
	def fake_post(url, json, timeout):
		calls["url"] = url
		calls["body"] = json
		return _Resp({"dense": [[0.1, 0.2, 0.3]]})
	monkeypatch.setattr(mod.requests, "post", fake_post)
	emb = BgeM3Embeddings("http://GPU-HOST:8082")
	vec = emb.embed_query("hallo")
	assert vec == [0.1, 0.2, 0.3]
	assert calls["url"].endswith("/embed")


def test_embed_documents_batch(monkeypatch):
	def fake_post(url, json, timeout):
		return _Resp({"dense": [[1.0], [2.0]]})
	monkeypatch.setattr(mod.requests, "post", fake_post)
	emb = BgeM3Embeddings("http://GPU-HOST:8082")
	assert emb.embed_documents(["a", "b"]) == [[1.0], [2.0]]
