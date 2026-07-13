"""
Kapselt die Referenz-Bibliothek FlagEmbedding als HTTP-Service. Kein
Inferenz-Server (TEI, Infinity, vLLM) liefert BGE-M3s native Sparse-Gewichte,
sie stammen aus FlagEmbeddings sparse_linear-Kopf. Dieser Dienst macht beide
Vektor-Typen über einen Aufruf verfügbar.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel
from convert import lexical_to_sparse

MODEL_ID = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
	# Modell einmal beim Start in den GPU-Speicher laden (fp16 spart VRAM)
	_state["model"] = BGEM3FlagModel(MODEL_ID, use_fp16=True)
	yield
	_state.clear()


app = FastAPI(lifespan=lifespan)


class EmbedRequest(BaseModel):
	inputs: list[str]


@app.get("/health")
def health():
	return {"status": "ok", "model": MODEL_ID, "ready": "model" in _state}


@app.post("/embed")
def embed(req: EmbedRequest):
	"""Dense + Sparse je Input, Reihenfolge = Eingabe-Reihenfolge."""
	out = _state["model"].encode(
		req.inputs, return_dense=True, return_sparse=True
	)
	dense = [vec.tolist() for vec in out["dense_vecs"]]
	sparse = [lexical_to_sparse(w) for w in out["lexical_weights"]]
	return {"dense": dense, "sparse": sparse}
