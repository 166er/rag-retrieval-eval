"""Konvertiert FlagEmbedding-lexical_weights in Qdrant-Sparse-Form.

FlagEmbedding gibt Sparse als Dict {token_id: gewicht} zurück. 
Qdrant erwartet getrennte Arrays {indices, values}. 
Token-IDs dienen direkt als Indizes.
"""


def lexical_to_sparse(weights: dict) -> dict:
	"""{token_id: gewicht} → {"indices": [int...], "values": [float...]}."""
	indices = [int(tok) for tok in weights.keys()]
	values = [float(w) for w in weights.values()]
	return {"indices": indices, "values": values}
