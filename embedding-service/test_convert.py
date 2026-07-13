from convert import lexical_to_sparse


def test_lexical_to_sparse_basic():
	# FlagEmbedding liefert {token_id(str): gewicht(float)}
	out = lexical_to_sparse({"8647": 0.2455, "38250": 0.281})
	assert out == {"indices": [8647, 38250], "values": [0.2455, 0.281]}


def test_lexical_to_sparse_empty():
	assert lexical_to_sparse({}) == {"indices": [], "values": []}


def test_lexical_to_sparse_casts_types():
	# numpy-/str-Eingänge müssen zu int/float werden (JSON-serialisierbar)
	import numpy as np
	out = lexical_to_sparse({np.int64(5): np.float32(0.5)})
	assert out == {"indices": [5], "values": [0.5]}
	assert isinstance(out["indices"][0], int)
	assert isinstance(out["values"][0], float)
