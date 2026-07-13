import math
from eval.ragas_significance import describe, paired_permutation_test, compute_significance, read_raw


def test_describe_liefert_mittelwert_sd_und_ci95():
	stats = describe([0.4, 0.6, 0.8, 1.0])
	assert stats["mean"] == 0.7
	assert round(stats["sd"], 6) == round(math.sqrt(0.2 / 3), 6)		# Stichproben-SD (n-1)
	lo, hi = stats["ci95"]
	assert lo < 0.7 < hi
	assert round(hi - 0.7, 6) == round(1.96 * stats["sd"] / 2, 6)		# Halbweite = 1,96*SD/sqrt(n)


def test_describe_bei_zu_kleiner_stichprobe_nan():
	stats = describe([0.5])
	assert stats["mean"] == 0.5
	assert math.isnan(stats["sd"])


def test_permutationstest_identische_laeufe_p_gleich_eins():
	a = [0.1, 0.5, 0.9, 0.3]
	assert paired_permutation_test(a, list(a), n_permutations=500) == 1.0


def test_permutationstest_klare_differenz_signifikant():
	a = [0.9] * 20
	b = [0.1] * 20
	assert paired_permutation_test(a, b, n_permutations=2000) < 0.05


def test_permutationstest_deterministisch_bei_gleichem_seed():
	a = [0.8, 0.6, 0.7, 0.9, 0.5, 0.65, 0.75, 0.85]
	b = [0.7, 0.65, 0.6, 0.95, 0.55, 0.6, 0.8, 0.7]
	p1 = paired_permutation_test(a, b, n_permutations=2000, seed=42)
	p2 = paired_permutation_test(a, b, n_permutations=2000, seed=42)
	assert p1 == p2


def test_compute_significance_paart_nur_gueltige_scores():
	raw_p1 = {
		"q1": {"faithfulness": 0.8},
		"q2": {"faithfulness": float("nan")},	# NaN bei P1 -> Paar entfällt
		"q3": {"faithfulness": 0.6},
		"q4": {"faithfulness": 0.7},			# fehlt bei P2 -> Paar entfällt
	}
	raw_p2 = {
		"q1": {"faithfulness": 0.9},
		"q2": {"faithfulness": 0.5},
		"q3": {"faithfulness": 0.4},
	}
	out = compute_significance(raw_p1, raw_p2, metric_keys=("faithfulness",), n_permutations=500)
	m = out["metrics"]["faithfulness"]
	assert out["n_queries"] == 3					# Schnittmenge q1-q3
	assert m["n"] == 2								# q2 (NaN) raus
	assert m["p1"]["mean"] == 0.7					# (0.8+0.6)/2
	assert round(m["delta_mean_p2_minus_p1"], 6) == round(0.65 - 0.7, 6)
	assert 0.0 <= m["p_value"] <= 1.0


def test_read_raw_schliesst_unanswerable_aus(tmp_path):
	pfad = tmp_path / "ragas_raw.json"
	pfad.write_text(
		'[{"query_id": "q1", "type": "factual", "faithfulness": 0.8},'
		' {"query_id": "q9", "type": "unanswerable", "faithfulness": 1.0}]',
		encoding="utf-8",
	)
	raw = read_raw(pfad)
	assert set(raw) == {"q1"}
