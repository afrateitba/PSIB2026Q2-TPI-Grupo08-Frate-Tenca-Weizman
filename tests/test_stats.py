import numpy as np
import pandas as pd
import pytest

from src.split import person_folds
from src.stats import fisher_direction, percentile_interval, person_bootstrap_means


def _table():
    rows = [("a1", "h", "P1"), ("a2", "h", "P1"), ("b", "h", "P2"), ("c", "h", "P3"),
            ("d1", "dr", "P4"), ("d2", "dr", "P4"), ("e", "dr", "P5"), ("f", "dr", "P6"),
            ("g", "dr", "P7"), ("h1", "g", "P8"), ("h2", "g", "P8"), ("i", "g", "P9")]
    return pd.DataFrame(rows, columns=["image", "group", "person"])


def test_person_folds_keeps_persons_and_balances():
    t = _table()
    folds = person_folds(t, k=3, seed=1)
    assert set(folds.index) == set(t.person)
    assert sorted(folds.value_counts()) == [3, 3, 3]          # 9 personas en 3 grupos
    for g, d in t.drop_duplicates("person").groupby("group"):  # cada grupo clínico repartido
        assert folds[d.person].nunique() == min(3, len(d))
    assert folds.equals(person_folds(t, k=3, seed=1))


def test_person_bootstrap_resamples_whole_persons():
    t = _table().set_index("image")
    # Las dos imágenes de P1 valen 10; el resto 0. Si se sortea P1, entran las dos.
    vals = pd.Series(0.0, index=t.index)
    vals[["a1", "a2"]] = 10.0
    means = person_bootstrap_means(vals, t.person, reps=4000, seed=0)[:, 0]
    assert means.mean() == pytest.approx(vals.mean(), abs=0.15)
    lo, hi = percentile_interval(means)
    assert lo < vals.mean() < hi


def test_person_bootstrap_columns_share_draws():
    t = _table().set_index("image")
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(size=len(t)), index=t.index)
    frame = pd.DataFrame({"a": a, "b": a + 0.5})
    m = person_bootstrap_means(frame, t.person, reps=500)
    assert np.allclose(m[:, 1] - m[:, 0], 0.5)                # diferencia apareada exacta


def test_fisher_direction_points_to_vessels():
    rng = np.random.default_rng(0)
    bg = rng.normal([0, 0], [1, 1], (5000, 2))
    ves = rng.normal([2, 0], [1, 1], (500, 2))               # solo el primer mapa separa
    x = np.r_[bg, ves]
    y = np.r_[np.zeros(5000, bool), np.ones(500, bool)]
    w = fisher_direction(x, y)
    assert w[0] > 0.95 and abs(w[1]) < 0.15


def test_fit_logistic_recovers_coefficients():
    from src.stats import fit_logistic, predict_logistic
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20000, 2))
    true = np.array([-1.0, 2.0, 0.0])          # solo la primera columna importa
    p = 1 / (1 + np.exp(-(true[0] + x @ true[1:])))
    y = rng.random(len(p)) < p
    w = fit_logistic(x, y, ridge=1e-6)
    assert np.allclose(w, true, atol=0.1)
    assert np.all((predict_logistic(x, w) >= 0) & (predict_logistic(x, w) <= 1))


def test_fit_logistic_separable_does_not_explode():
    from src.stats import fit_logistic
    x = np.r_[np.linspace(-3, -1, 50), np.linspace(1, 3, 50)][:, None]
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    w = fit_logistic(x, y, ridge=1e-2)
    assert np.all(np.isfinite(w)) and w[1] > 0


def test_icc_absolute_known_values():
    from src.stats import icc_absolute
    x = np.arange(1.0, 6.0)
    assert icc_absolute(x, x) == pytest.approx(1.0)
    assert icc_absolute(x, x + 1) == pytest.approx(5 / 6)       # el corrimiento fijo baja el acuerdo
    rng = np.random.default_rng(0)
    assert abs(icc_absolute(rng.normal(size=500), rng.normal(size=500))) < 0.1


def test_bland_altman_bias_and_limits():
    from src.stats import bland_altman
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = bland_altman(x, x + np.array([0.5, 1.5, 0.5, 1.5]))
    assert out["sesgo"] == pytest.approx(1.0)
    sd = np.std([0.5, 1.5, 0.5, 1.5], ddof=1)
    assert out["límite superior"] == pytest.approx(1.0 + 1.96 * sd)


def test_person_bootstrap_stat_matches_means():
    from src.stats import person_bootstrap_means, person_bootstrap_stat
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 5.0, 8.0]}, index=list("vwxyz"))
    persons = pd.Series(["p", "p", "q", "r", "r"], index=frame.index)
    by_stat = person_bootstrap_stat(frame, persons, lambda f: f["a"].mean(), reps=200)
    by_means = person_bootstrap_means(frame["a"], persons, reps=200)[:, 0]
    assert np.allclose(by_stat, by_means)


def _datos_logisticos(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 8))
    x[:, 5] = x[:, 0] + 0.05 * rng.normal(0, 1, n)          # casi copia del mapa 0
    z = -0.5 + 2.0 * x[:, 0] - 1.0 * x[:, 3]
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    return x, y


def test_l1_sin_penalidad_es_la_regresion_comun():
    from src.stats import fit_logistic, fit_logistic_l1
    x, y = _datos_logisticos()
    x = x[:, [0, 1, 2, 3]]
    np.testing.assert_allclose(fit_logistic_l1(x, y, 0.0), fit_logistic(x, y, ridge=0.0), atol=1e-5)


def test_l1_deja_en_cero_lo_que_no_aporta_y_una_de_las_copias():
    from src.stats import fit_logistic_l1, l1_lambdas
    x, y = _datos_logisticos()
    lams = l1_lambdas(x, y)
    w_max = fit_logistic_l1(x, y, lams[0])
    assert np.all(w_max[1:] == 0)                              # con el λ más grande, solo la ordenada
    w = fit_logistic_l1(x, y, 0.01)
    activos = set(np.flatnonzero(w[1:]))
    assert 3 in activos and (0 in activos or 5 in activos)
    assert not ({1, 2, 4, 6, 7} & activos)                       # los que no aportan quedan en cero
    assert not ({0, 5} <= activos and min(abs(w[1]), abs(w[6])) > 0.2)   # no reparte entre las copias
