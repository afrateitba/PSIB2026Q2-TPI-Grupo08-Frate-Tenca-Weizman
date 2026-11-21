"""Pruebas de la corrección del fondo."""

import numpy as np
import pytest

from src.preprocess import MODOS_FONDO, corregir_fondo, normalizar_fondo


def test_fondo_constante_y_exterior_excluido():
    """Un campo uniforme queda en cero, aunque el exterior sea brillante."""
    campo = np.zeros((20, 20), dtype=bool)
    campo[4:16, 4:16] = True

    mapa = np.full((20, 20), 100.0)
    mapa[campo] = 3.0

    resultado = corregir_fondo(mapa, campo, sigma=2)

    np.testing.assert_allclose(
        resultado[campo], 0, atol=1e-5
    )


def _retina_multiplicativa(n=200, absorcion=0.25):
    """Verde = iluminación × reflectancia: iluminación de 50 a 200 de izquierda a derecha y dos
    vasos verticales de 3 píxeles que absorben la misma fracción de la luz. Devuelve (−verde, campo)."""
    x = np.arange(n)
    luz = np.tile(50 + 150 * x / (n - 1), (n, 1))
    refl = np.ones((n, n))
    refl[:, 38:41] = 1 - absorcion        # vaso en zona oscura
    refl[:, 158:161] = 1 - absorcion      # vaso en zona clara
    return -(luz * refl), np.ones((n, n), dtype=bool)


def test_resta_es_corregir_fondo():
    mapa, campo = _retina_multiplicativa()
    np.testing.assert_allclose(normalizar_fondo(mapa, campo, 15, "resta"), corregir_fondo(mapa, campo, 15),
                               rtol=1e-5, atol=1e-4)


def test_division_iguala_el_contraste_con_iluminacion_multiplicativa():
    """Con la resta, el vaso de la zona clara queda con más contraste (la relación de iluminación, ~2,1); con la división, igual."""
    mapa, campo = _retina_multiplicativa()
    resta = normalizar_fondo(mapa, campo, 15, "resta")
    div = normalizar_fondo(mapa, campo, 15, "división")
    c_resta = resta[:, 158:161].mean() / resta[:, 38:41].mean()
    c_div = div[:, 158:161].mean() / div[:, 38:41].mean()
    assert c_resta > 1.8
    assert abs(c_div - 1) < 0.1
    assert abs(div[:, 158:161].mean() - 0.25) < 0.05      # el contraste relativo es la absorción


def test_luminosidad_y_contraste_iguala_el_ruido_del_fondo():
    """Fondo con el doble de ruido a la derecha: después de normalizar, el desvío es parecido en las dos mitades."""
    rng = np.random.default_rng(0)
    n = 200
    ruido = rng.normal(0, 1, (n, n)) * np.where(np.arange(n) < n // 2, 2.0, 4.0)[None, :]
    mapa = -(100 + ruido)
    campo = np.ones((n, n), dtype=bool)
    out = normalizar_fondo(mapa, campo, 15, "luminosidad y contraste")
    izq, der = out[20:-20, 20:80].std(), out[20:-20, 120:180].std()
    assert abs(izq / der - 1) < 0.15


@pytest.mark.parametrize("modo", MODOS_FONDO)
def test_campo_uniforme_da_cero(modo):
    campo = np.zeros((40, 40), dtype=bool)
    campo[5:35, 5:35] = True
    mapa = np.where(campo, -80.0, 0.0)
    out = normalizar_fondo(mapa, campo, 4, modo)
    np.testing.assert_allclose(out[campo], 0, atol=1e-4)


def test_modo_desconocido():
    with pytest.raises(ValueError):
        normalizar_fondo(np.zeros((5, 5)), np.ones((5, 5), bool), 1, "otro")


def test_relleno_de_exudados_rellena_la_mancha_y_respeta_vaso_y_disco():
    """Una mancha clara se rellena con el nivel del fondo; el vaso (oscuro) y el disco (la zona más clara) quedan igual."""
    from src.preprocess import rellenar_exudados
    rng = np.random.default_rng(0)
    n = 200
    yy, xx = np.mgrid[:n, :n]
    g = 100 + rng.normal(0, 1, (n, n))
    g[(yy - 40) ** 2 + (xx - 40) ** 2 <= 12 ** 2] = 200       # disco óptico
    g[(yy - 140) ** 2 + (xx - 140) ** 2 <= 4 ** 2] = 150      # exudado
    g[:, 150:153] = 60                                          # vaso al lado
    campo = np.ones((n, n), bool)
    out, mask = rellenar_exudados(-g, campo, diametro=200)
    assert mask[140, 140] and not mask[40, 40]                  # rellena el exudado, no el disco
    assert not mask[:, 150:153].any()                          # nunca tapa el vaso
    assert abs(-out[140, 140] - 100) < 10                       # queda al nivel del fondo
    np.testing.assert_array_equal(out[:, 150:153], -g[:, 150:153])
