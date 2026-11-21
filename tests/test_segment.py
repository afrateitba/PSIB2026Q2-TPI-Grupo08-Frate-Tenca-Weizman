"""Pruebas de la segmentación por Otsu."""

import numpy as np
from src.segment import segmentar_otsu


def test_otsu_separa_vasos_y_excluye_exterior():
    """Detecta una línea clara sin incluir píxeles fuera del campo."""
    mapa = np.zeros((10, 10))
    mapa[:, 5] = 1
    mapa[:, 0] = 100  # Exterior brillante: no debe afectar el umbral.

    campo = np.ones((10, 10), dtype=bool)
    campo[:, 0] = False

    esperado = np.zeros((10, 10), dtype=bool)
    esperado[:, 5] = True

    mascara, _ = segmentar_otsu(mapa, campo)

    np.testing.assert_array_equal(mascara, esperado)


def test_fraccion_marca_la_proporcion_pedida():
    from src.segment import segmentar_fraccion
    mapa = np.arange(100, dtype=float).reshape(10, 10)
    campo = np.ones((10, 10), dtype=bool)
    mascara, _ = segmentar_fraccion(mapa, campo, 0.1)
    assert mascara.sum() == 10 and mascara[9].all()


def test_histeresis_recupera_lo_debil_conectado():
    from src.segment import segmentar_histeresis
    rng = np.random.default_rng(0)
    mapa = rng.normal(0, 0.01, (40, 60))    # fondo con un poco de ruido
    mapa[10, 5:30] += 1.0         # vaso fuerte
    mapa[10, 30:55] += 0.5        # continuación débil, conectada
    mapa[30, 5:55] += 0.5         # vaso débil suelto
    campo = np.ones((40, 60), dtype=bool)
    # alto: parte del vaso fuerte (12 de 2400 píxeles); bajo: todo lo marcado (144)
    mascara, _ = segmentar_histeresis(mapa, campo, frac_alto=0.005, frac_bajo=0.06)
    assert mascara[10, 5:55].all()            # fuerte y su continuación
    assert not mascara[30].any()              # el débil suelto no


def test_limpiar_por_forma_quita_manchas_redondas():
    from src.segment import limpiar_por_forma
    m = np.zeros((80, 120), dtype=bool)
    m[10:12, 5:100] = True                    # vaso fino y largo
    yy, xx = np.mgrid[:80, :120]
    m |= (yy - 50) ** 2 + (xx - 30) ** 2 <= 36   # mancha redonda chica
    m |= (yy - 50) ** 2 + (xx - 90) ** 2 <= 225  # mancha redonda grande
    limpia = limpiar_por_forma(m, alargamiento_min=4, area_max=300)
    assert limpia[10:12, 5:100].all()         # el vaso queda
    assert not limpia[50, 30]                 # la mancha chica se va
    assert limpia[50, 90]                     # la grande supera area_max y queda


def test_histeresis_valor_recupera_lo_conectado_y_no_lo_suelto():
    from src.segment import segmentar_histeresis_valor
    mapa = np.zeros((30, 30), np.float32)
    mapa[10, 2:20] = 0.95             # vaso seguro
    mapa[10, 20:28] = 0.6             # continuación débil, conectada
    mapa[11:13, 28] = 0.6             # sigue en diagonal
    mapa[25, 5:9] = 0.6               # mancha débil suelta
    campo = np.ones_like(mapa, bool)
    m = segmentar_histeresis_valor(mapa, campo, alto=0.9, bajo=0.5)
    assert m[10, 2:28].all() and m[11:13, 28].all()
    assert not m[25, 5:9].any()
    np.testing.assert_array_equal(segmentar_histeresis_valor(mapa, campo, 0.5, 0.5), mapa > 0.5)


def test_conteos_histeresis_coinciden_con_las_mascaras():
    from scipy.ndimage import gaussian_filter

    from src.metrics import confusion, matthews
    from src.segment import conteos_histeresis, matthews_conteos, segmentar_histeresis_valor
    rng = np.random.default_rng(1)
    mapa = gaussian_filter(rng.random((80, 90)), 2).astype(np.float32)
    mapa = (mapa - mapa.min()) / np.ptp(mapa)
    verdad = gaussian_filter(rng.random((80, 90)), 2) > 0.52
    campo = np.zeros_like(verdad); campo[5:75, 5:85] = True
    region = np.zeros_like(verdad); region[3:70, 8:88] = True
    altos, bajos = np.array([0.5, 0.6, 0.7, 0.8]), np.array([0.3, 0.6, 0.75])
    c = conteos_histeresis(mapa, campo, verdad, region, altos, bajos)
    for b, bajo in enumerate(bajos):
        for a, alto in enumerate(altos):
            if bajo > alto:
                assert np.isnan(c["tp"][b, a])
                continue
            k = confusion(segmentar_histeresis_valor(mapa, campo, alto, bajo), verdad, region)
            assert c["tp"][b, a] == k["tp"] and c["marcados"][b, a] == k["tp"] + k["fp"]
    for a, alto in enumerate(altos):
        m = (mapa > alto) & campo
        k = confusion(m, verdad, region)
        assert c["tp_fijo"][a] == k["tp"] and c["marcados_fijo"][a] == k["tp"] + k["fp"]
        assert np.isclose(matthews_conteos(c["tp_fijo"][a], c["marcados_fijo"][a], c["positivos"], c["total"]),
                          matthews(m, verdad, region))
    # con bajo = alto, la histéresis es el valor fijo
    assert c["tp"][1, 1] == c["tp_fijo"][1]
