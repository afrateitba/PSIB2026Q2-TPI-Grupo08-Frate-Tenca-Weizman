import numpy as np
import pytest

from src.config import ENHANCE_FILTERS
from src.enhance import (apply_filter, fill_outside, frangi_filter, gabor_filter,
                         glcm_texture, line_detector, matched_filter, orient,
                         prepare_maps, quantize, scale_px, sigma_for_width, tophat,
                         upscale)


def synthetic_line(width=5, dark=True, n=101):
    """Imagen plana con una línea vertical de ancho conocido en el centro."""
    background, vessel = (100.0, 50.0) if dark else (50.0, 100.0)
    img = np.full((n, n), background, dtype=np.float32)
    c = n // 2
    img[:, c - width // 2: c + width // 2 + 1] = vessel
    line = np.zeros((n, n), bool)
    line[:, c - width // 2: c + width // 2 + 1] = True
    return img, line


def diagonal_line(width=5, n=101):
    """Línea clara a 45°, de ancho aproximado `width`, sobre fondo plano."""
    y, x = np.mgrid[:n, :n]
    line = np.abs(x - y) / np.sqrt(2) <= width / 2
    return np.where(line, 100.0, 50.0).astype(np.float32), line


# --- Preparación ---------------------------------------------------------------

def test_scale_px():
    assert scale_px(0.001, 3262) == pytest.approx(3.262)


def test_orient_flips_green():
    m = np.array([[0.2, 0.8]], dtype=np.float32)
    assert np.array_equal(orient(m, "Verde"), -m)
    assert np.array_equal(orient(m, "PC3"), m)


def test_fill_outside_removes_step():
    img = np.zeros((50, 50), np.float32)
    region = np.zeros((50, 50), bool)
    region[10:40, 10:40] = True
    img[region] = 7.0
    assert np.all(fill_outside(img, region) == 7.0)   # afuera copia el valor del borde de adentro


# --- Sombrero de copa y Frangi -------------------------------------------------

def test_tophat_highlights_line():
    img, line = synthetic_line(width=5, dark=False)
    out = tophat(img, radius_px=6)
    assert out[line].mean() == pytest.approx(50.0)          # todo el contraste en la línea
    assert np.abs(out[~line]).max() == pytest.approx(0.0)   # fondo plano → cero


def test_tophat_ignores_wide_structures():
    # una banda de 31 píxeles no entra en un disco de radio 6: no se resalta entera
    img, line = synthetic_line(width=31, dark=False)
    assert tophat(img, radius_px=6)[line].mean() < 5


def test_frangi_highlights_line():
    img, line = synthetic_line(width=5, dark=False)
    out = frangi_filter(img, [sigma_for_width(5)])
    c = img.shape[0] // 2
    assert out[10:-10, c].min() > 10 * out[10:-10, :c - 10].max()


def test_frangi_multiscale_is_not_dominated_by_noise():
    # vasos de 3 a 20 píxeles con perfil suave y ruido: con todas las escalas juntas,
    # Frangi tiene que separar casi tan bien como con la mejor escala sola
    from src.metrics import auc
    rng = np.random.default_rng(0)
    n = 400
    img = np.zeros((n, n), np.float32)
    mask = np.zeros((n, n), bool)
    for x, w, c in [(60, 3, 6), (140, 6, 8), (240, 12, 10), (330, 20, 12)]:
        xx = np.arange(n) - x
        img += (c * np.exp(-xx ** 2 / (2 * (w / 2.5) ** 2)))[None, :]
        mask[:, max(0, x - w // 2 - 1): x + w // 2 + 2] = True
    img += rng.normal(0, 2, img.shape).astype(np.float32)
    assert auc(frangi_filter(img, [1, 2, 4, 8, 12]), mask) > 0.95


def test_frangi_ignores_fov_edge_after_filling():
    # un disco claro (la retina) sobre negro: sin rellenar, el borde parece un vaso
    n = 201
    yy, xx = np.mgrid[:n, :n]
    region = np.hypot(yy - n // 2, xx - n // 2) < 80
    img = np.where(region, 1.0, 0.0).astype(np.float32)
    edge = region & ~np.pad(region[2:-2, 2:-2] & region[:-4, 2:-2] & region[4:, 2:-2]
                            & region[2:-2, :-4] & region[2:-2, 4:], 2)
    raw = frangi_filter(img, [2.0])
    filled = frangi_filter(fill_outside(img, region), [2.0])
    assert filled[edge].max() < 0.1 * raw[edge].max()


# --- Filtros orientados ----------------------------------------------------------

ORIENTED = [
    lambda m: matched_filter(m, sigma_px=2.5, length_px=9),
    lambda m: line_detector(m, length_px=15),
    lambda m: gabor_filter(m, sigma_px=2.5),
]


@pytest.mark.parametrize("f", ORIENTED)
def test_oriented_highlights_line_not_background(f):
    img, line = synthetic_line(width=5, dark=False)
    out = f(img)
    c = img.shape[0] // 2
    assert out[20:-20, c].min() > 1                      # centro de la línea: positivo
    assert np.abs(out[20:-20, 5:c - 20]).max() < 1e-3    # fondo plano, también junto al borde: cero


@pytest.mark.parametrize("f", ORIENTED)
def test_oriented_works_on_diagonal(f):
    vertical, _ = synthetic_line(width=5, dark=False)
    diagonal, _ = diagonal_line(width=5)
    c = vertical.shape[0] // 2
    assert f(diagonal)[c, c] == pytest.approx(f(vertical)[c, c], rel=0.25)   # no depende de la orientación


@pytest.mark.parametrize("f", [ORIENTED[0], ORIENTED[2]])
def test_matched_and_gabor_ignore_wide_structures(f):
    narrow, _ = synthetic_line(width=5, dark=False)
    wide, _ = synthetic_line(width=31, dark=False)
    c = narrow.shape[0] // 2
    assert f(wide)[c, c] < 0.1 * f(narrow)[c, c]         # el centro de una meseta da casi 0


# --- Textura -----------------------------------------------------------------------

def test_quantize_splits_evenly():
    m = np.random.default_rng(2026).random((100, 100)).astype(np.float32)
    q = quantize(m, 8)
    assert q.min() == 0 and q.max() == 7
    counts = np.bincount(q.ravel(), minlength=8)
    assert counts.min() > 0.9 * counts.max()       # cada nivel con ~1/8 de los píxeles


def test_glcm_constant_image():
    t = glcm_texture(np.full((40, 40), 3, np.uint8), levels=8, window_px=7)
    assert np.allclose(t["contraste"], 0)
    assert np.allclose(t["homogeneidad"], 1)
    assert np.allclose(t["energía"], 1)


def test_glcm_vertical_stripes():
    # columnas 0, 1, 0, 1…: el vecino horizontal y los diagonales siempre difieren en 1;
    # el vertical, nunca. Promedio de las 4 direcciones: contraste 3/4, homogeneidad
    # (3 · 1/2 + 1) / 4 = 0,625; energía 1/2 en cada dirección.
    # ventana de lado par: cubre tantas columnas 0 como 1
    q = np.tile(np.array([0, 1], np.uint8), (40, 20))
    t = glcm_texture(q, levels=2, window_px=8)
    center = (slice(10, 30), slice(10, 30))
    assert np.allclose(t["contraste"][center], 0.75)
    assert np.allclose(t["homogeneidad"][center], 0.625)
    assert np.allclose(t["energía"][center], 0.5)


def test_glcm_matches_skimage():
    # con una ventana que cubre casi toda la imagen, el contraste y la energía
    # (promediados en las 4 direcciones) coinciden con graycomatrix
    from skimage.feature import graycomatrix, graycoprops
    rng = np.random.default_rng(2026)
    q = rng.integers(0, 4, (60, 60)).astype(np.uint8)
    t = glcm_texture(q, levels=4, window_px=61)
    g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=4,
                     symmetric=True, normed=True)
    assert t["contraste"][30, 30] == pytest.approx(graycoprops(g, "contrast").mean(), rel=0.05)
    assert t["energía"][30, 30] == pytest.approx((graycoprops(g, "energy") ** 2).mean(), rel=0.05)


# --- Preparación de los mapas y resolución -------------------------------------------

def synthetic_photo():
    """Retina rojiza recortada arriba y abajo (como en la base), con un vaso vertical oscuro."""
    rng = np.random.default_rng(2026)
    h, w = 400, 600
    yy, xx = np.mgrid[:h, :w]
    fov = np.hypot(yy - h / 2, xx - w / 2) < 250
    vessel = fov & (np.abs(xx - 330) <= 3)
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[fov] = (180, 90, 40)
    rgb[vessel] = (150, 50, 30)
    rgb[fov] += rng.normal(0, 3, (fov.sum(), 3))
    return rgb.clip(0, 255).astype(np.uint8), fov, vessel


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_prepare_maps_reduces_and_measures_diameter(factor):
    rgb, fov, vessel = synthetic_photo()
    maps, region, diameter = prepare_maps(rgb, ["Verde", "PC3"], factor)
    assert maps["Verde"].shape == (400 // factor, 600 // factor)
    assert diameter == pytest.approx(500 / factor, abs=3)


def test_prepare_maps_orients_and_fills():
    rgb, fov, vessel = synthetic_photo()
    maps, region, _ = prepare_maps(rgb, ["Verde"])
    green = maps["Verde"]
    assert green[vessel].mean() > green[fov & ~vessel].mean()     # vaso claro tras invertir
    inside = green[region]
    assert inside.min() - 1e-6 <= green[0, 0] <= inside.max() + 1e-6   # sin escalón afuera


def test_upscale_returns_original_shape():
    assert upscale(np.ones((100, 150), np.float32), (400, 600)).shape == (400, 600)


@pytest.mark.parametrize("factor", [1, 2])
def test_frangi_at_reduced_resolution_finds_vessel(factor):
    from src.metrics import auc
    rgb, fov, vessel = synthetic_photo()
    maps, _, diameter = prepare_maps(rgb, ["Verde"], factor)
    sigma = scale_px(sigma_for_width(7) / 500, diameter)     # vaso de 7 px en la original
    out = upscale(frangi_filter(maps["Verde"], [sigma]), rgb.shape[:2])
    assert auc(out, vessel, fov) > 0.95


# --- Filtros con los parámetros de la etapa 4 ------------------------------------------

def line_for_filters(width=5, n=121):
    img = np.full((n, n), 50.0, np.float32)
    c = n // 2
    img[:, c - width // 2: c + width // 2 + 1] = 100.0
    return img, c


@pytest.mark.parametrize("name", ENHANCE_FILTERS[1:])
def test_each_filter_highlights_line(name):
    img, c = line_for_filters()
    out = apply_filter(name, img, diameter=3262)
    assert out.shape == img.shape
    assert out[30:-30, c].mean() > out[30:-30, 5:c - 25].mean()


def test_no_filter_returns_map():
    img, _ = line_for_filters()
    assert np.array_equal(apply_filter("sin filtro", img, 3262), img)


def test_unknown_filter():
    img, _ = line_for_filters()
    with pytest.raises(ValueError):
        apply_filter("Sobel", img, 3262)


@pytest.mark.parametrize("name", ENHANCE_FILTERS[1:])
def test_works_with_small_diameter(name):
    # a la cuarta parte de la resolución las σ más chicas quedan en el piso de 0,5
    img, c = line_for_filters(width=3)
    assert np.isfinite(apply_filter(name, img, diameter=3262 / 4)).all()

# --- Mapas contra las lesiones (etapa 5h) ------------------------------------------

def _linea_y_mancha(n=121):
    """Fondo 0, una línea clara horizontal de 3 píxeles de ancho y una mancha clara redonda de radio 5."""
    yy, xx = np.mgrid[:n, :n]
    img = np.zeros((n, n), np.float32)
    img[30:33, 10:111] = 1.0
    img[(yy - 85) ** 2 + (xx - 60) ** 2 <= 25] = 1.0
    return img


def test_apertura_por_reconstruccion_conserva_la_linea_y_borra_la_mancha():
    from src.enhance import line_opening_reconstruction

    img = _linea_y_mancha()
    out = line_opening_reconstruction(img, length_px=21)
    assert out[31, 20:100].min() > 0.99          # la línea queda entera
    assert out[80:91, 55:66].max() < 0.01        # la mancha (más chica que el segmento) desaparece


def test_borde_asimetrico_alto_en_escalon_y_bajo_en_linea():
    from src.enhance import fdog_edge

    n = 101
    linea = np.zeros((n, n), np.float32); linea[:, 49:52] = 1.0          # vaso: perfil simétrico
    escalon = np.zeros((n, n), np.float32); escalon[:, 50:] = 1.0         # borde de una lesión
    kw = dict(sigma_px=1.5, length_px=7, window_px=15)   # ventana más ancha que el vaso
    assert fdog_edge(escalon, **kw)[50, 50] > 2 * fdog_edge(linea, **kw)[50, 50]
    assert np.allclose(fdog_edge(np.ones((n, n), np.float32), **kw), 0, atol=1e-6)   # fondo plano: 0


# --- Familias nuevas (etapa 5l) ---------------------------------------------------------

def _linea_clara(n=96, contraste=1.0, ancho=2.0):
    yy, xx = np.mgrid[:n, :n]
    return contraste * np.exp(-((yy - n // 2) ** 2) / (2 * (ancho / 2) ** 2))


def test_simetria_de_fase_marca_lineas_no_bordes_y_no_depende_del_contraste():
    from src.enhance import phase_symmetry
    n = 96
    escalon = np.zeros((n, n)); escalon[:, 70:] = 1.0              # un borde vertical
    fuerte, _ = phase_symmetry(_linea_clara(n, 1.0) + escalon, [4, 8])
    tenue, _ = phase_symmetry(_linea_clara(n, 0.2) + escalon, [4, 8])
    centro = fuerte[n // 2, 10:60].mean()
    assert centro > 0.3
    assert fuerte[20:40, 69:71].mean() < 0.3 * centro              # el borde casi no responde
    assert tenue[n // 2, 10:60].mean() > 0.6 * centro               # la línea tenue da parecido


def test_bcosfire_prefiere_una_barra_a_una_mancha():
    from src.enhance import bcosfire
    n = 120
    img = _linea_clara(n)
    img[:, :20] = 0                                                  # barra de x = 20 al final
    yy, xx = np.mgrid[:n, :n]
    img += np.exp(-((yy - 20) ** 2 + (xx - 60) ** 2) / 2.0)         # mancha suelta del mismo ancho
    r = bcosfire(img, sigma=2.0, rhos=[0, 3, 6, 9, 12], sigma0=1.0, alpha=0.1)
    assert r[n // 2, 60] > 3 * r[20, 60]


def _brute_paths(b):
    """Camino más largo por cada píxel en los 4 conos, por búsqueda directa (memoizada)."""
    from functools import lru_cache
    rows, cols = b.shape
    cones = [[(-1, -1), (-1, 0), (-1, 1)], [(-1, -1), (0, -1), (1, -1)],
             [(-1, 0), (-1, 1), (0, 1)], [(-1, 0), (-1, -1), (0, -1)]]
    best = np.zeros(b.shape, int)
    for cone in cones:
        @lru_cache(None)
        def run(i, j, sign):
            if not (0 <= i < rows and 0 <= j < cols) or not b[i, j]:
                return 0
            return 1 + max(run(i + sign * di, j + sign * dj, sign) for di, dj in cone)
        for i in range(rows):
            for j in range(cols):
                if b[i, j]:
                    best[i, j] = max(best[i, j], run(i, j, 1) + run(i, j, -1) - 1)
    return best


def test_largo_de_caminos_coincide_con_la_busqueda_directa():
    from src.enhance import path_lengths
    rng = np.random.default_rng(3)
    for _ in range(5):
        b = rng.random((9, 13)) < 0.55
        np.testing.assert_array_equal(path_lengths(b), _brute_paths(b))


def test_apertura_por_caminos_conserva_el_vaso_tortuoso_y_borra_la_mancha():
    from src.enhance import line_opening_reconstruction, path_openings
    n = 100
    img = 0.1 * np.random.default_rng(0).random((n, n))           # fondo con algo de textura
    x = np.arange(5, 95)
    y = (50 + 5 * np.sin(x / 8)).round().astype(int)               # vaso fino ondulado (hasta 32°)
    img[y, x] = 1.0
    img[y + 1, x] = 1.0
    img[10:15, 10:15] = 1.0                                         # mancha de 5 × 5
    region = np.ones_like(img, bool)
    (out,) = path_openings(img, region, [40], n_levels=6, q_low=0.5, q_high=0.99)
    assert out[y, x].min() == out.max()                             # el vaso queda entero
    assert out[12, 12] < out.max()                                  # la mancha no
    recta = line_opening_reconstruction(img, 40)
    assert recta[y, x].mean() < 0.5                                 # con segmentos rectos se pierde


def test_autovalores_del_hessiano_en_una_linea_clara():
    from src.enhance import hessian_eigenvalues
    l1, l2 = hessian_eigenvalues(_linea_clara(64, ancho=4.0), sigma=2.0)
    assert l2[32, 32] < -0.1 and abs(l1[32, 32]) < 0.05 * abs(l2[32, 32])
