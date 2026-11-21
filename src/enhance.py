"""Realce de vasos y textura.  Esquema v1, etapa 4 (notebooks/04_realce.ipynb).

Convención: todos los filtros reciben un mapa de una sola banda con los vasos
CLAROS (valores altos) y devuelven un mapa del mismo tamaño en el que un valor
alto indica vaso. Los mapas con vasos oscuros se invierten antes con `orient`.
Antes de filtrar se rellena el exterior del campo de visión (`fill_outside`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import oaconvolve
from skimage.morphology import disk, white_tophat

# Sentido de cada mapa de la etapa 2: True si los vasos son claros. Las
# componentes principales quedan con vasos claros por la regla de signo
# "vessels" (color.sign_by_vessels); en el verde y la escala de grises los
# vasos son oscuros porque la hemoglobina absorbe el verde.
BRIGHT_VESSELS = {"PC1": True, "PC2": True, "PC3": True,
                  "Verde": False, "Escala de grises": False,
                  "Azul": True}   # área media 0,508 (etapa 2c): sentido casi indiferente

N_ORIENTATIONS = 12   # cada 15°


# --- Preparación ---------------------------------------------------------------

def orient(m: np.ndarray, name: str) -> np.ndarray:
    """Devuelve el mapa con los vasos claros, invirtiéndolo si hace falta."""
    if name not in BRIGHT_VESSELS:
        raise KeyError(f"Sentido no definido para el mapa {name!r}")
    m = np.asarray(m, dtype=np.float32)
    return m if BRIGHT_VESSELS[name] else -m


def scale_px(scale_rel: float, diameter: float) -> float:
    """Pasa una escala expresada en proporción al diámetro del campo de visión a píxeles."""
    return scale_rel * diameter


def fill_outside(m: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Copia a cada píxel de afuera de `region` el valor del píxel de adentro más cercano.

    Evita el escalón entre la retina y el fondo negro, que los filtros
    confunden con un vaso en el borde del campo de visión.
    """
    _, (iy, ix) = ndi.distance_transform_edt(~region, return_indices=True)
    return np.asarray(m, dtype=np.float32)[iy, ix]


def sigma_for_width(width_px: float) -> float:
    """Escala de la gaussiana que mejor responde a un vaso de ese ancho: σ = ancho / 2.

    Con las derivadas normalizadas por σ² (como en Frangi), la respuesta en
    el centro de una franja de ancho w es máxima en σ = w/2.
    """
    return width_px / 2


# --- Sombrero de copa y Frangi ---------------------------------------------------

def tophat(m: np.ndarray, radius_px: float) -> np.ndarray:
    """Sombrero de copa blanco: resalta estructuras claras más angostas que el disco."""
    radius = max(1, int(round(radius_px)))
    return white_tophat(np.asarray(m, dtype=np.float32), disk(radius))


def frangi_filter(m: np.ndarray, sigmas_px, beta: float = 0.5) -> np.ndarray:
    """Filtro de Frangi (Frangi et al., 1998) para estructuras claras, en las escalas `sigmas_px`.

    En cada escala se calculan los autovalores de la hessiana multiplicados
    por σ² (así las escalas son comparables) y la "vascularidad"
    exp(−R²/2β²) · (1 − exp(−S²/2c²)), con R = |λ1|/|λ2| y S = √(λ1² + λ2²);
    vale 0 donde la curvatura no es la de una cresta clara (λ2 ≥ 0). Se toma,
    en cada píxel, la mayor respuesta entre las escalas.

    La constante c es la misma para todas las escalas: la mitad del mayor S.
    No se usa skimage.filters.frangi porque normaliza cada escala por su
    cuenta, y entonces la respuesta al ruido de la escala más chica le gana a
    los vasos de las escalas grandes (en una imagen sintética con vasos de
    3 a 20 píxeles, área 0,65 contra 0,998 de esta versión). Los bordes de
    la imagen se extienden por reflexión (por defecto, hessian_matrix rellena
    con ceros y el borde de la foto parece un vaso).
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    m = np.asarray(m, dtype=np.float32)
    per_scale = []
    for s in sigmas_px:
        hess = hessian_matrix(m, sigma=s, order="rc", mode="reflect",
                              use_gaussian_derivatives=False)
        e1, e2 = hessian_matrix_eigvals(hess)
        swap = np.abs(e1) > np.abs(e2)               # ordenar: |a| <= |b|
        a = np.where(swap, e2, e1) * s ** 2
        b = np.where(swap, e1, e2) * s ** 2
        per_scale.append((a, b))
    c = 0.5 * max(float(np.sqrt(a ** 2 + b ** 2).max()) for a, b in per_scale)
    if c == 0:                                       # imagen plana: no hay crestas
        return np.zeros(m.shape, np.float32)
    out = None
    for a, b in per_scale:
        r2 = (a / np.where(b == 0, 1e-12, b)) ** 2
        v = np.exp(-r2 / (2 * beta ** 2)) * (1 - np.exp(-(a ** 2 + b ** 2) / (2 * c ** 2)))
        v[b >= 0] = 0                                # cresta clara: curvatura negativa
        out = v if out is None else np.maximum(out, v)
    return out.astype(np.float32)


# --- Filtros orientados: filtro adaptado, detector de líneas y Gabor ----------------

def _axes(radius: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordenadas de un núcleo de (2·radio+1)² píxeles rotadas un ángulo `theta`.

    `u` es la distancia a través del vaso; `v`, a lo largo del vaso.
    """
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    return u, v


def _angles(n_orientations: int) -> np.ndarray:
    return np.arange(n_orientations) * np.pi / n_orientations


def _max_oriented(m: np.ndarray, kernels) -> np.ndarray:
    """Mayor respuesta, píxel a píxel, entre los núcleos (uno por orientación).

    Antes de convolucionar se extiende la imagen por reflexión: con el relleno
    con ceros por defecto, los bordes de la imagen (donde el campo de visión
    toca el borde de la foto, arriba y abajo) darían respuestas falsas.
    """
    m = np.asarray(m, dtype=np.float32)
    radius = max(k.shape[0] for k in kernels) // 2
    padded = np.pad(m, radius, mode="reflect")
    out = None
    for k in kernels:
        r = oaconvolve(padded, k, mode="same")[radius:-radius, radius:-radius].astype(np.float32)
        out = r if out is None else np.maximum(out, r)
    return out


def matched_kernel(sigma_px: float, length_px: float, theta: float) -> np.ndarray:
    """Núcleo del filtro adaptado (Chaudhuri et al., 1989) para vasos claros.

    Perfil gaussiano de desvío `sigma_px` a través del vaso, constante a lo
    largo de `length_px`, recortado en ±3σ y con media cero (un fondo plano da 0).
    """
    radius = int(np.ceil(max(3 * sigma_px, length_px / 2))) + 1
    u, v = _axes(radius, theta)
    support = (np.abs(u) <= 3 * sigma_px) & (np.abs(v) <= length_px / 2)
    k = np.exp(-u ** 2 / (2 * sigma_px ** 2))
    k[support] -= k[support].mean()
    k[~support] = 0
    return (k / np.abs(k).sum()).astype(np.float32)


def matched_filter(m: np.ndarray, sigma_px: float, length_px: float,
                   n_orientations: int = N_ORIENTATIONS) -> np.ndarray:
    """Filtro adaptado en `n_orientations` ángulos; para cada píxel, la mayor respuesta."""
    return _max_oriented(m, [matched_kernel(sigma_px, length_px, t)
                             for t in _angles(n_orientations)])


def line_kernel(length_px: float, theta: float) -> np.ndarray:
    """Segmento de largo `length_px` y un píxel de ancho, normalizado a suma 1."""
    radius = int(np.ceil(length_px / 2))
    u, v = _axes(radius, theta)
    k = ((np.abs(u) <= 0.5) & (np.abs(v) <= length_px / 2)).astype(np.float32)
    return k / k.sum()


def line_detector(m: np.ndarray, length_px: float,
                  n_orientations: int = N_ORIENTATIONS) -> np.ndarray:
    """Detector de líneas (Ricci y Perfetti, 2007) para vasos claros.

    Promedio sobre el mejor segmento de largo `length_px` menos el promedio de
    la ventana cuadrada del mismo lado. Un fondo plano da 0.
    """
    m = np.asarray(m, dtype=np.float32)
    lines = _max_oriented(m, [line_kernel(length_px, t) for t in _angles(n_orientations)])
    side = 2 * int(np.ceil(length_px / 2)) + 1
    return lines - ndi.uniform_filter(m, size=side)


def gabor_kernel(sigma_px: float, theta: float, elongation: float = 2.0) -> np.ndarray:
    """Parte real (par) de un núcleo de Gabor orientado, para vasos claros.

    A través del vaso: coseno de longitud de onda 4·σ bajo una gaussiana de
    desvío `sigma_px`, de modo que el lóbulo positivo central tiene un ancho
    de unos 2·σ, como el vaso (sigma_for_width). A lo largo del vaso: una
    gaussiana `elongation` veces más larga. Media cero (un fondo plano da 0).
    """
    sigma_v = elongation * sigma_px
    radius = int(np.ceil(3 * max(sigma_px, sigma_v)))
    u, v = _axes(radius, theta)
    k = (np.exp(-u ** 2 / (2 * sigma_px ** 2) - v ** 2 / (2 * sigma_v ** 2))
         * np.cos(2 * np.pi * u / (4 * sigma_px)))
    k -= k.mean()
    return (k / np.abs(k).sum()).astype(np.float32)


def gabor_filter(m: np.ndarray, sigma_px: float, elongation: float = 2.0,
                 n_orientations: int = N_ORIENTATIONS) -> np.ndarray:
    """Filtro de Gabor en `n_orientations` ángulos; para cada píxel, la mayor respuesta."""
    return _max_oriented(m, [gabor_kernel(sigma_px, t, elongation)
                             for t in _angles(n_orientations)])


# --- Mapas contra las lesiones (etapa 5h, notebooks/05h_lesiones.ipynb) -------------

def line_opening_reconstruction(m: np.ndarray, length_px: float,
                                n_orientations: int = N_ORIENTATIONS) -> np.ndarray:
    """Apertura por reconstrucción con segmentos (Zana y Klein, 2001), para vasos claros.

    1. Apertura del mapa con un segmento de largo `length_px` en cada una de
       las `n_orientations` direcciones, y en cada píxel la mayor: sobrevive
       lo claro que contiene un segmento de ese largo en alguna dirección
       (los vasos); las manchas más chicas que el segmento (lesiones) bajan
       al nivel del fondo.
    2. Reconstrucción por dilatación de ese resultado (el marcador) debajo del
       mapa original (la máscara), con el cuadrado de 3 × 3: lo que sobrevivió
       recupera su forma exacta, incluidas las partes curvas del vaso que la
       apertura recortó.
    """
    from skimage.morphology import reconstruction

    m = np.asarray(m, dtype=np.float64)
    marker = None
    for t in _angles(n_orientations):
        fp = line_kernel(length_px, t) > 0
        o = ndi.grey_opening(m, footprint=fp, mode="reflect")
        marker = o if marker is None else np.maximum(marker, o)
    return reconstruction(np.minimum(marker, m), m, method="dilation").astype(np.float32)


def fdog_kernel(sigma_px: float, length_px: float, theta: float) -> np.ndarray:
    """Núcleo de la primera derivada de la gaussiana a través del vaso, constante a lo largo.

    Mismo soporte que matched_kernel (±3σ a través, `length_px` a lo largo).
    Es impar: responde a los escalones de brillo (bordes) y no a un perfil
    simétrico centrado (un vaso).
    """
    radius = int(np.ceil(max(3 * sigma_px, length_px / 2))) + 1
    u, v = _axes(radius, theta)
    support = (np.abs(u) <= 3 * sigma_px) & (np.abs(v) <= length_px / 2)
    k = np.where(support, -u / sigma_px ** 2 * np.exp(-u ** 2 / (2 * sigma_px ** 2)), 0.0)
    return (k / np.abs(k).sum()).astype(np.float32)


def fdog_edge(m: np.ndarray, sigma_px: float, length_px: float, window_px: int,
              n_orientations: int = N_ORIENTATIONS) -> np.ndarray:
    """Borde asimétrico (Zhang, Zhang, Zhang y Karray, 2010): |media local de la derivada de la gaussiana|.

    En cada dirección, la respuesta CON SIGNO al núcleo fdog_kernel se
    promedia en una ventana de `window_px` de lado; en cada píxel queda el
    mayor valor absoluto entre las direcciones. Sobre un vaso, los dos bordes
    dan respuestas de signo opuesto que se cancelan en la ventana (perfil
    simétrico); junto a un escalón de brillo (bordes de exudados y
    hemorragias, el borde del disco) no se cancelan. El filtro adaptado, en
    cambio, responde a los dos. La ventana tiene que ser más ancha que el vaso.
    """
    m = np.asarray(m, dtype=np.float32)
    radius = fdog_kernel(sigma_px, length_px, 0.0).shape[0] // 2
    padded = np.pad(m, radius, mode="reflect")
    size = max(3, int(window_px) | 1)
    best = None
    for t in _angles(n_orientations):
        r = oaconvolve(padded, fdog_kernel(sigma_px, length_px, t), mode="same")[radius:-radius, radius:-radius]
        r = np.abs(ndi.uniform_filter(r.astype(np.float32), size=size, mode="reflect"))
        best = r if best is None else np.maximum(best, r)
    return best.astype(np.float32)


# --- Textura: matriz de coocurrencia en una ventana --------------------------------
# Cuatro direcciones (0°, 45°, 90° y 135°); cada medida se promedia entre ellas.
GLCM_OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))   # (fila, columna) para distancia 1


def quantize(m: np.ndarray, levels: int, region: np.ndarray | None = None) -> np.ndarray:
    """Lleva el mapa a `levels` niveles de gris (0 … levels − 1) con igual cantidad de píxeles.

    Los cortes son los cuantiles del mapa dentro de `region` (por ejemplo, el
    campo de visión detectado); sin `region`, de toda la imagen.
    """
    m = np.asarray(m, dtype=np.float32)
    values = m[region] if region is not None else m.ravel()
    cuts = np.quantile(values, np.arange(1, levels) / levels)
    return np.searchsorted(cuts, m, side="right").astype(np.uint8)


def glcm_texture(q: np.ndarray, levels: int, window_px: int,
                 distance: int = 1) -> dict[str, np.ndarray]:
    """Contraste, homogeneidad y energía de la matriz de coocurrencia alrededor de cada píxel.

    `q` es un mapa cuantizado (quantize). Para cada píxel se cuentan los
    pares (píxel, vecino a `distance` en cada dirección) dentro de una
    ventana de `window_px` de lado, con la matriz simétrica y normalizada,
    como graycomatrix(symmetric=True, normed=True). Se calcula con filtros de
    promedio sobre los pares, sin recorrer las ventanas una por una.

    "Energía" es la suma de los cuadrados de la matriz (segundo momento
    angular); graycoprops(..., "energy") da su raíz cuadrada.

    Devuelve un diccionario con los tres mapas. No siguen la convención
    "alto = vaso": su sentido lo decide la combinación de la etapa 5.
    """
    q = np.asarray(q, dtype=np.int16)
    h, w = q.shape
    d = distance
    padded = np.pad(q, d, mode="reflect")
    mean = lambda a: ndi.uniform_filter(a.astype(np.float32), size=window_px, mode="reflect")
    contrast = np.zeros(q.shape, np.float32)
    homogeneity = np.zeros(q.shape, np.float32)
    energy = np.zeros(q.shape, np.float32)
    for dy, dx in GLCM_OFFSETS:
        neighbor = padded[d + dy * d: d + dy * d + h, d + dx * d: d + dx * d + w]
        diff2 = (q - neighbor).astype(np.float32) ** 2
        contrast += mean(diff2)
        homogeneity += mean(1.0 / (1.0 + diff2))
        a, b = np.minimum(q, neighbor), np.maximum(q, neighbor)   # par sin orden
        for i in range(levels):
            for j in range(i, levels):
                p = mean((a == i) & (b == j))
                # matriz simétrica: un par i ≠ j se reparte en (i, j) y (j, i)
                energy += p ** 2 if i == j else p ** 2 / 2
    n = len(GLCM_OFFSETS)
    return {"contraste": contrast / n, "homogeneidad": homogeneity / n, "energía": energy / n}


# --- Preparación de los mapas y resolución de trabajo ---------------------------------

def prepare_maps(rgb: np.ndarray, names: list[str],
                 factor: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Mapas de la etapa 2 listos para filtrar, calculados con la foto reducida `factor` veces.

    1. Reduce la foto promediando bloques de factor × factor (igual que
       color.color_maps_at_scale).
    2. Detecta el campo de visión en la foto reducida y mide su diámetro.
    3. Calcula los mapas `names`, los orienta con los vasos claros y
       rellena el exterior del campo de visión.

    Devuelve (mapas, campo de visión detectado, diámetro en píxeles de la
    foto reducida). Las escalas de los filtros se pasan a píxeles con ese
    diámetro (scale_px), así que se achican solas al reducir.
    """
    from PIL import Image
    from .color import color_maps
    from .fov import detect_fov_robust, fov_diameter

    small = rgb if factor == 1 else np.asarray(Image.fromarray(rgb).reduce(factor))
    region = detect_fov_robust(small)
    diameter = fov_diameter(region)
    maps = color_maps(small, region)
    ready = {n: fill_outside(orient(maps[n], n), region) for n in names}
    return ready, region, diameter


def upscale(out: np.ndarray, shape) -> np.ndarray:
    """Lleva una salida calculada a resolución reducida al tamaño `shape`, con interpolación lineal.

    Así se evalúa siempre a resolución original (plan, sección 3).
    """
    from skimage.transform import resize

    if out.shape == tuple(shape):
        return out
    return resize(out, shape, order=1, preserve_range=True,
                  anti_aliasing=False).astype(np.float32)


# --- Filtros con los parámetros de la etapa 4 -------------------------------------------

def apply_filter(name: str, m: np.ndarray, diameter: float) -> np.ndarray:
    """Aplica el filtro `name` (config.ENHANCE_FILTERS) con las escalas de la etapa 4.

    Las escalas están en `config` en proporción al diámetro del campo de
    visión y se pasan a píxeles con `diameter` (el de la foto con que se
    trabaja), así que se achican solas al reducir la resolución. Los filtros
    orientados se aplican en todas las escalas y se toma, en cada píxel, la
    mayor respuesta; Frangi recibe todas las σ juntas.
    """
    from . import config as c

    widths = [scale_px(r, diameter) for r in c.ENHANCE_WIDTHS_REL]
    sigmas = [max(sigma_for_width(w), c.ENHANCE_SIGMA_MIN_PX) for w in widths]
    m = np.asarray(m, dtype=np.float32)

    if name == "sin filtro":
        return m
    if name == "sombrero de copa":
        return tophat(m, scale_px(c.ENHANCE_TOPHAT_RADIUS_REL, diameter))
    if name == "Frangi":
        return frangi_filter(m, sigmas)
    if name == "filtro adaptado":
        outs = (matched_filter(m, s, c.ENHANCE_MATCHED_LENGTH * s) for s in sigmas)
    elif name == "Gabor":
        outs = (gabor_filter(m, s, c.ENHANCE_GABOR_ELONGATION) for s in sigmas)
    elif name == "detector de líneas":
        outs = (line_detector(m, max(c.ENHANCE_LINE_LENGTH * w, 3)) for w in widths)
    else:
        raise ValueError(f"Filtro desconocido: {name!r}")
    out = None
    for o in outs:
        out = o if out is None else np.maximum(out, o)
    return out

# --- Familias nuevas de mapas (etapa 5l, notebooks/05l_mapas_nuevos.ipynb) -----------------
# Todas reciben el verde normalizado (vasos claros) a la resolución de trabajo, con el exterior del
# campo de visión ya rellenado, y devuelven un mapa del mismo tamaño.

def phase_symmetry(m: np.ndarray, wavelengths, n_orient: int = 8, sigma_f: float = 0.55,
                   theta_on_sigma: float = 1.2) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simetría de fase para estructuras claras (Kovesi, 1999), con filtros log-Gabor en cuadratura.

    Para cada longitud de onda λ y orientación θ, un filtro log-Gabor de un
    solo lado en frecuencia da la respuesta par e (un detector de líneas) y la
    impar o (un detector de bordes). En el centro de una línea clara e > 0 y
    o ≈ 0; en un borde, al revés. La medida es
        Σ_θ [Σ_λ (e − |o|) − T_θ]⁺ / (Σ_θ Σ_λ √(e² + o²) + ε),
    que no depende del contraste: un vaso tenue da casi lo mismo que uno
    marcado (Läthén, Jonasson y Borga, 2010, usan la fase local por eso). T_θ
    descuenta la respuesta típica del fondo: la mediana de la amplitud en cada
    escala (los vasos son una minoría de los píxeles), y ε es el 1 % de la
    mediana de la amplitud total. Devuelve (todas las escalas juntas, una por
    escala con la misma fórmula).
    """
    from scipy import fft

    m = np.asarray(m, dtype=np.float64)
    rows, cols = m.shape
    spectrum = fft.fft2(m - m.mean())
    fy = fft.fftfreq(rows)[:, None]
    fx = fft.fftfreq(cols)[None, :]
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1.0
    angle = np.arctan2(-fy, fx)
    sigma_theta = np.pi / n_orient / theta_on_sigma
    num_all = np.zeros_like(m)
    amp_all = np.zeros_like(m)
    per_scale = []
    for lam in wavelengths:
        radial = np.exp(-np.log(radius * lam) ** 2 / (2 * np.log(sigma_f) ** 2))
        radial[0, 0] = 0.0
        num_s = np.zeros_like(m)
        amp_s = np.zeros_like(m)
        for k in range(n_orient):
            theta = k * np.pi / n_orient
            d = np.abs(np.arctan2(np.sin(angle - theta), np.cos(angle - theta)))
            resp = fft.ifft2(spectrum * radial * np.exp(-d ** 2 / (2 * sigma_theta ** 2)))
            e, o = resp.real, resp.imag
            amp = np.hypot(e, o)
            t = np.median(amp)
            num_s += np.maximum(e - np.abs(o) - t, 0.0)
            amp_s += amp
            num_all += np.maximum(e - np.abs(o) - t, 0.0)
            amp_all += amp
        per_scale.append((num_s / (amp_s + 0.01 * np.median(amp_s))).astype(np.float32))
    return (num_all / (amp_all + 0.01 * np.median(amp_all))).astype(np.float32), per_scale


def bcosfire(m: np.ndarray, sigma: float, rhos, sigma0: float, alpha: float,
             symmetric: bool = True, n_orient: int | None = None) -> np.ndarray:
    """Filtro B-COSFIRE para barras claras (Azzopardi, Strisciuglio, Vento y Petkov, 2015).

    1. Diferencia de gaussianas con centro claro (σ/2 menos σ), rectificada.
    2. Para cada distancia ρ, la respuesta se "difunde" con un máximo
       ponderado por una gaussiana de desvío σ' = σ0 + α·ρ (tolerancia a la
       posición, que crece con la distancia). Se aproxima en escalones: el
       máximo de g(r)·(máximo local de radio r) para r = 0, σ'/2, σ', 3σ'/2 y
       2σ', con g(r) = exp(−r² / 2σ'²).
    3. Para cada orientación θ, media geométrica ponderada (pesos
       exp(−ρ² / 2σ̂²), σ̂ = ρ_max / 3) de las respuestas tomadas a distancia ρ
       a lo largo de θ: a los dos lados del punto (simétrico, responde a lo
       largo de un vaso) o a uno solo (asimétrico, responde también en las
       terminaciones de los vasos). Es un "y": hace falta respuesta en todos
       los puntos, así que una mancha no basta.
    4. El máximo entre orientaciones: 12 en 180° (simétrico) o 24 en 360°.
    """
    m = np.asarray(m, dtype=np.float64)
    dog = np.maximum(ndi.gaussian_filter(m, 0.5 * sigma) - ndi.gaussian_filter(m, sigma), 0.0)
    eps = 1e-3 * max(float(np.quantile(dog, 0.99)), 1e-12)
    rhos = [float(r) for r in rhos]
    blurred = {}
    for r in sorted(set(rhos)):
        sp = sigma0 + alpha * r
        spread = dog.copy()
        for k in range(1, 5):
            radius = int(round(k * sp / 2))
            if radius > 0:
                spread = np.maximum(spread, np.exp(-(k / 2) ** 2 / 2) * ndi.maximum_filter(dog, size=2 * radius + 1))
        blurred[r] = np.log(spread + eps)
    s_hat = max(rhos) / 3 if max(rhos) > 0 else 1.0
    n_orient = n_orient or (12 if symmetric else 24)
    span = np.pi if symmetric else 2 * np.pi
    best = np.full(m.shape, -np.inf)
    for k in range(n_orient):
        theta = k * span / n_orient
        points = []
        for r in rhos:
            angles = [theta, theta + np.pi] if (symmetric and r > 0) else [theta]
            points += [(r, a) for a in angles]
        acc = np.zeros_like(m)
        wsum = 0.0
        for r, a in points:
            w = np.exp(-r ** 2 / (2 * s_hat ** 2))
            dy, dx = -r * np.sin(a), r * np.cos(a)
            acc += w * ndi.shift(blurred[r], (-round(dy), -round(dx)), order=0, mode="nearest")
            wsum += w
        best = np.maximum(best, acc / wsum)
    return (np.exp(best) - eps).astype(np.float32)


def _longest_paths_cone(b: np.ndarray, cone: str) -> np.ndarray:
    """Largo del camino más largo que llega a cada píxel de `b` dentro de un cono de adyacencia.

    Conos de Talbot y Appleton (2007): "vertical" (vecinos de arriba: arriba a
    la izquierda, arriba, arriba a la derecha) y "diagonal" (arriba, arriba a la
    derecha, derecha). Los otros dos conos salen de girar o reflejar la imagen.
    """
    rows, cols = b.shape
    lam = np.zeros(b.shape, dtype=np.int32)
    if cone == "vertical":
        lam[0] = b[0]
        for i in range(1, rows):
            prev = lam[i - 1]
            best = prev.copy()
            best[1:] = np.maximum(best[1:], prev[:-1])
            best[:-1] = np.maximum(best[:-1], prev[1:])
            lam[i] = np.where(b[i], best + 1, 0)
        return lam
    # diagonal: predecesores (i−1, j), (i−1, j+1), (i, j+1); se recorren las diagonales t = i − j
    # en orden creciente. En coordenadas (t, i) los predecesores son (t−1, i−1), (t−1, i) y (t−2, i−1).
    n_t = rows + cols - 1
    s = np.zeros((n_t, rows), dtype=bool)
    ii, jj = np.indices(b.shape)
    tt = ii - jj + cols - 1
    s[tt, ii] = b
    out = np.zeros((n_t, rows), dtype=np.int32)
    for t in range(n_t):
        best = np.zeros(rows, dtype=np.int32)
        if t >= 1:
            best[1:] = out[t - 1, :-1]
            best = np.maximum(best, out[t - 1])
        if t >= 2:
            best[1:] = np.maximum(best[1:], out[t - 2, :-1])
        out[t] = np.where(s[t], best + 1, 0)
    lam[ii, jj] = out[tt, ii]
    return lam


def path_lengths(b: np.ndarray) -> np.ndarray:
    """Para cada píxel de `b`, el largo del camino más largo de `b` que pasa por él, entre los 4 conos.

    En cada cono, el camino que pasa por p mide (lo que llega desde un lado) +
    (lo que llega desde el otro) − 1; el del otro lado es el mismo cálculo con
    la imagen girada 180°.
    """
    b = np.asarray(b, dtype=bool)
    best = np.zeros(b.shape, dtype=np.int32)
    views = [(lambda x: x, lambda x: x, "vertical"),
             (lambda x: x.T, lambda x: x.T, "vertical"),
             (lambda x: x, lambda x: x, "diagonal"),
             (lambda x: x[:, ::-1], lambda x: x[:, ::-1], "diagonal")]
    for fwd, back, cone in views:
        v = fwd(b)
        up = _longest_paths_cone(v, cone)
        down = _longest_paths_cone(v[::-1, ::-1], cone)[::-1, ::-1]
        best = np.maximum(best, back(np.where(v, up + down - 1, 0)))
    return best


def path_openings(m: np.ndarray, region: np.ndarray, lengths, n_levels: int = 16,
                  q_low: float = 0.5, q_high: float = 0.99) -> list[np.ndarray]:
    """Aperturas por caminos en escala de grises (Talbot y Appleton, 2007), una por largo.

    Una apertura con segmentos rectos (line_opening_reconstruction) conserva
    lo claro que contiene un segmento de largo L; una apertura por caminos
    conserva lo claro que contiene un camino de L píxeles que puede curvarse
    (en cada paso avanza hacia uno de tres vecinos de un cono), así que no
    pierde los vasos finos tortuosos. En escala de grises se descompone por
    umbrales: `n_levels` niveles, cuantiles de `q_low` a `q_high` del mapa en
    el campo, y cada píxel recibe el nivel más alto en que pertenece a un
    camino de largo ≥ L. Debajo del nivel más bajo queda ese nivel.

    Limitación: un camino no cambia de cono, así que sigue curvas que no se
    apartan más de 45° de una dirección; un vaso que gira más dentro de un
    largo L queda cortado en tramos.
    """
    m = np.asarray(m, dtype=np.float64)
    region = np.asarray(region, dtype=bool)
    levels = np.quantile(m[region], np.linspace(q_low, q_high, n_levels))
    outs = [np.full(m.shape, levels[0]) for _ in lengths]
    for h in levels:
        lengths_here = path_lengths((m >= h) & region)
        for out, length in zip(outs, lengths):
            out[lengths_here >= length] = h
    return [o.astype(np.float32) for o in outs]


def hessian_eigenvalues(m: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Los dos autovalores del hessiano a escala σ, normalizados por σ² (λ1 ≥ λ2).

    Frangi los resume en un solo número con parámetros fijos; sueltos, la
    regresión aprende cómo combinarlos (Niemeijer et al., 2004, y Staal et
    al., 2004, usan derivadas gaussianas de segundo orden como características).
    En el centro de un vaso claro, λ2 es muy negativo (curvatura a través del
    vaso) y λ1 casi cero (a lo largo).
    """
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    h = hessian_matrix(np.asarray(m, dtype=np.float64), sigma=sigma, order="rc",
                       use_gaussian_derivatives=False)
    l1, l2 = hessian_matrix_eigvals(h)
    return (sigma ** 2 * l1).astype(np.float32), (sigma ** 2 * l2).astype(np.float32)
