"""Preprocesamiento de los mapas de características."""

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter

# Modos de normalizar_fondo (ronda de normalización, notebooks/05g_normalizacion.ipynb).
MODOS_FONDO = ("resta", "división", "luminosidad y contraste")


def corregir_fondo(mapa, campo, sigma):
    """Resta un fondo suave estimado dentro del campo de visión."""
    mapa = np.asarray(mapa, dtype=np.float32)
    campo = np.asarray(campo, dtype=bool)

    numerador = gaussian_filter(
        np.where(campo, mapa, 0.0), sigma=sigma
    )
    denominador = gaussian_filter(
        campo.astype(np.float32), sigma=sigma
    )

    fondo = numerador / np.maximum(denominador, 1e-8)
    return mapa - fondo


def _media_local(valores: np.ndarray, pesos: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Media gaussiana de `valores` usando solo los píxeles con `pesos` (convolución normalizada).

    Devuelve (media, suma de pesos suavizada); la segunda sirve para saber
    dónde casi no hubo píxeles que promediar.
    """
    den = gaussian_filter(pesos, sigma=sigma)
    num = gaussian_filter(np.where(pesos > 0, valores, 0.0), sigma=sigma)
    return num / np.maximum(den, 1e-8), den


def normalizar_fondo(mapa, campo, sigma, modo: str = "resta"):
    """Corrige la iluminación despareja del mapa, con tres modelos posibles.

    `mapa` es el verde orientado con los vasos claros (−verde, valores
    negativos), como lo devuelve enhance.prepare_maps; `campo` es el campo de
    visión detectado y `sigma` la escala de la gaussiana, en píxeles.

    - "resta": mapa − fondo (corregir_fondo). Supone una iluminación aditiva.
    - "división": (mapa − fondo) / |fondo|. Como el mapa es −verde, es
      1 − verde / fondo: el contraste relativo del vaso. Supone una
      iluminación multiplicativa (verde = iluminación × reflectancia).
    - "luminosidad y contraste": (mapa − L) / C, con L y C la media y el desvío
      locales estimados solo con los píxeles de fondo: los que están a menos de
      un desvío de la media local en una primera pasada con todos los píxeles
      (Foracchia, Grisan y Ruggeri, 2005, con ventanas gaussianas en lugar de
      su grilla). Corrige también el contraste, que cae hacia la periferia.

    El exterior del campo no entra en las estimaciones.
    """
    if modo not in MODOS_FONDO:
        raise ValueError(f"Modo de fondo desconocido: {modo!r}")
    mapa = np.asarray(mapa, dtype=np.float64)
    campo = np.asarray(campo, dtype=bool)
    if modo == "resta":
        return corregir_fondo(mapa, campo, sigma)
    pesos = campo.astype(np.float64)
    fondo, _ = _media_local(mapa, pesos, sigma)
    if modo == "división":
        escala = np.abs(fondo)
        piso = 1e-3 * max(float(escala[campo].max()), 1e-12)
        return ((mapa - fondo) / np.maximum(escala, piso)).astype(np.float32)
    # luminosidad y contraste
    m2, _ = _media_local(mapa ** 2, pesos, sigma)
    desvio = np.sqrt(np.maximum(m2 - fondo ** 2, 0.0))
    es_fondo = (campo & (np.abs(mapa - fondo) <= desvio)).astype(np.float64)
    lum, den = _media_local(mapa, es_fondo, sigma)
    l2, _ = _media_local(mapa ** 2, es_fondo, sigma)
    con = np.sqrt(np.maximum(l2 - lum ** 2, 0.0))
    poco = den < 1e-3                      # casi sin píxeles de fondo cerca: primera pasada
    lum = np.where(poco, fondo, lum)
    con = np.where(poco, desvio, con)
    piso = 1e-3 * max(float(con[campo].max()), 1e-12)
    return ((mapa - lum) / np.maximum(con, piso)).astype(np.float32)


# --- Relleno de exudados (etapa 5i, notebooks/05i_exudados.ipynb) --------------------
SUFIJO_RELLENO = " + relleno de exudados"


def rellenar_exudados(mapa, campo, diametro, k=4.0, radio_disco_rel=0.10, area_min_rel=0.0025,
                      dilatacion_rel=0.003, sigma_relleno_rel=0.01, sigma_fondo_rel=0.02):
    """Detecta las lesiones claras (exudados) y las rellena con una estimación de su entorno.

    Sigue la idea de Annunziata et al. (2016): rellenar los exudados antes del
    realce, para que sus bordes no parezcan vasos. `mapa` es el verde orientado
    (−verde), como lo devuelve enhance.prepare_maps.

    1. Brillo relativo del verde: verde / fondo − 1 (fondo gaussiano de
       `sigma_fondo_rel` × diámetro). Es candidato lo que supera `k` desvíos
       robustos (1,4826 × desviación mediana absoluta) dentro del campo.
    2. Se excluye un círculo de radio `radio_disco_rel` × diámetro alrededor
       de la zona más clara (el disco óptico, que también es claro y por el
       que pasan vasos) y las componentes de menos de (`area_min_rel` ×
       diámetro)² píxeles.
    3. Se dilatan los candidatos `dilatacion_rel` × diámetro, pero solo hacia
       píxeles más claros que el fondo (así el relleno nunca tapa un vaso, que
       es más oscuro), y se rellenan con la media gaussiana (σ =
       `sigma_relleno_rel` × diámetro) de los píxeles de fondo cercanos: los que
       no son candidatos y están a menos de 2 desvíos robustos del fondo
       (convolución normalizada).

    Devuelve (mapa rellenado, máscara de lo rellenado). Los parámetros son de
    criterio, en proporción al diámetro del campo; no se ajustaron con datos.
    """
    m = np.asarray(mapa, dtype=np.float64)
    campo = np.asarray(campo, dtype=bool)
    g = -m
    pesos = campo.astype(np.float64)
    fondo, _ = _media_local(g, pesos, sigma_fondo_rel * diametro)
    rel = g / np.maximum(np.abs(fondo), 1e-6) - 1
    v = rel[campo]
    escala = 1.4826 * np.median(np.abs(v - np.median(v)))
    cand = campo & (rel > np.median(v) + k * escala)
    suave, _ = _media_local(g, pesos, 0.03 * diametro)
    cy, cx = np.unravel_index(np.argmax(np.where(campo, suave, -np.inf)), g.shape)
    yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
    cand &= (yy - cy) ** 2 + (xx - cx) ** 2 > (radio_disco_rel * diametro) ** 2
    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    if n:
        area = np.bincount(lab.ravel())
        cand = (area >= (area_min_rel * diametro) ** 2)[lab] & cand
    r = max(1, int(round(dilatacion_rel * diametro)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    relleno = ndi.binary_dilation(cand, structure=(yy ** 2 + xx ** 2 <= r * r)) & campo & (rel > 0)
    if not relleno.any():
        return m.astype(np.float32), relleno
    de_fondo = campo & ~relleno & (np.abs(rel - np.median(v)) < 2 * escala)
    estimado, _ = _media_local(m, de_fondo.astype(np.float64), sigma_relleno_rel * diametro)
    return np.where(relleno, estimado, m).astype(np.float32), relleno


# --- Wiener con el ruido estimado en cada imagen (etapa 5k, notebooks/05k_wiener.ipynb) ------
PASO_WIENER_ESTIMADO = "Wiener con ruido estimado"
PASO_WIENER_LOCAL = "Wiener local"
PASO_RELLENO = SUFIJO_RELLENO[3:]
PASOS_PREVIOS = (PASO_WIENER_ESTIMADO, PASO_WIENER_LOCAL, PASO_RELLENO)   # en el orden en que se aplican

# Máscara de Immerkær (1996): dos laplacianos restados; anula planos y rampas.
_MASCARA_RUIDO = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def estimar_ruido(mapa, campo, margen=3):
    """Desvío del ruido blanco del mapa, estimado dentro del campo de visión.

    Filtra con la máscara de Immerkær (1996), que deja pasar casi solo el ruido
    (con ruido de desvío σ, la salida tiene desvío 6σ), y toma la mediana del
    valor absoluto en lugar de la media, como la desviación mediana absoluta
    de Donoho y Johnstone (1994): así los bordes de los vasos, que también
    responden, no inflan la estimación mientras sean una minoría de los
    píxeles. Se excluye una franja de `margen` píxeles junto al borde del campo.
    """
    m = np.asarray(mapa, dtype=np.float64)
    campo = np.asarray(campo, dtype=bool)
    interior = ndi.binary_erosion(campo, iterations=margen) if margen > 0 else campo
    respuesta = ndi.convolve(m, _MASCARA_RUIDO, mode="reflect")
    return float(np.median(np.abs(respuesta[interior])) / (0.6745 * 6))


def wiener_ruido_estimado(mapa, campo, tamaño=5):
    """Filtro de Wiener local (Lee, 1980) con la varianza del ruido estimada en la imagen.

    En cada ventana de `tamaño` × `tamaño`: media + (1 − ruido / varianza) ×
    (valor − media), y la media donde la varianza no supera al ruido. Es lo
    que hace scipy.signal.wiener, salvo por el ruido: scipy usa el promedio de
    las varianzas locales, que en una foto limpia sale alto por los propios
    vasos y la suaviza igual. Con estimar_ruido, en una foto limpia el ruido es
    chico y el filtro casi no cambia nada; en una ruidosa, suaviza el fondo y
    respeta los vasos (donde la varianza local es mucho mayor que el ruido).
    Afuera del campo de visión deja el mapa como estaba.
    """
    m = np.asarray(mapa, dtype=np.float64)
    campo = np.asarray(campo, dtype=bool)
    ruido = estimar_ruido(m, campo) ** 2
    media = ndi.uniform_filter(m, tamaño, mode="reflect")
    var = np.maximum(ndi.uniform_filter(m * m, tamaño, mode="reflect") - media ** 2, 0.0)
    ganancia = np.maximum(var - ruido, 0.0) / np.maximum(var, 1e-12)
    return np.where(campo, media + ganancia * (m - media), m).astype(np.float32)


def preprocesar_verde(mapa, campo, diametro, modo, sigma_fondo_rel):
    """Preprocesamiento del verde orientado según `modo`.

    `modo` es un modo de normalizar_fondo seguido, opcionalmente, de pasos
    previos separados por " + ", por ejemplo "división + relleno de exudados".
    Los pasos se aplican en el orden de PASOS_PREVIOS, sea cual sea el orden en
    que estén escritos: Wiener (con el ruido estimado, wiener_ruido_estimado, o
    el de la etapa 3, restoration.wiener_denoise), después el relleno de los
    exudados (rellenar_exudados) y al final la normalización del fondo.
    """
    from .config import WIENER_SIZE

    base, *pasos = modo.split(" + ")
    desconocidos = set(pasos) - set(PASOS_PREVIOS)
    if desconocidos:
        raise ValueError(f"Paso de preprocesamiento desconocido: {sorted(desconocidos)}")
    if PASO_WIENER_ESTIMADO in pasos:
        mapa = wiener_ruido_estimado(mapa, campo, WIENER_SIZE)
    if PASO_WIENER_LOCAL in pasos:
        from .restoration import wiener_denoise
        mapa = wiener_denoise(mapa, campo, WIENER_SIZE)
    if PASO_RELLENO in pasos:
        mapa, _ = rellenar_exudados(mapa, campo, diametro, sigma_fondo_rel=sigma_fondo_rel)
    return normalizar_fondo(mapa, campo, sigma_fondo_rel * diametro, base)
