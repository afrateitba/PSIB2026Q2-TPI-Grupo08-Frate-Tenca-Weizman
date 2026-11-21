"""Preprocesamiento de los mapas de características."""

import numpy as np
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
    from scipy import ndimage as ndi

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


def preprocesar_verde(mapa, campo, diametro, modo, sigma_fondo_rel):
    """Preprocesamiento del verde orientado según `modo`: un modo de normalizar_fondo, con
    SUFIJO_RELLENO al final si antes se rellenan los exudados (rellenar_exudados)."""
    if modo.endswith(SUFIJO_RELLENO):
        modo = modo[: -len(SUFIJO_RELLENO)]
        mapa, _ = rellenar_exudados(mapa, campo, diametro, sigma_fondo_rel=sigma_fondo_rel)
    return normalizar_fondo(mapa, campo, sigma_fondo_rel * diametro, modo)
