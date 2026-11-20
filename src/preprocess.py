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
