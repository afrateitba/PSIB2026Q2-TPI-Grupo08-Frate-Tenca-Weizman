"""Funciones para obtener máscaras de vasos."""

import numpy as np
from skimage.filters import threshold_otsu


def segmentar_otsu(mapa, campo):
    """Segmenta un mapa con vasos claros dentro del campo de visión."""
    mapa = np.asarray(mapa)
    campo = np.asarray(campo, dtype=bool)

    valores = mapa[campo]

    if valores.size == 0:
        raise ValueError("El campo de visión está vacío.")

    umbral = threshold_otsu(valores)
    mascara = (mapa > umbral) & campo

    return mascara, float(umbral)


def segmentar_fraccion(mapa, campo, fraccion):
    """Marca como vaso la `fraccion` de píxeles del campo con valores más altos.

    Es un umbral distinto en cada imagen, pero con la misma proporción de
    vasos: se adapta al brillo de cada foto como Otsu, sin suponer que el
    histograma tenga dos grupos.
    """
    mapa = np.asarray(mapa)
    campo = np.asarray(campo, dtype=bool)
    valores = mapa[campo]
    if valores.size == 0:
        raise ValueError("El campo de visión está vacío.")
    umbral = np.quantile(valores, 1 - fraccion)
    return (mapa > umbral) & campo, float(umbral)


def segmentar_histeresis(mapa, campo, frac_alto, frac_bajo):
    """Umbral con histéresis, con los dos umbrales como fracciones de píxeles del campo.

    El umbral alto deja como vaso la `frac_alto` de píxeles más altos (vasos
    seguros); el bajo, la `frac_bajo` más alta (con frac_bajo > frac_alto),
    pero esos píxeles solo quedan si están conectados (cuatro vecinos: la
    conexión por defecto de skimage.filters.apply_hysteresis_threshold) a un
    vaso seguro. Así se recuperan vasos finos débiles que continúan a uno
    grueso, sin aceptar manchas sueltas del fondo.
    """
    from skimage.filters import apply_hysteresis_threshold

    if not 0 < frac_alto < frac_bajo < 1:
        raise ValueError("Se necesita 0 < frac_alto < frac_bajo < 1.")
    mapa = np.asarray(mapa, dtype=np.float32)
    campo = np.asarray(campo, dtype=bool)
    valores = mapa[campo]
    if valores.size == 0:
        raise ValueError("El campo de visión está vacío.")
    alto, bajo = np.quantile(valores, [1 - frac_alto, 1 - frac_bajo])
    dentro = np.where(campo, mapa, valores.min())
    return apply_hysteresis_threshold(dentro, bajo, alto) & campo, (float(bajo), float(alto))


def limpiar_por_forma(mascara, alargamiento_min, area_max):
    """Quita las componentes chicas que no son alargadas.

    Para cada componente (ocho vecinos): área A y largo L de su línea
    central. El alargamiento L² / A vale alrededor de largo / ancho: es
    grande en un vaso y chico en una mancha redonda. Se quitan las
    componentes con A <= `area_max` y alargamiento < `alargamiento_min`; las
    grandes (el árbol de vasos principal) se conservan siempre.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    mascara = np.asarray(mascara, dtype=bool)
    etiquetas, n = ndi.label(mascara, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return mascara.copy()
    area = np.bincount(etiquetas.ravel(), minlength=n + 1)
    largo = np.bincount(etiquetas.ravel(), weights=skeletonize(mascara).ravel(), minlength=n + 1)
    alargamiento = largo ** 2 / np.maximum(area, 1)
    quitar = (area <= area_max) & (alargamiento < alargamiento_min)
    quitar[0] = False
    return mascara & ~quitar[etiquetas]


# --- Histéresis sobre la probabilidad (etapa 5j, notebooks/05j_histeresis.ipynb) ---------

def _etiquetas_bajo(mapa, campo, bajo):
    from scipy import ndimage as ndi

    return ndi.label((mapa > bajo) & campo, structure=np.ones((3, 3), dtype=int))


def segmentar_histeresis_valor(mapa, campo, alto, bajo):
    """Umbral con histéresis con valores fijos del mapa (por ejemplo, la probabilidad de vaso).

    Es vaso lo que supera `alto` y, además, lo que supera `bajo` (bajo <= alto)
    y está conectado, con ocho vecinos, a algo que supera `alto`. Con bajo =
    alto es el valor fijo de siempre. Condurache y Mertins (2012) usan esta
    regla a la salida de un clasificador lineal para segmentar vasos de retina.
    """
    from scipy import ndimage as ndi

    if bajo > alto:
        raise ValueError("Se necesita bajo <= alto.")
    mapa = np.asarray(mapa, dtype=np.float32)
    campo = np.asarray(campo, dtype=bool)
    lab, n = _etiquetas_bajo(mapa, campo, bajo)
    if n == 0:
        return np.zeros_like(campo)
    maximo = np.concatenate([[-np.inf], ndi.maximum(mapa, lab, np.arange(1, n + 1))])
    return maximo[lab] > alto


def conteos_histeresis(mapa, campo, verdad, region, altos, bajos):
    """Cuentas para elegir los dos umbrales de la histéresis sin armar una máscara por combinación.

    La máscara es segmentar_histeresis_valor(mapa, campo, alto, bajo) y se
    cuenta dentro de `region` (el campo de visión de la base) contra `verdad`.
    Para cada `bajo` se etiquetan una vez las componentes de mapa > bajo: con
    un `alto` dado queda cada componente cuyo máximo supera `alto`, así que las
    cuentas salen de sumar las de esas componentes.

    Devuelve un diccionario con:
    - "tp", "marcados": arreglos (len(bajos), len(altos)) con los vasos
      marcados y el total marcado; NaN donde bajo > alto;
    - "tp_fijo", "marcados_fijo": lo mismo con el valor fijo mapa > alto;
    - "positivos", "total": vasos y píxeles de `region`.
    """
    from scipy import ndimage as ndi

    mapa = np.asarray(mapa, dtype=np.float32)
    campo, verdad, region = (np.asarray(a, dtype=bool) for a in (campo, verdad, region))
    altos, bajos = np.asarray(altos, float), np.asarray(bajos, float)
    tp = np.full((len(bajos), len(altos)), np.nan)
    marcados = np.full_like(tp, np.nan)
    es_vaso = region & verdad
    for b, bajo in enumerate(bajos):
        lab, n = _etiquetas_bajo(mapa, campo, bajo)
        validos = altos >= bajo
        if n == 0:
            tp[b, validos], marcados[b, validos] = 0, 0
            continue
        maximo = ndi.maximum(mapa, lab, np.arange(1, n + 1))
        n_vaso = np.bincount(lab[es_vaso], minlength=n + 1)[1:]
        n_region = np.bincount(lab[region], minlength=n + 1)[1:]
        orden = np.argsort(maximo)
        maximo, n_vaso, n_region = maximo[orden], n_vaso[orden], n_region[orden]
        # componentes con máximo > alto: las del final del orden
        desde = np.searchsorted(maximo, altos[validos], side="right")
        cola_vaso = np.concatenate([np.cumsum(n_vaso[::-1])[::-1], [0]])
        cola_region = np.concatenate([np.cumsum(n_region[::-1])[::-1], [0]])
        tp[b, validos], marcados[b, validos] = cola_vaso[desde], cola_region[desde]
    dentro = campo & region
    v_vaso, v_todo = np.sort(mapa[dentro & verdad]), np.sort(mapa[dentro])
    tp_fijo = len(v_vaso) - np.searchsorted(v_vaso, altos, side="right")
    marcados_fijo = len(v_todo) - np.searchsorted(v_todo, altos, side="right")
    return {"tp": tp, "marcados": marcados, "tp_fijo": tp_fijo.astype(float),
            "marcados_fijo": marcados_fijo.astype(float),
            "positivos": float(es_vaso.sum()), "total": float(region.sum())}


def matthews_conteos(tp, marcados, positivos, total):
    """Matthews a partir de las cuentas de conteos_histeresis (acepta arreglos)."""
    tp, marcados = np.asarray(tp, float), np.asarray(marcados, float)
    fp, fn = marcados - tp, positivos - tp
    tn = total - marcados - fn
    den = np.sqrt(marcados * positivos * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), np.nan)
