"""Constantes del proyecto: rutas, base de datos y partición."""
from pathlib import Path

# --- Rutas ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # no se versiona (ver .gitignore)
SPLIT_CSV = DATA_DIR / "split.csv"  # se versiona una sola vez
RESULTS_DIR = ROOT / "results"

# --- Base de datos -------------------------------------------------------
# High-Resolution Fundus Image Database (Budai et al., 2013), licencia
# Creative Commons 4.0 con atribución. all.zip trae imágenes, máscaras de
# vasos de los expertos y máscaras del campo de visión de las 45 imágenes.
HRF_URL = "https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
HRF_ZIP_BYTES = 76_317_613

# Sufijo del nombre de archivo -> grupo. 15 imágenes por grupo: 01_h ... 15_g.
GROUPS = {"h": "sano", "dr": "retinopatía diabética", "g": "glaucoma"}
N_PER_GROUP = 15
IMAGE_IDS = [f"{i:02d}_{g}" for g in GROUPS for i in range(1, N_PER_GROUP + 1)]

# Las máscaras de vasos son 0/255, salvo 11_h, 12_h y 13_h, que tienen entre
# 600 y 1.600 píxeles grises en los bordes. Es vaso lo que supera este valor.
MASK_THRESHOLD = 127

# Las imágenes ya vienen en .jpg con dos niveles de compresión. La media de la
# tabla de cuantización del brillo vale 2,94 en unas y 11,06 en otras; este
# valor las separa.
QUANT_SPLIT = 6.0

# La base no dice qué imágenes son de la misma persona. Dos fotos sacadas con
# a lo sumo estos minutos de diferencia se toman como de la misma persona
# (en la base son los dos ojos, sacados con 1 a 5 minutos de diferencia).
SAME_PERSON_MAX_GAP_MIN = 10

# --- Partición -----------------------------------------------------------
SEED = 2026

# --- Elección entre candidatas (esquema 5.3; plan, desvío 14.4) -----------
# Una candidata es no inferior si el límite inferior del intervalo al 95 %
# de (candidata − mejor) es >= −margen.
NONINF_MARGIN_AUC = 0.02
NONINF_MARGIN_MATTHEWS = 0.03   # también para el Dice de línea central
TEST_PER_GROUP = 5   # 15 imágenes de prueba, 30 de desarrollo

# --- Calibración (etapa 1, notebooks/01_contexto.ipynb) -------------------
# Diámetro del campo de visión: 3.261 a 3.263 píxeles en desarrollo.
FOV_DIAMETER_REF = 3262
# Límites entre vasos finos, medianos y gruesos: terciles del ancho sobre la
# línea central de las 30 máscaras de desarrollo. Fino <= 3,47; grueso > 9.
# Fijos para todas las imágenes; no se cambian más (plan, sección 5).
WIDTH_LIMITS_PX = (3.4721359549995796, 9.0)
WIDTH_LIMITS_REL = (WIDTH_LIMITS_PX[0] / FOV_DIAMETER_REF, WIDTH_LIMITS_PX[1] / FOV_DIAMETER_REF)
# Campo de visión: el algoritmo lo detecta solo con fov.detect_fov_robust.
# La candidata original (Otsu sobre el rojo) falló en 14_h (0,954); la robusta
# da Dice >= 0,9993 en las 30 de desarrollo y >= 0,996 en 9 variantes que
# simulan otras cámaras (notebooks/01b_campo_de_vision.ipynb; desvío 14.2).
USE_AUTO_FOV = True

# --- Mapas de color (etapa 2) ----------------------------------------------
# Mapa principal provisional: tercera componente principal del color por
# imagen, con vasos claros (notebooks/02_mapas_color.ipynb). Signo de las
# componentes: "vessels" (orientadas por la oscuridad local del verde), que
# no invierte ninguna imagen de desarrollo ni con la foto sobreexpuesta
# (notebooks/02b_signo_componentes.ipynb).
PCA_SIGN_RULE = "vessels"
# Elección del mapa (etapa 2d, notebooks/02d_mapas_corregidos.ipynb, después
# de la etapa 3): con el fondo corregido el verde es el mejor mapa (0,942) en
# las 5 vueltas, y ningún otro le suma 0,01 al combinarlo. Reemplaza la
# elección de la etapa 2c (tercera componente), que comparaba mapas crudos.
COLOR_MAIN_MAP = "Verde"
COLOR_REFERENCE_MAP = "Verde"
COLOR_EXTRA_MAPS: list[str] = []

# Las etapas 4 y 5 (notebooks/04_realce.ipynb, 05_segmentacion.ipynb) se corrieron
# con la tercera componente y el verde; el algoritmo final usa el verde.


# --- Realce de vasos y textura (etapa 4, notebooks/04_realce.ipynb) --------
# Escalas fijas (decisión de criterio, no se ajustan con los datos), sacadas
# de la etapa 1: anchos representativos de vasos finos a gruesos, hasta el
# percentil 99 del ancho (26 píxeles). Se guardan en proporción al diámetro
# del campo de visión para que se achiquen solas al reducir la resolución.
ENHANCE_WIDTHS_PX = (2, 4, 8, 16, 24)
ENHANCE_WIDTHS_REL = tuple(w / FOV_DIAMETER_REF for w in ENHANCE_WIDTHS_PX)
ENHANCE_SIGMA_MIN_PX = 0.5          # piso de σ al reducir la resolución
ENHANCE_MATCHED_LENGTH = 4.5        # largo del filtro adaptado, en múltiplos de σ
ENHANCE_GABOR_ELONGATION = 2.0      # largo de Gabor a lo largo del vaso, en múltiplos de σ
ENHANCE_LINE_LENGTH = 2.0           # largo del detector de líneas, en múltiplos del ancho
ENHANCE_TOPHAT_RADIUS_REL = 14 / FOV_DIAMETER_REF   # disco más ancho que el vaso más grueso
# Candidatas en orden de simplicidad (esquema 5.3): "sin filtro" es la más simple.
ENHANCE_FILTERS = ["sin filtro", "sombrero de copa", "detector de líneas",
                   "filtro adaptado", "Gabor", "Frangi"]
ENHANCE_FACTORS = (1, 2, 4)         # resolución original, mitad y cuarta parte
# Textura: matriz de coocurrencia con 8 niveles, distancia 1, cuatro
# direcciones y ventana de 15 píxeles a resolución original.
GLCM_LEVELS = 8
GLCM_WINDOW_REL = 15 / FOV_DIAMETER_REF

# --- Mejoramiento y restauración (etapa 3b) --------------------------------
# Se calcula a 1/STAGE3_FACTOR de la resolución y se evalúa a la original.
STAGE3_FACTOR = 2
# Ventana de la corrección del fondo: unas 3 veces el vaso más grueso (99 %
# de los anchos: 26 píxeles), en proporción al diámetro del campo de visión.
BACKGROUND_WINDOW_REL = 0.024
HOMOMORPHIC_SIGMA_REL = 0.01
CLAHE_CLIP = 0.01
CLAHE_KERNEL_FRAC = 0.125
WIENER_SIZE = 5
WIENER_DEBLUR_SIGMAS = (1.0, 2.0)       # píxeles de la resolución de trabajo
WIENER_DEBLUR_BALANCES = (0.01, 0.1)
# Cadena que mejor le funciona al verde en la etapa 3b (notebooks/03b_restauracion.ipynb): suavizado
# de bloques (foto original) → Wiener contra el ruido → corrección del fondo.
# Wiener contra el desenfoque se probó y no se adopta (empeora 0,035).
PREPROCESS_CHAIN = ("bloques", "wiener_ruido", "fondo")

# --- Evaluación rápida en desarrollo (src/evaluation.py) ------------------
# Fracción del fondo que entra en la muestra fija de evaluación (todos los
# vasos entran siempre). La evaluación final en prueba usa todos los píxeles.
EVAL_BG_FRACTION = 0.2

# --- Algoritmo final (etapa 5b, notebooks/05b_combinacion_filtros.ipynb) ---
# Adoptado por criterio (plan, desvío 14.5): verde a la mitad de resolución →
# corrección del fondo (gaussiana, σ = 0,02 del diámetro) → los filtros de la
# etapa 4, estandarizados dentro del campo detectado → regresión logística por
# píxel → umbral. Referencia clásica: el detector de líneas solo, con la misma
# regla de umbral. La regla (fracción o valor fijo) se cierra en la etapa 8.
SEGMENT_FACTOR = 2
SEGMENT_BACKGROUND_SIGMA_REL = 0.02
# Modelo de la corrección del fondo (preprocess.normalizar_fondo): "resta" (iluminación
# aditiva), "división" o "luminosidad y contraste" (iluminación multiplicativa).
# Etapa 5g (notebooks/05g_normalizacion.ipynb; plan, desvío 14.11): se adopta la división
# (Matthews +0,021 en validación cruzada, mejor en 29 de 30 imágenes). Es el modo con que se
# calculan los mapas para entrenar; segment_final usa el guardado con los pesos (load_background),
# así que los pesos congelados en congelado-test siguen usándose con la resta.
SEGMENT_BACKGROUND_MODE = "división"
# Etapa 5i (notebooks/05i_exudados.ipynb; desvío 14.11): antes de normalizar se rellenan los exudados
# (preprocess.rellenar_exudados; Annunziata et al., 2016). Matthews +0,0013, en el límite de la regla.
SEGMENT_BACKGROUND_MODE = SEGMENT_BACKGROUND_MODE + " + relleno de exudados"
# Etapa 5k (notebooks/05k_wiener.ipynb; desvío 14.11): antes de todo, Wiener local con el ruido estimado
# en cada imagen (preprocess.wiener_ruido_estimado). Sin ruido agregado −0,0008 de Matthews; con ruido 10, +0,064.
SEGMENT_BACKGROUND_MODE = SEGMENT_BACKGROUND_MODE + " + Wiener con ruido estimado"
SEGMENT_BASE_FEATURES = ["verde corregido", "sombrero de copa", "detector de líneas",
                         "filtro adaptado", "Gabor", "Frangi"]
# Ronda de mejoras (etapa 5e, notebooks/05e_ronda_mejoras.ipynb; plan, desvío 14.6):
# además de la mayor respuesta entre escalas, cada filtro orientado y Frangi en
# cada escala de la etapa 4 por separado (26 mapas). La textura no sumó.
SEGMENT_FEATURES = SEGMENT_BASE_FEATURES + [
    f"{f} {w}" for w in ENHANCE_WIDTHS_PX
    for f in ("detector de líneas", "filtro adaptado", "Gabor", "Frangi")]
# Mapas contra las lesiones (etapa 5h, notebooks/05h_lesiones.ipynb). Se calculan sobre el verde
# normalizado solo si están en la lista de mapas pedida.
# - Apertura por reconstrucción con segmentos (Zana y Klein, 2001): segmentos de 2 % y 4 % del
#   diámetro del campo (el artículo usa 15 píxeles en imágenes de unos 540 de diámetro, un 2,8 %).
# - Borde asimétrico (Zhang et al., 2010): derivada de la gaussiana en las escalas de la etapa 4,
#   con la media local en una ventana de 31/540 del diámetro, como en el artículo.
LESION_LINE_REL = (0.02, 0.04)
LESION_FDOG_WINDOW_REL = 31 / 540
LESION_FEATURES = ([f"apertura por reconstrucción {int(round(100 * r))} %" for r in LESION_LINE_REL]
                   + [f"borde asimétrico {w}" for w in ENHANCE_WIDTHS_PX])
# Etapa 5h (plan, desvío 14.11): se agregan las dos aperturas por reconstrucción (28 mapas); el borde
# asimétrico no sumó. Los pesos congelados en congelado-test son de los 26 mapas anteriores: el
# archivo de pesos guarda su lista de mapas y segment_final usa esa.
SEGMENT_FEATURES = SEGMENT_FEATURES + LESION_FEATURES[:2]
# Familias nuevas de mapas (etapa 5l, notebooks/05l_mapas_nuevos.ipynb), apuntadas a los vasos finos.
# Se calculan sobre el verde normalizado solo si están en la lista de mapas pedida.
# - Simetría de fase (Kovesi, 1999; Läthén et al., 2010): log-Gabor con longitud de onda 2 × ancho de
#   cada escala de la etapa 4 (mínimo 3 píxeles), 8 orientaciones; todas las escalas juntas y cada una.
# - B-COSFIRE (Azzopardi et al., 2015): las dos configuraciones del artículo para DRIVE (campo de unos
#   540 píxeles), llevadas al diámetro del campo, y la simétrica a un tercio de esa escala: la base
#   resuelve vasos unas tres veces más finos, en proporción al campo, que DRIVE.
# - Aperturas por caminos (Talbot y Appleton, 2007): caminos de 2 % y 4 % del diámetro, los mismos
#   largos que los segmentos rectos de la apertura por reconstrucción.
# - Autovalores del hessiano (Niemeijer et al., 2004; Staal et al., 2004) en las escalas de la etapa 4.
PHASE_WAVELENGTH_MIN_PX = 3.0
PHASE_FEATURES = ["simetría de fase"] + [f"simetría de fase {w}" for w in ENHANCE_WIDTHS_PX]
COSFIRE_DRIVE_DIAMETER = 540
COSFIRE_SYMMETRIC = {"sigma": 2.4, "rhos": (0, 2, 4, 6, 8), "sigma0": 3.0, "alpha": 0.7}
COSFIRE_ASYMMETRIC = {"sigma": 1.8, "rhos": tuple(range(0, 23, 2)), "sigma0": 2.0, "alpha": 0.1}
COSFIRE_FINE_FACTOR = 1 / 3
COSFIRE_FEATURES = ["B-COSFIRE simétrico", "B-COSFIRE simétrico fino", "B-COSFIRE asimétrico"]
PATH_LENGTH_REL = (0.02, 0.04)
PATH_FEATURES = [f"apertura por caminos {int(round(100 * r))} %" for r in PATH_LENGTH_REL]
HESSIAN_FEATURES = [f"hessiano λ{k} {w}" for w in ENHANCE_WIDTHS_PX for k in (1, 2)]
NEW_FEATURE_FAMILIES = {"fase": PHASE_FEATURES, "B-COSFIRE": COSFIRE_FEATURES,
                        "caminos": PATH_FEATURES, "hessiano": HESSIAN_FEATURES}
# Etapa 5l (desvío 14.11): se agrega B-COSFIRE por la regla (31 mapas), con un efecto mínimo
# (vasos finos +0,2 puntos). La simetría de fase sube la precisión (+1,8 puntos) pero baja la línea
# central; las aperturas por caminos bajan los finos; el hessiano no excluye el cero.
SEGMENT_FEATURES = SEGMENT_FEATURES + COSFIRE_FEATURES
# Etapa 5m (notebooks/05m_l1.ipynb; desvío 14.11): los pesos se ajustan con penalidad L1 sobre estos 31
# mapas (pipeline.fit_weights_l1: λ por validación cruzada por personas, regla de un error estándar).
# Quedan 13 a 15 mapas por vuelta; Matthews −0,0025 y vasos finos −1,8 puntos, dentro del margen. La lista
# final sale del ajuste con las 30 imágenes (versión 2) y se guarda con los pesos.
SEGMENT_FIT = "L1"
SEGMENT_REFERENCE_FILTER = "detector de líneas"
SEGMENT_FRACTION = 0.09        # elegida en las 5 vueltas
# Regla del umbral, cerrada en la etapa 8 (notebooks/08_robustez.ipynb) con el
# desvío 14.7: en robustez ninguna pierde; por concordancia clínica (etapa 7)
# gana el valor fijo (correlación intraclase 0,21 contra −0,02). El valor es el
# elegido con todo desarrollo y guardado con save_weights (models/pesos_combinacion.json).
SEGMENT_THRESHOLD_RULE = "valor fijo"
# Ronda para los vasos finos (etapa 5f; plan, desvío 14.8): entrenar con un tercio de
# los píxeles de vaso de cada clase de grosor en lugar de sorteados al azar. Adoptado:
# los vasos finos suben de 34 % a 40 % con Matthews igual. La resolución completa no ayudó.
SEGMENT_BALANCED_TRAINING = True

# --- Medidas clínicas (etapa 7, src/clinical.py) ----------------------------
# Todos en proporción al diámetro del campo (en píxeles a resolución original:
# el valor × 3262). La poda de ramitas y el largo mínimo de los tramos se
# fijaron en notebooks/07a_medidas_expertos.ipynb solo con los expertos: el
# valor más chico cuya tortuosidad por imagen casi no cambia al duplicarlo.
CLINICAL_HOLE_REL = 40 / FOV_DIAMETER_REF        # se rellenan los agujeros de menos de este lado al cuadrado
CLINICAL_STEP_REL = 6 / FOV_DIAMETER_REF         # un punto cada tantos píxeles para medir el largo de un tramo
CLINICAL_SPUR_REL = 5 / FOV_DIAMETER_REF         # se podan las ramitas más cortas que esto
CLINICAL_MIN_SEGMENT_REL = 25 / FOV_DIAMETER_REF  # los tramos más cortos no entran en la tortuosidad

# --- Robustez (etapa 8, notebooks/08_robustez.ipynb) ------------------------
# Degradaciones agregadas a la foto de desarrollo, en niveles crecientes
# (decisión de criterio, fijada antes de correr el desenfoque y el ruido).
ROBUST_JPEG_QUALITIES = (90, 70, 50, 30)
# Desenfoque gaussiano: σ de 1, 2 y 4 píxeles a resolución original (el límite
# de los vasos finos es 3,47 píxeles), en proporción al diámetro.
ROBUST_BLUR_SIGMA_REL = tuple(s / FOV_DIAMETER_REF for s in (1, 2, 4))
# Ruido gaussiano: desvío de 5, 10 y 20 niveles de gris (de 255), en cada canal.
ROBUST_NOISE_SIGMAS = (5, 10, 20)
# Restauración de la foto (src/degrade.restore_photo), una por degradación:
# JPEG → suavizado de bloques; ruido → Wiener local; desenfoque → deconvolución
# de Wiener con el desenfoque conocido. La ventana del Wiener local cubre la
# misma porción de retina que en la etapa 3b (5 píxeles a 1/2 de resolución).
ROBUST_WIENER_SIZE = WIENER_SIZE * STAGE3_FACTOR + 1
ROBUST_DEBLUR_BALANCE = 0.1

# --- Robustez, parte de imagen (etapa 8; temas obligatorios de compresión y reconstrucción) ---
# Calidad de la foto (no del algoritmo): tamaño de archivo, PSNR y SSIM del verde
# contra la original, para calidades .jpg de 90 a 10.
IMAGE_JPEG_QUALITIES = (90, 70, 50, 30, 10)
# Restauración clásica con degradación conocida, sobre el verde: desenfoque de
# σ = 1, 2 y 4 píxeles, sin ruido y con ruido de σ = 5 niveles.
RESTORE_BLUR_PX = (1, 2, 4)
RESTORE_NOISE_SIGMAS = (0, 5)
# Parámetro de cada método: se elige el de mayor PSNR medio en el interior del campo
# en 5 imágenes de desarrollo, para cada degradación, y se informa en las otras 25.
RESTORE_PARAMS = {"filtro inverso": (0.01, 0.05, 0.2),        # umbral de |H|
                  "Wiener": (0.001, 0.01, 0.1),               # cociente ruido / señal K
                  "mínimos cuadrados restringidos": (0.001, 0.01, 0.1)}   # γ