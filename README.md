# Segmentación de puentes con SPG, WSPG y SOL

Proyecto de tesis para preparar nubes de puntos y evaluar su segmentación
semántica. Python 3.11 de 64 bits.

## Estado

Funcionan la revisión de PLY, la reducción por celdas, el cálculo de geometría y
la partición en superpuntos, la preparación de entrada, la construcción del grafo
SPG y la herramienta de evaluación de pureza con etiquetas de referencia.
El completado híbrido AdaPoinTr + ajuste geométrico está implementado y su
resultado visual fue aceptado. Hay una anotación asistida de Lurín y una medición
preliminar de pureza sobre el 74,08 % de sus puntos observados. La referencia
incorpora la información de ingeniería aportada para este caso. El flujo de M2 añade
contexto estructural automático y una corrección residual aprendida a WSPG,
con una extensión SOL de relaciones y continuidad. Se entrena con cinco
puentes reales y 48 escenas sintéticas externas; las épocas se eligen con
dos puentes reales de validación. Lurín no se usa para entrenar.

La comparación actual **en Lurín**, contra 2,607,868 puntos de referencia
asistida, obtiene 48.07 % de mIoU con WSPG y 52.44 % con WSPG + SOL. La segunda
variante alcanza IoU de 89.05 % en columnas, 56.56 % en vigas/cabezales,
51.66 % en tablero y 91.11 % en barandas. No baja ninguna de las siete IoU
evaluadas frente al WSPG pareado ni a las salidas anteriores utilizadas como
control. M2 queda cerrado con esta versión aceptada; accesos y separador se
conservan registrados como limitaciones del resultado.

Se corrigió la correspondencia de `superstructure` de SemanticBridge: es una
anotación amplia, no una viga exacta. Se conserva como unión viga/tablero sin
alterar la fuente. Los antiguos porcentajes de validación externa no son
resultados de Lurín y los de vigas no son comparables con la taxonomía corregida.

La referencia de Lurín **sí se usa para aceptar/elegir candidatas después de
entrenar**, por lo que estas cifras son desarrollo del caso, no una prueba
independiente de generalización. Cada PLY conserva los 3,629,106 puntos de M1,
incluido el completado. Véase [M2/README.md](M2/README.md) para ejecutar el flujo,
consultar la comparación y conocer sus límites. Los colores de los superpuntos
no son clases de componentes.

```text
M1/
  nube_de_puntos_3D/                    # fused.ply: lo añade el usuario
  limpieza_y_preparacion/
    procesamiento_manual/              # puente_orientacion_revision.ply
    nube.py
    reduccion.py
    limpieza.py
    resultados/                        # reducción y preparación para completado
  AdaPoinTr/                           # método híbrido, pesos y resultado aprobado
  SPG/
    geometria.py
    particion.py
    preparacion.py
    grafo.py
    pureza.py
    resultados/                        # única ejecución: entrada, geometría, superpuntos, grafo, entrada_red
  clases/                              # anotación asistida específica de Lurín
    configs/                           # clases, regiones y alcance de la referencia
    datos/
    resultados/
M2/
  WSPG/                                # PointNet + ECC/GRU y corrección residual final
  SOL/                                 # relaciones estructurales y continuidad
  configs/                             # corpus y control de aceptación
  datos/                               # único corpus final
  modelos/                             # WSPG, WSPG_SOL e inicialización necesaria
  resultados/                          # PLY, métricas y figuras finales
  tests/                               # pruebas del método conservado
main.py                                # limpieza y herramientas individuales
requirements.txt
tests/
```

La reducción de densidad pertenece a limpieza y preparación; SPG se ocupa de
la partición y del grafo. El módulo `clases` produce propuestas por punto y una
referencia asistida preliminar a partir de regiones contrastadas con fotogramas
y vistas de la nube. No es un clasificador entrenado ni forma parte de WSPG+SOL.
No usa los superpuntos para decidir etiquetas y excluye los puntos estimados de
la evaluación principal. La referencia incorpora información de ingeniería y
conserva las regiones dudosas sin evaluar; su alcance está declarado en los informes.

El SPG vigente se calculó desde el puente completado aprobado y conserva
3.629.106 puntos (3.520.409 observados y 108.697 estimados). No repite la reducción
de densidad de limpieza. Los resultados SPG anteriores se descartaron. La nube recortada manualmente está en
`M1/limpieza_y_preparacion/procesamiento_manual/puente_orientacion_revision.ply`.
La reconstrucción original está en `M1/nube_de_puntos_3D/fused.ply`.

## Instalación y uso

Con el entorno virtual del proyecto activo:

```powershell
python -m pip install -r requirements.txt
python main.py revisar "M1/nube_de_puntos_3D/fused.ply"
python main.py reducir M1/limpieza_y_preparacion/procesamiento_manual/puente_orientacion_revision.ply --celda 0.03
python main.py limpiar
# Después de aceptar la preparación:
.\M1\AdaPoinTr\scripts\completar.ps1
# Después de aceptar el completado:
python -m M1.SPG
python -m M1.SPG.verificar
```

Los comandos se ejecutan desde la raíz del proyecto, en orden y con revisión
entre fases. El script AdaPoinTr usa su entorno CUDA separado. `--help` muestra
las opciones. Si SPG ya terminó, `python -m M1.SPG --reanudar` verifica sus etapas
guardadas sin recalcularlas ni mezclar entradas diferentes.
El archivo de entrada debe tener previamente la escala y orientación elegidas.
El formato PLY no certifica que sus coordenadas representen metros.

## Limpieza y preparación antes del completado

Con la reducción existente y su nube manual correspondiente:

```powershell
python main.py limpiar
```

Guarda en `M1/limpieza_y_preparacion/resultados/preparado/`:

- `puente_preparado.ply`: XYZ, RGB e identificador `reduccion_id`; conserva el
  marco local y los colores de los centroides que superan el filtro.
- `puntos_retirados.ply`: candidatos descartados en rojo, para superponerlos y
  revisar qué se ha retirado antes de aceptar la preparación.
- `preparacion.npz`: coordenadas, colores, correspondencias y parámetros para
  reproducir el proceso; contiene también `xyz_normalizado` y su inversa.
- `revision_limpieza.png`: comparación visual de esta ejecución.

El filtrado combina dos condiciones: distancia media a 20 vecinos mayor que
media global + 3 desviaciones, y menos de 3 vecinos dentro de un radio de cuatro
vóxeles (0.12 en esta ejecución). Las cadenas localmente lineales se protegen.
Estos valores y la protección son decisiones de implementación propias; la tesis
propone filtrado y normalización, sin fijar aquí estos parámetros específicos.
El criterio estadístico y el de radio se describen en la
[documentación de Open3D](https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud_outlier_removal.html).
La implementación utiliza NumPy/SciPy, sin instalar Open3D.

La normalización usa centro de caja y extensión máxima con una sola escala:
`xyz_normalizado=(xyz-centro)/escala`. Se puede revertir con
`xyz=xyz_normalizado*escala+centro`. No cambia las coordenadas del PLY ni las
proporciones geométricas. La escala física sigue siendo estimada; una superficie
dominante casi horizontal permite comprobar coherencia de Z, no medir gravedad.
El método futuro de completado puede necesitar otra preparación de entrada.

Se verifican las huellas de las entradas, se conserva la relación con los puntos
manuales retenidos y se rechaza sobrescribir una preparación existente. El filtro
es conservador: no certifica que toda superficie restante sea correcta. Revisar
la nube y los descartes antes de pasar al completado. No realiza completado ni etiquetado.

SPG recibe únicamente el resultado aprobado del completado. Su archivo de
entrada de cálculo es `M1/SPG/resultados/entrada/nube_completada.npz`, con una
correspondencia identidad: todos los puntos del PLY entran sin otra reducción.
Conserva XYZ, RGB, identificadores y `origen` (0 observado, 1 estimado). Las formas
propuestas por el completado no se utilizan como etiquetas de referencia.

## Qué procede del artículo y qué se ha adaptado

La revisión del artículo WSPG de Yang et al. (2022), DOI
https://doi.org/10.1016/j.autcon.2022.104519, establece:

- Sección 3.3.1, página 6: reducción por centroides de vóxeles de 0.03 m.
- Sección 3.3.2, página 7: grafo de 10 vecinos y cinco atributos de partición:
  linealidad, planaridad, dispersión, verticalidad y elevación.
- Secciones 4.1 y 4.2.2.1, páginas 10 y 13: comparación de regularizaciones
  0.01, 0.03 y 0.05; seleccionan 0.03 en sus experimentos.
- Sección 3.3.3, página 8: entrada de 128 puntos con 8 atributos y tratamiento
  de superpuntos de menos de 25 puntos. El muestreo ya está preparado; el tratamiento en la red está pendiente.

La implementación NO se presenta como reproducción exacta del artículo:

1. WSPG enumera elevación, pero no proporciona su fórmula operativa ni su
   normalización. La referencia [36] (Guinard y Landrieu, 2017) utiliza altura
   respecto a un modelo de carretera. Sin ese terreno, elegimos explícitamente
   `z-minmax`: `(z-z_min)/(z_max-z_min)` sobre toda la nube reducida. Una nube
   horizontal recibe ceros. Es adaptación propia, sensible a puntos extremos
   y al eje Z elegido; no es altura sobre terreno ni cota geográfica.
2. Los otros cuatro descriptores siguen `compute_geof` del código SPG: raíces
   de autovalores para dimensionalidad y autovectores absolutos ponderados por
   autovalores para verticalidad. Los 45 vecinos de geometría provienen de ese
   código; no están confirmados como configuración explícita de WSPG 2022.
3. El peso adicional de verticalidad es 1 (sin duplicarlo); el artículo WSPG no
   especifica el multiplicador 2 usado por el código SPG de referencia.
4. El solucionador es `cut-pursuit-py==1.0.12`, no `libcp` original. Los valores
   de regularización no garantizan resultados numéricos equivalentes entre
   implementaciones. Sus iteraciones máximas son 20 y la tolerancia 1e-4.

La fórmula de pesos de arista se hereda del código SPG:
`regularizacion/(1+distancia/media_distancias)`. Cada entrada k-NN aporta un
peso, incluyendo reciprocidad. Nuestro código incorpora la regularización en
el vector de pesos porque el envoltorio de esa versión no lo hace mediante
`reg_strength` cuando recibe pesos por arista.

Cada centroide tiene peso unitario durante la partición. No se eliminan puntos,
no se impone radio máximo de vecindad ni tamaño mínimo de grupo. Se verifica la
conectividad de los superpuntos. El comando utiliza un hilo OpenMP; la biblioteca
no expone semilla del solucionador, por lo que no garantiza particiones idénticas.
Solo los colores tienen una semilla fija (42).

## Resultados y trazabilidad

La reducción, la geometría y la partición conservan sus archivos de cálculo y visualización en sus respectivas carpetas:

| Etapa | Archivo de cálculo | Archivo para CloudCompare |
|---|---|---|
| Entrada SPG, sin reducción adicional | `M1/SPG/resultados/entrada/nube_completada.npz` | PLY de completado aprobado |
| Geometría | `M1/SPG/resultados/geometria/geometria.npz` | `geometria.ply` |
| Partición | `M1/SPG/resultados/superpuntos/superpuntos.npz` | `superpuntos.ply` |

Los NPZ contienen parámetros, versiones y huellas SHA-256. La reducción conserva
la correspondencia de cada punto original con su centroide; la partición conserva
su asignación a un superpunto. No se modifican las entradas ni se sobrescriben
salidas automáticamente. La salida SPG vigente está directamente en
`M1/SPG/resultados/`: no hay variantes paralelas sin completado. Las subcarpetas
`entrada`, `geometria`, `superpuntos`, `grafo` y `entrada_red` son etapas de una
misma ejecución. `proceso.json` enlaza sus huellas y `verificacion.json` confirma
la conservación de coordenadas, colores, identificadores y procedencia.

La geometría se calcula por bloques de 10000 puntos en CPU. Los bloques limitan
memoria sin cambiar el vecindario. Se almacena el radio de los 45 vecinos para
inspeccionar zonas dispersas y se marcan los vecindarios sin extensión.
La pureza de los superpuntos requiere etiquetas de referencia; una imagen por
colores no la acredita. Los parámetros del artículo tampoco son óptimos
comprobados para nuestros puentes.

Datos, resultados, pesos de redes, entorno virtual y configuración de PyCharm
quedan fuera de Git mediante `.gitignore`.

## Comprobaciones y fuentes

```powershell
python -m unittest discover -s tests -v
```

- WSPG: https://doi.org/10.1016/j.autcon.2022.104519
- SPG: https://github.com/loicland/superpoint_graph
- Descriptores y elevación de referencia: https://doi.org/10.5194/isprs-archives-XLII-1-W1-151-2017
- Solucionador: https://github.com/truebelief/cut_pursuit_py

Los métodos y las bibliotecas citados no se presentan como invenciones propias.


## Completado AdaPoinTr

Entrada: `M1/limpieza_y_preparacion/resultados/preparado/preparacion.npz`.
Salida aprobada: `M1/AdaPoinTr/resultados/completado/puente_completado.ply`.

Es un método híbrido: AdaPoinTr oficial propone puntos y un ajuste geométrico
automático reconstruye superficies compatibles con los fragmentos visibles.
Compara columnas circulares, rectangulares, elípticas y octagonales, perfiles de
cabezal y terminaciones. La forma global procede principalmente del ajuste;
no es una salida exclusivamente neuronal. Su alcance actual es un conjunto
dominante de apoyos alineados y cabezal, con incertidumbre cuando falta evidencia.
La generalización a otros puentes reales sigue pendiente.

Los pesos están en `M1/AdaPoinTr/modelos/completado_contextual/mejor.pt`; la
inicialización necesaria para reproducir el entrenamiento está en
`modelos/inicializacion/`. Las pruebas y registros están en `evaluacion/`.
Los pesos y el entorno nativo no se incluyen en Git y deben proporcionarse aparte.
La inferencia requiere el entorno `M1/AdaPoinTr/instalacion/venv` con CUDA,
PointNet++ y Chamfer compilados; el SPG usa el entorno principal del proyecto.

```powershell
.\M1\AdaPoinTr\scripts\completar.ps1
```

El PLY conserva los puntos observados y añade estimados en naranja; `origen`
distingue ambos. Los identificadores de formas propuestas no son clases
semánticas verificadas. Las superficies ausentes siguen siendo aproximaciones.

## Preparar entrada de superpuntos

```powershell
python main.py preparar M1/SPG/resultados/entrada/nube_completada.npz M1/SPG/resultados/geometria/geometria.npz M1/SPG/resultados/superpuntos/superpuntos.npz
```

Escribe un único `M1/SPG/resultados/entrada_red/entrada_superpuntos.npz`: 128 puntos por grupo,
XYZ local y cinco atributos geométricos (128 x 8). Muestrea sin reemplazo para
grupos de al menos 128 puntos; usa reemplazo en los menores. Semilla 42.
Se conserva el índice reducido de cada muestra, el centro, la escala y el tamaño
del grupo. XYZ se centra y divide por la extensión máxima del grupo entero:
normalización propia explícita, no fórmula confirmada de WSPG. Los cinco
atributos quedan sin ponderar. Los grupos menores de 25 puntos se conservan con
`embedding_habilitado=False`; la futura red deberá asignarles embedding cero.
El muestreo no constituye entrenamiento, clasificación ni construcción del grafo.
Las correspondencias originales de superpuntos.npz permanecen intactas.


## Grafo de superpuntos y atributos

```powershell
python main.py grafo M1/SPG/resultados/entrada/nube_completada.npz M1/SPG/resultados/superpuntos/superpuntos.npz
```

Crea `grafo/grafo.npz` e `informe.json`. La adyacencia procede de Delaunay 3D
(dual de Voronoi) sobre TODOS los puntos de entrada del SPG: une grupos cuyos puntos son
vecinos en esa estructura. No utiliza k-NN de centroides ni el muestreo de 128
puntos. Conserva grupos pequeños y registra identificadores compatibles con la
partición y la entrada de red. Cada conexión se guarda en ambas direcciones.
Delaunay puede tardar y necesitar varios GB de RAM; no es un entrenamiento GPU.
No se agregan puntos ni se modifica la partición. La entrada coplanar/degenerada
se rechaza; no se perturba con ruido para forzar la triangulación.

Perfil de 13 atributos inspirado en SPG: media del offset (3), desviación estándar
poblacional del offset (3), log-cocientes de longitud/superficie/volumen/conteo (4)
y diferencia de centroides (3). Offset = origen - destino. Los descriptores de
forma propios son lambda1, lambda1*lambda2 y lambda1*lambda2*lambda3 de la
covarianza muestral; en singletons son cero. Se estabiliza cada log con 1e-10.
La referencia usa operaciones y casos degenerados algo diferentes; no afirmamos
identidad numérica con su implementación ni una reproducción exacta de WSPG.
No se estandarizan atributos todavía: el ajuste del scaler deberá usar solo
entrenamiento cuando se definan las divisiones. Se guardan los conteos de contactos
puntuales y formas por nodo para inspección y futuras variantes.

No se impone distancia máxima; el informe muestra percentiles y máximo de los
contactos, grupos aislados y componentes conexas. Voronoi puede conectar regiones
a través de vacíos y no significa contacto físico entre elementos. Revisar el
informe antes de decidir cualquier corte de distancia. Incluye hashes de entradas,
versiones y nombres de atributos.

Fuentes: WSPG 2022 §3.3.3;
https://github.com/loicland/superpoint_graph/blob/master/partition/graphs.py y
https://github.com/loicland/superpoint_graph/blob/master/learning/main.py .

## Clases de Lurín y pureza preliminar

```powershell
python main.py clases
```

En `M1/clases/resultados/puente_clases.ply` se muestran propuestas: 0 columna/pilar
(rojo), 1 viga/cabezal (naranja), 2 tablero (azul), 3 baranda lateral (morado),
4 pista de accesos (verde), 5 separador central (cian), 6 superficie inferior
suelo/base (marrón), -1 sin asignar (gris). La categoría inferior no identifica el
material ni confirma cimentación. `clase_propuesta` no es una verdad de referencia.
Se conserva el índice por punto para compararlo con SPG; no se alteran XYZ ni
los archivos de entrada. Las regiones están vinculadas por hash a esta nube de
Lurín y no deben aplicarse como clasificador general a cualquier puente.

`M1/clases/datos/referencia_preliminar.ply` contiene 2.607.868 observaciones
anotadas en regiones reconocibles (74,08 % de los puntos observados). Se excluyen
los 108.697 estimados por completado. La selección conservadora deja fronteras
dudosas sin evaluar y puede elevar artificialmente la pureza. Su resultado está
marcado como preliminar en `M1/clases/resultados/pureza_preliminar/pureza.json`;
no es la exactitud de WSPG ni la pureza comprobada de todo Lurín. Las fuentes
visuales y reglas están en `M1/clases/configs/lurin.json`. Incorpora las correcciones
del usuario sobre accesos, franja de tablero, separador y superficie inferior.
La pureza preliminar actual es 97,850 %. Cambiaron la taxonomía y la cobertura;
no debe compararse como mejora o empeoramiento del SPG, que no se recalculó.
Tras modificar la configuración explícitamente, la actualización se ejecuta con
`python -m M1.clases --actualizar`.

Para una referencia posteriormente revisada, se conserva la herramienta general:

```powershell
python main.py pureza-plantilla M1/SPG/resultados/entrada/nube_completada.npz
```

Crea `M1/clases/datos/para_etiquetar.ply` con XYZ, RGB, `punto_id` estable y `clase=-1`.
Los colores son los del levantamiento, no los colores aleatorios de superpuntos.
Definir una tabla de clases antes de etiquetar. Marcar regiones por componentes
físicos, incluyendo sus fronteras; NO asignar una clase a cada superpunto porque
eso haría circular la evaluación. Mantener `punto_id` como campo escalar al
importar y guardar en CloudCompare; no transformar coordenadas, reducir densidad
ni mezclar recortes solapados. Guardar un PLY etiquetado aparte. Desconocido=-1;
las clases de referencia son enteros >=0 (0 sí es una clase). Se admite un subconjunto
etiquetado, pero su resultado no se extrapola al puente completo.

```powershell
python main.py pureza M1/SPG/resultados/entrada/nube_completada.npz M1/SPG/resultados/superpuntos/superpuntos.npz M1/clases/datos/etiquetas_revisadas.ply
```

El campo de referencia se llama `clase` por defecto; puede indicarse con `--campo`.
Verifica IDs únicos, coordenadas correspondientes y firma de la partición; no
supone conservar el orden del PLY. Genera `pureza.json` y `pureza.npz` con
histogramas por grupo/clase, cobertura, etiqueta mayoritaria y pureza por grupo.
Pureza ponderada = suma de los conteos mayoritarios de los grupos / puntos
etiquetados. También se informa la media entre grupos evaluados. Son métricas
sobre los puntos observados de entrada SPG con peso uniforme; las estimaciones se excluyen.
Los puntos sin etiqueta se excluyen; grupos no evaluados reciben NaN. Con cero
etiquetas se rechaza la evaluación. Una pureza alta puede favorecer particiones
muy finas; debe interpretarse junto a tamaños de grupos y cobertura. No es
exactitud de clasificación de una red. Los PLY marcados como propuestas automáticas
se rechazan como referencia; las anotaciones asistidas conservan la advertencia
de evaluación preliminar.
