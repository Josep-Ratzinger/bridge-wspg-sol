# M2: WSPG y WSPG + SOL estructural

Esta carpeta contiene únicamente la versión final aceptada de M2. La rama
predeterminada es **WSPG_SOL**; WSPG se conserva como comparación del mismo
método. Los ensayos anteriores no son dependencias de esta versión.

## Uso

Desde la raíz del proyecto, con Python 3.11 y el entorno del proyecto:

```powershell
python -m pip install -r requirements.txt -r M2/requirements.txt
python -m M2 aplicar
```

La entrada predeterminada es el SPG completo de `M1/SPG/resultados`. Para otro
SPG, conservar su estructura de archivos y la gravedad en Z:

```powershell
python -m M2 aplicar --spg RUTA_SPG --salida RUTA_SALIDA
```

Se necesitan `superpuntos/superpuntos.ply`, su partición NPZ, el grafo NPZ y la
entrada de red NPZ de M1. Los colores aleatorios del PLY no sustituyen esos
atributos. La inferencia no consulta etiquetas, no recorta la nube y conserva
las coordenadas, los identificadores y el origen observado/estimado.

Para repetir la comparación de Lurín:

```powershell
python -m M2 aplicar --modelo WSPG
python -m M2 aplicar --modelo WSPG_SOL
python -m M2 evaluar
python -m M2 visualizar
```

Se generan `resultados/WSPG/puente_etiquetado.ply`, la salida equivalente de
`WSPG_SOL`, `evaluacion.json`, `comparacion.png` y `barandas.png`. Las figuras
muestran anotación asistida previa, WSPG y WSPG + SOL. Los recortes son solo
ventanas de visualización; ambos PLY contienen todos los puntos.

## Entrenamiento y datos

`datos/corpus` conserva exactamente las siete escenas reales y las 56 escenas
sintéticas usadas por esta versión: cinco reales y 48 sintéticas train, dos
reales y ocho sintéticas val. El manifiesto contiene firmas, clases y fuentes.
Los atributos reales incluyen consultas geométricas en el soporte original
denso. La taxonomía es única: `superstructure` permite viga o tablero; no se
convierte en viga exacta. No hay etiquetas de Lurín en el entrenamiento.

```powershell
python -m M2 preparar --fuentes "D:\USUARIO\Documentos\Nube de puntos tesis\P2C_Data"
python -m M2 entrenar
```

La preparación verifica y reutiliza los datos ya guardados. Para repetir el
entrenamiento sin sobrescribir los pesos aceptados, indicar `--salida` con un
directorio nuevo. `modelos/inicializacion_wspg.pt` es la base congelada usada
por las dos ramas finales, no otro ensayo. Los checkpoints finales contienen
también esa base: aplicar el modelo no requiere los datasets ni inicializaciones.

La arquitectura conserva PointNet/ECC y añade una corrección residual aprendida
con 20 atributos estructurales calculados automáticamente desde XYZ y grupos:
superficie vial, recorrido/bordes, extremos y persistencia vertical de apoyos.
No se fija el número ni la sección de las columnas. El ámbito es un tablero
vial continuo. Ambas ramas comparten arquitectura, datos, semilla 614,
normalización y calibración; SOL usa coeficiente 0.5 y la otra rama 0.

SOL es la adaptación implementada de relaciones de apoyo/altura y continuidad
lateral. No es una reproducción literal del paper, ni un recoloreado por reglas
después de inferir. Una cara vertical de viga no obliga a clasificarla como
columna, y una conexión no observada no se penaliza como contradicción segura.

## Resultado aceptado en Lurín

| IoU | WSPG | WSPG + SOL |
|---|---:|---:|
| Columnas | 88.67 % | 89.05 % |
| Vigas/cabezales | 34.67 % | 56.56 % |
| Tablero | 51.05 % | 51.66 % |
| Barandas | 87.82 % | 91.11 % |
| Accesos | 1.46 % | 3.05 % |
| Separador central | 0.00 % | 0.00 % |
| Suelo/base | 72.84 % | 75.67 % |
| mIoU | 48.07 % | 52.44 % |

La evaluación usa los mismos 2,607,868 puntos observados con referencia
asistida e información de ingeniería (74.08 % de los observados). Los
estimados por completado y los puntos sin referencia se excluyen de las
métricas, pero permanecen en los 3,629,106 puntos exportados. IoU penaliza
omisiones y falsos positivos; mIoU no es el porcentaje de todo el puente
correctamente etiquetado.

Lurín se usó para aceptar la versión después del entrenamiento, no para
gradientes ni selección de épocas. Las cifras son evaluación de desarrollo
del caso, no un test independiente. Accesos y separador quedan registrados
como limitaciones del resultado cerrado y no se eliminan para subir la media.

## Estructura

```text
M2/
  WSPG/             # modelo y corrección residual
  SOL/              # relaciones estructurales de la versión final
  configs/          # especificación del corpus y control de aceptación
  datos/            # corpus final y fuente Seokdap conservada
  modelos/          # WSPG, WSPG_SOL e inicialización de entrenamiento
  resultados/       # PLY, predicciones, evaluación y figuras finales
  tests/            # pruebas del método conservado
```

Los datos y pesos están excluidos de Git; deben distribuirse por separado si
se quiere aplicar los pesos ya entrenados. El código y el README son únicos.

## Fuentes

- [SemanticBridge y taxonomía oficial](https://github.com/mvg-inatech/3d_bridge_segmentation):
  respetar atribución y licencia CC BY NC 4.0 indicada por la fuente.
- [UAV LiDAR de puentes](https://zenodo.org/records/18531104): uso académico.
- [FHWA: tipologías de apoyos](https://www.fhwa.dot.gov/bridge/Lrfd/us_ds8.cfm)
  y [NYSDOT: detalles de pilares](https://www.dot.ny.gov/main/business-center/engineering/cadd-info/drawings/bridge-detail-sheets/pr-piers).
- [BrPCD](https://doi.org/10.1111/mice.13384): referencia de variación de formas.
  Las dimensiones sintéticas son propias, no planos calculados ni medidas de Lurín.
