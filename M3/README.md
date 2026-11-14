# M3 · Aumento sintético para WSPG + SOL

La prueba seleccionada compara el corpus base con dos lotes de 12 escenas y
22 000 puntos por escena. El uniforme usa escenas generales con las diez clases
presentes y distinta cantidad de puntos por clase. El dirigido refuerza
baranda–tablero, señal–entorno y estribo–terreno con cuatro escenas por situación.
Los lotes usan semillas distintas. Ambos reemplazan una de las cuatro escenas
sintéticas del plan de entrenamiento por época.

| Variante | mIoU en Lurín | mIoU externa |
|---|---:|---:|
| Sin aumento adicional | 52.21 % | 76.58 % |
| Uniforme | 53.68 % | 80.09 % |
| Dirigido | 55.93 % | 78.90 % |

`evaluacion/` conserva la comparación visual, el diagnóstico, las métricas,
los manifiestos y el protocolo de esta prueba. La referencia asistida de Lurín
se aplica a los puntos observados. Los puentes externos participaron en el
diagnóstico y la selección de pesos.

## Carpetas por etapa

- `aumento_sintetico/` genera los lotes uniforme y dirigido.
- `entrenamiento/` ejecuta la comparación controlada.
- `evaluacion/` reúne el código de visualización, las métricas y las predicciones locales.

## Reproducir

Ejecutar desde la raíz con el entorno de M2. Usar carpetas de salida con
permiso de escritura y espacio suficiente, preferiblemente en otro disco.

```powershell
python -m M3 generar-lotes --salida "D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data\datos_sinteticos"
python -m M3 entrenar-comparar --datos "D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data\datos_sinteticos" --salida "D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data\experimento"
python -m M3 visualizar-lurin --resultados "D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data\experimento\resultados"
```

La visualización requiere Pillow. El entrenamiento usa la inicialización, el
corpus y el SPG de M1/M2. Guarda predicciones compactas por defecto. Agregar
`--exportar-ply` al comando de entrenamiento para generar los tres PLY completos.

## Almacenamiento actual

Los lotes y resultados completos están en
`D:\USUARIO\Documentos\Nube de puntos tesis\M3_Data`.
Los comandos usan esa carpeta por defecto. Se puede cambiar mediante
`M3_DATA_ROOT` o los argumentos de cada comando.

Para visualizar la prueba guardada:

```powershell
python -m M3 visualizar-lurin
```
