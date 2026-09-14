# Laboratorio 2 - Complejidad y búsqueda de hiperparámetros

ISIS2611 - Aprendizaje de Máquina · Caso AlpesPlanck

*Shaiel Jiménez - 202323846*
*Juan Esteban Triviño - 202315338*

## Estructura del repositorio

```
.
├── data/
│   ├── raw/                 # Datos crudos del Laboratorio 1 (Datos Lab 1.csv)
│   └── processed/           # train.csv / test.csv generados por src/data_prep.py (no versionados)
├── src/
│   └── data_prep.py         # Limpieza y split reutilizados del Laboratorio 1 (sin fuga de datos)
├── sm.jimenezp1_j.trivinon.ipynb   # Notebook principal del laboratorio
├── requirements.txt
└── README.md
```

## Por qué existe `src/data_prep.py`

El enunciado del laboratorio indica explícitamente que el énfasis **no** está en la
exploración ni en la limpieza de datos, sino en el modelado, la validación y el
análisis de desempeño. Por eso la limpieza (corrección de categorías, errores de
escala, duplicados, outliers imposibles e imputación sin fuga de datos) se
reutiliza tal cual del Laboratorio 1 y se encapsula en un módulo aparte, en vez de
repetirla celda por celda en el notebook. El notebook solo llama a
`cargar_datos_limpios()` y arranca directamente en la Actividad 1.

## Cómo correrlo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (opcional) genera data/processed/train.csv y test.csv para inspección rápida
python -m src.data_prep

jupyter notebook sm.jimenezp1_j.trivinon.ipynb
```

## Estado del notebook

El notebook está montado como **esqueleto**: todas las secciones, pipelines y
preguntas de análisis del enunciado están estructuradas con celdas `TODO`
listas para completar. La carga y limpieza de datos (`src/data_prep.py`) ya
está implementada y verificada.

## Entrega

- Notebook (`.ipynb` y exportado a `.html`) + video explicativo (máx. 3 min).
- Fecha límite: **14 de septiembre, 20:00** (penalización del 30% hasta 15 de
  septiembre 2:00 a.m.; después de esa hora, nota 0).
