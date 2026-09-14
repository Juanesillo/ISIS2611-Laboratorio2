"""Pipeline de limpieza y preparación de datos para AlpesPlanck.

Este módulo reutiliza, de forma condensada, el proceso de limpieza y
preparación desarrollado en el Laboratorio 1 (exploración, corrección de
inconsistencias categóricas, corrección de errores de escala, eliminación de
duplicados/outliers imposibles e imputación de nulos sin fuga de datos).

El Laboratorio 2 indica explícitamente que el énfasis no está en la
exploración/procesamiento de datos sino en el modelado, por lo que esta
etapa se encapsula aquí y el notebook del Laboratorio 2 solo debe llamar a
``cargar_datos_limpios()`` para obtener conjuntos de entrenamiento y prueba
ya limpios, sin nulos y sin fuga de información entre train y test.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "Datos Lab 1.csv"
TARGET_COL = "temp_max_manana"

_MESES_MAP = {
    "enero": "enero", "jan": "enero", "january": "enero",
    "febrero": "febrero", "feb": "febrero", "february": "febrero",
    "marzo": "marzo", "mar": "marzo", "march": "marzo",
    "abril": "abril", "apr": "abril", "april": "abril",
    "mayo": "mayo", "may": "mayo",
    "junio": "junio", "jun": "junio", "june": "junio",
    "julio": "julio", "jul": "julio", "july": "julio",
    "agosto": "agosto", "aug": "agosto", "august": "agosto",
    "septiembre": "septiembre", "sep": "septiembre", "sept": "septiembre", "september": "septiembre",
    "octubre": "octubre", "oct": "octubre", "october": "octubre",
    "noviembre": "noviembre", "nov": "noviembre", "november": "noviembre",
    "diciembre": "diciembre", "dec": "diciembre", "december": "diciembre",
}

_ESTACION_MAP = {
    "invierno": "invierno", "winter": "invierno", "invernio": "invierno", "inverano": "invierno",
    "primavera": "primavera", "spring": "primavera", "primav": "primavera", "primaveraa": "primavera",
    "verano": "verano", "summer": "verano", "berano": "verano", "verno": "verano",
    "otono": "otono", "autumn": "otono", "fall": "otono", "otoño": "otono",
}

_SECTOR_MAP = {
    "n": "N", "north": "N", "norte": "N",
    "ne": "NE", "northeast": "NE", "noreste": "NE",
    "e": "E", "east": "E", "este": "E",
    "se": "SE", "southeast": "SE", "sureste": "SE",
    "s": "S", "south": "S", "sur": "S",
    "so": "SO", "southwest": "SO", "suroeste": "SO",
    "o": "O", "west": "O", "oeste": "O",
    "no": "NO", "northwest": "NO", "noroeste": "NO",
}


def _normaliza(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    return re.sub(r"[^a-záéíóúñ]", "", s)


def _corregir_categoricas(data: pd.DataFrame) -> pd.DataFrame:
    data["mes"] = data["mes"].apply(_normaliza).map(_MESES_MAP)
    data["estacion_anio"] = data["estacion_anio"].apply(_normaliza).map(_ESTACION_MAP)
    data["sector_viento"] = data["sector_viento"].apply(_normaliza).map(_SECTOR_MAP)
    return data


def _corregir_escalas(data: pd.DataFrame) -> pd.DataFrame:
    mask_presion = data["presion_media"] > 5000
    data.loc[mask_presion, "presion_media"] = data.loc[mask_presion, "presion_media"] / 10

    mask_hum_media = data["humedad_media"] <= 1
    data.loc[mask_hum_media, "humedad_media"] = data.loc[mask_hum_media, "humedad_media"] * 100

    mask_hum_min = data["humedad_min"] <= 1
    data.loc[mask_hum_min, "humedad_min"] = data.loc[mask_hum_min, "humedad_min"] * 100
    return data


def _eliminar_outliers_imposibles(data: pd.DataFrame) -> pd.DataFrame:
    imposibles = pd.Series(False, index=data.index)

    if "rafaga_media" in data.columns:
        imposibles |= data["rafaga_media"] < 0
    if "viento_norte" in data.columns:
        imposibles |= data["viento_norte"].abs() > 100
    if "registros_del_dia" in data.columns:
        imposibles |= data["registros_del_dia"] > 144

    return data[~imposibles]


def limpiar_datos(raw_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Aplica el proceso de limpieza del Laboratorio 1 sobre los datos crudos."""
    data = pd.read_csv(raw_path)

    data = data.dropna(subset=["fecha"])
    data = _corregir_categoricas(data)
    data = _corregir_escalas(data)

    n_antes = data.shape[0]
    data = data.drop_duplicates(keep="first")

    data["fecha"] = pd.to_datetime(data["fecha"])
    data = data[data["anio"] == data["fecha"].dt.year]

    data = _eliminar_outliers_imposibles(data)

    # El target no se imputa: una fila sin etiqueta real no aporta información
    # válida para entrenar ni para evaluar.
    data = data.dropna(subset=[TARGET_COL])

    return data.reset_index(drop=True)


def cargar_datos_limpios(
    raw_path: Path | str = RAW_DATA_PATH,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Devuelve ``X_train, X_test, y_train, y_test`` limpios y sin fuga de datos.

    Reproduce el mismo split (75/25, ``random_state=42``) usado en el
    Laboratorio 1 y difiere la imputación de nulos hasta después del split,
    ajustando el imputador únicamente con datos de entrenamiento.
    """
    data = limpiar_datos(raw_path)

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include="object").columns

    num_imputer = SimpleImputer(strategy="median").fit(X_train[num_cols])
    cat_imputer = SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols])

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[num_cols] = num_imputer.transform(X_train[num_cols])
    X_train[cat_cols] = cat_imputer.transform(X_train[cat_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = cargar_datos_limpios()

    X_train.assign(**{TARGET_COL: y_train}).to_csv(processed_dir / "train.csv", index=False)
    X_test.assign(**{TARGET_COL: y_test}).to_csv(processed_dir / "test.csv", index=False)

    print(f"train.csv: {X_train.shape[0]} filas, {X_train.shape[1]} columnas")
    print(f"test.csv:  {X_test.shape[0]} filas, {X_test.shape[1]} columnas")
