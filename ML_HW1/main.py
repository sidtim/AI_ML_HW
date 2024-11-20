from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pandas as pd
import numpy as np

import joblib
import json

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def prepros_nan(x):
    """
    Функция для предобработки NaN-значений
    """
    x = str(x).split(' ')[0] if x is not np.nan else np.nan
    try:
        x = float(x)
    except ValueError:
        x = np.nan
    return x

# Я буду обучать модель только на вещественных числах. Как мы это делали в части 2. Так проще.
def preprocess_data(df_object):
    """
    В этой функции делаем всю необходимую предобработку JSON,
    для того, чтобы можно было подать эти данные на вход в PipeLine
    """
    #df_object = pd.DataFrame(json_object, index=[0])
    for i_col in ['mileage', 'engine', 'max_power', 'seats']:
        df_object[i_col] = df_object[i_col].apply(prepros_nan)
    
    return df_object[['year', 'km_driven', 'mileage', 'engine', 'max_power','seats']]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # Десериализуем модель, которую получили в части 2
    pipe1 = joblib.load('linregscaler.pkl')
    # Преобразуем данные типа Item в pd.DataFrame
    df_object = pd.DataFrame(item.model_dump(), index=[0])
    # Делаем предобработку, чтобы данные можно было подать на вход PipeLine
    df_object = preprocess_data(df_object)
    # Вычисляем стоимость автомобился
    result = pipe1.predict(df_object)[-1][-1]
    return result


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    # Десериализуем модель, которую получили в части 2
    pipe1 = joblib.load('linregscaler.pkl')
    # Преобразуем данные типа Item в pd.DataFrame
    df_object = pd.DataFrame([i_obj.model_dump() for i_obj in items])
    # Делаем предобработку, чтобы данные можно было подать на вход PipeLine
    df_object = preprocess_data(df_object)
    # Вычисляем стоимость автомобился
    result = pipe1.predict(df_object)
    return result