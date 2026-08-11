[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)

### 🌐 Языки

[English](../../README.md) | **Русский**

# GeoCode

Набор средств на языке python для разработки месторождений.

![img](../../static/main.jpg)

## Возможности

* визуализация месторождения через сетку (Grid), породу (Rock), состояния (States), скважины (Wells), разломы (Faults) и таблицы PVT
* трёхмерное отображение с возможностью вращать и приближать
* средства предварительной подготовки данных о месторождении
* подробное [описание](https://geo-kit.github.io/GeoCode/)
* [тетради](./notebooks) для пошагового знакомства с набором средств

 > [!TIP]
 > Попробуйте новое [веб-приложение](https://github.com/geo-kit/GeoView.git) на основе GeoCode для просмотра и изучения моделей месторождений.

## Установка

Склонируйте хранилище:

    git clone https://github.com/geo-kit/GeoCode.git

Чтобы проводить расчёты месторождения с помощью [JutulDarcy](https://github.com/sintefmath/JutulDarcy.jl), установите [Julia](https://julialang.org/downloads/) и один раз соберите зависимости управляющей части:

    julia --project=geocode/bin -e "using Pkg; Pkg.instantiate()"

> [!Note]
> Примечание: проект находится в разработке. Будем рады участию и совместной работе.

## Начало работы

Загрузите модель месторождения из файла `.DATA` (несколько моделей лежат в каталоге [open_data](../../open_data)):

```python

  from geocode import Field

  model = Field('model.data').load()
```

Загляните в [ноутбуки](./notebooks), чтобы разобраться в наборе средств шаг за шагом, и в [описание](https://geo-kit.github.io/GeoCode/) — за подробностями.
