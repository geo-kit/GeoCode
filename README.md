[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)


# GeoCode

Python framework for reservoir engineering.

![img](static/main.jpg)

## Features

* reservoir representation with Grid, Rock, States, Wells, Faults, and PVT-tables
* interactive 3D visualization
* reservoir preprocessing tools
* detailed [documentation](https://geo-kit.github.io/GeoCode/)
* [tutorials](/tutorials) to explore the framework step-by-step

 > [!TIP]
 > Try out a new [web application](https://github.com/geo-kit/GeoView.git) based on GeoCode for visualization and exploration of reservoir models.

## Installation

Clone the repository:

    git clone https://github.com/geo-kit/GeoCode.git


> [!Note]
> Note: the project is in developement. We welcome contributions and collaborations.

## Quick start

Load a reservoir model from `.DATA` file (some models are given in the [open_data](./open_data) directory):

```python

  from geocode import Field

  model = Field('model.data').load()
```

See the [tutorials](./tutorials) to explore the framework step-by-step
and the [documentation](https://geo-kit.github.io/GeoCode/) for more details.
