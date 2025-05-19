# GlobalBuildingMap

GlobalBuildingMap (GBM): the highest accuracy and highest resolution global building map ever created. GBM is derived from nearly 800,000 PlanetScope satellite images, and is distributed in the form of a binary raster (building and non-building) at a resolution of 3 meters.

![cairo](/assets/cairo.png)

Visual comparison of building footprints from different data sources in Cairo, Egypt. The three building footprint layers from GBM (purple), Google (cyan) and OSM (yellow) are overlaid with high-resolution aerial image. Two selected areas, i.e., dense area/informal
settlement (orange) and non-dense area (green) are zoomed in. Each area has three subfigures, which show the corresponding high-resolution aerial image as reference (left), GBM overlaid with satellite image (mid) and Google overlaid with satellite image (right). Background images © Google Maps.

## Installation

All Python libraries needed to use this code can be installed using:
```
pip install -r requirements.txt
```

## Training
please define data directory, checkpoint directory and log directory before running following command
```
python planet_training_demo.py
```

## Inferencing
please define satellite image directory, prediction directory and prediction files' name before running following command
```
python planet_proc_inferDemo.py
```
