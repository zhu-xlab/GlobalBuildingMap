# GlobalBuildingMap

GlobalBuildingMap (GBM): the highest accuracy and highest resolution global building map ever created. GBM is derived from nearly 800,000 PlanetScope satellite images, and is distributed in the form of a binary raster (building and non-building) at a resolution of 3 meters.

![cairo](/assets/cairo.png)

Visual comparison of building footprints from different data sources in Cairo, Egypt. The three building footprint layers from GBM (purple), Google (cyan) and OSM (yellow) are overlaid with high-resolution aerial image. Two selected areas, i.e., dense area/informal settlement (orange) and non-dense area (green) are zoomed in. Each area has three subfigures, which show the corresponding high-resolution aerial image as reference (left), GBM overlaid with satellite image (mid) and Google overlaid with satellite image (right). Background images © Google Maps.

## Data

While the PlanetScope imagery required to reproduce this effort is not publicly available due to licensing restrictions, a list of all 790,101 images used in this work can be found in the `assets/downloaded_items.txt` file. A GeoJSON file containing the bounding boxes of all processed images can also be found in `assets/merged_roi.geojson`. Note that Planet data requires a license to download.

## Installation

All Python libraries needed to use this code can be installed using:
```
pip install -r requirements.txt
```

## Training

Please define data directory, checkpoint directory and log directory before running following command:
```
python planet_training_demo.py
```

## Inferencing

Please define satellite image directory, prediction directory and prediction files' name before running following command:
```
python planet_proc_inferDemo.py
```
Global predictions for all continents can be downloaded from: https://doi.org/10.14459/2024MP1764505.002.
