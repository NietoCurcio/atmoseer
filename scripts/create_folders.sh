#!/bin/bash

echo "Creating data folders"

mkdir -p ./data

mkdir -p ./data/ws

mkdir -p ./data/ws/inmet
mkdir -p ./data/ws/alertario
mkdir -p ./data/ws/alertario/ws
mkdir -p ./data/ws/alertario/rain_gauge_era5_fused

mkdir -p ./data/as

mkdir -p ./data/NWP
mkdir -p ./data/NWP/ERA5
mkdir -p ./data/NWP/ERA5/montly_data

mkdir -p ./data/datasets
mkdir -p ./data/goes16

echo "Creating models folders"

mkdir -p ./models

echo "Folders created"
