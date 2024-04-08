#!/bin/bash

echo "Creating data folders"

echo "Initializing data folders"
mkdir -p ./data

mkdir -p ./data/ws

mkdir -p ./data/ws/inmet
echo "INMET data folder created"

mkdir -p ./data/ws/alertario
mkdir -p ./data/ws/alertario/ws
mkdir -p ./data/ws/alertario/rain_gauge
mkdir -p ./data/ws/alertario/rain_gauge_era5_fused
echo "AlertaRio data folder created"

mkdir -p ./data/ws/websirenes
mkdir -p ./data/ws/websirenes/rain_gauge
mkdir -p ./data/ws/websirenes/rain_gauge_era5_fused
echo "Websirenes data folder created"

mkdir -p ./data/as

mkdir -p ./data/NWP
mkdir -p ./data/NWP/ERA5
mkdir -p ./data/NWP/ERA5/montly_data
echo "NWP ERA5 data folder created"

mkdir -p ./data/datasets
mkdir -p ./data/goes16
echo "GOES16 data folder created"

mkdir -p ./models
echo "Models folder created"

echo "Folders created"
