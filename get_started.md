# Atmoseer get started

## Project setup

Create folders:
```sh
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
```

Create an `atmoseer` [conda](https://www.anaconda.com/download/) environment and install its dependencies:
```sh
conda create --name atmoseer python=3.9
conda activate atmoseer

TODO install packages or conda env create -f environment.yml
```

## Data retrieval

Copy the [`WeatherStations.csv`](https://portal.inmet.gov.br/paginas/catalogoaut#) file into `./data/ws` folder.

### INMET
- retrieve INMET:
    ```sh
    python src/retrieve_ws_inmet.py -s A602 -b 2007 -e 2023 --api_token INMET_TOKEN
    ```
    That script creates the `./data/ws/inmet/A602.parquet` file

### AlertaRio

- AlertaRio weather stations

    ```sh
    python create_alertario_ws_parquet.py
    ```
    That script creates the `./data/ws/alertario/ws/sao_cristovao.parquet` and `guaratiba.parquet` files.

- AlertaRio rain gauge stations

    Since rain gauge stations only have pluimoetric info, we need to fuse this data with ERA5 NWP, to 
    simulate values for temperature, relative humidity, barometric pressure, wind speed and sind direction.

    First, retrieve ERA5 data:
    1. Go to https://cds.climate.copernicus.eu/api-how-to and register an account
    2. After account registration, go to https://cds.climate.copernicus.eu/user and get the UID and API key
    3. go to your home directory and create the `.cdsapirc` file:
        ```sh
        cd ~
        touch .cdsapirc
        ```
    4. Paste the following content to the `.cdsapirc` file, replace `YOUR_UID` and `YOUR_APIKEY` with the values retrieved from step 2:
        ```sh
        url: https://cds.climate.copernicus.eu/api/v2
        key: YOUR_UID:YOUR_APIKEY
        ```
    5. Confirm terms acceptance at https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products
    6. Exeute the ERA5 data retrival script:
    ```sh
    python src/retrieve_ERA5.py -b 1997 -e 2023
    ```
    Creates `./data/NWP/ERA5

    TODO
    Execute the script `create_alertario_gs_parquet.py`
    ```sh
    python create_alertario_gs_parquet.py
    ```
    Creates `./data/ws/alertario/ws/sao_cristovao.parquet` and `guaratiba.parquet`.

### Sirenes


Execute the


2.0 - preprocess INMET (only weather stations)
```sh
python src/preprocess_ws.py -s A602
# Please see, ./src/globals:INMET_WEATHER_STATION_IDS
```
Creates `./data/ws/inmet/A602_preprocessed.parquet.gzip` file

2.1 - preprocess AlertaRio (both weather and gauge stations)

2.1.0 - preprocess AlertaRio weather stations

Note it assumes the files `./data/ws/alertario/ws/sao_cristovao_parquet` (and `guaratiba.parquet`) exists
```sh
# See: data/alertario_weather-station/create_aleratrio_ws_parquet.ipynb

python src/preprocess_ws.py -s sao_cristovao
python src/preprocess_ws.py -s guaratiba
```

2.1.1  - preprocess AlertaRio gauge stations
```sh
# See: data/alertario_rain_gauge/create_aleratrio_gs_parquet.ipynb
```

# todo
# ## data retrival - inmet, aleratrio, sirenes
# ## data preprocess - inmet, aleratrio, sirenes
# ...