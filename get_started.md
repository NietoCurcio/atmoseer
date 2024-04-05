# Atmoseer get started

## Project setup

Create dependency folders by executing the following shell script:
```sh
./scripts/create_folders.sh
```

Ensure [conda](https://www.anaconda.com/download/) package manager is installed:
```sh
conda --version
```

Create the `atmoseer` environment:
```sh
conda create --name atmoseer python=3.9
conda activate atmoseer
```

Install conda [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) for fast solving environment when installing dependencies and set it in the `atmoseer` env:
```sh
conda update -n base conda
conda install -n atmoseer conda-libmamba-solver
conda config --env --set solver libmamba
# It helps us from being stuck at "Solving environment: \ "
```

Add the following package channels to `atmoseer` environment:
```sh
conda config --env --add channels conda-forge
conda config --env --add channels pytorch
conda config --env --add channels anaconda
# run conda config --show channels to see configured channels
```

Install the dependencies in the `atmoseer` conda environment from the `config/environment.yml` file:
```sh
conda env update --name atmoseer --file config/environment.yml
```

Exporting an environment file across platforms:
```sh
conda env export --from-history | grep -v "^prefix: " > config/environment.yml
# This command writes the environment in the config/environment.yml file, ignoring the prefix local setting.
```

Exporting an environment is useful if some contributor has installed a package not listed in the `config/environment.yml` file. Please, consider passing the package version explicitly when installing a conda package, so that when exporting the environment the package version will be configured as well. Installing a package with version example:
```sh
# conda install conda-forge::pandas==2.2.1
```

## Data retrieval

Copy the [`WeatherStations.csv`] file into `./data/ws` folder.

### INMET
- retrieve INMET:
    ```sh
    python src/retrieve_ws_inmet.py -s A602 -b 2007 -e 2023 --api_token INMET_TOKEN
    ```
    That script creates the `./data/ws/inmet/A602.parquet` file. Please see `./src/globals:INMET_WEATHER_STATION_IDS` for the available station IDs.

### AlertaRio

- AlertaRio weather stations

    ```sh
    jupyter nbconvert --execute --to notebook --inplace notebooks/alertario/create_alertario_ws_parquet.ipynb
    ```
    That script creates the `./notebooks/alertario/sao_cristovao.parquet` and `guaratiba.parquet` files by executing the jupyter notebook. Once created, move these files to the `./data/ws/alertario/ws/` folder. The notebook assumes the `./notebooks/alertario/alertario_weather_station` folder exists with the alertario data within it.

- AlertaRio rain gauge stations

    Since rain gauge stations only have pluviometric info, we need to fuse this data with ERA5 NWP, to 
    simulate values for temperature, relative humidity, barometric pressure, wind speed and wind direction.

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
    6. Execute the ERA5 data retrival script:
        ```sh
        python src/retrieve_ERA5.py -b 1997-01 -e 2023-12
        ```
        It creates the `./data/NWP/ERA5/RJ_1997_2023.nc` file. Note, this script also has a `--prepend_dataset` flag to merge new data with an existing dataset. For example, `python src/retrieve_ERA5.py -b 2023-05 -e 2024-03 --prepend_dataset data/NWP/ERA5/RJ_1997_2023.nc` will prepend `RJ_1997_2023.nc` into `RJ_2023_2024.nc`, creating the `RJ_1997_2024.nc` dataset.

    Second, create the `.parquet` files in the `./data/ws/alertario/rain_gauge/` and `./data/ws/alertario/rain_gauge_era5_fused/` folders:
    1. Execute the `create_alertario_gs_parquet.ipynb` notebook:
        ```sh
        jupyter nbconvert --execute --to notebook --inplace notebooks/alertario/create_alertario_gs_parquet.ipynb
        ```
        
        This jupyter notebook creates the `./notebooks/alertario/alertario_stations.parquet` file and a list of parquet files `[alto_da_boa_vista.parquet, anchieta.parquet, ..., vidigal.parquet]`. Move `alertario_stations.parquet` into `./data/ws/alertario_stations.parquet`. Move the list of `.parquet` files into `./data/ws/alertario/rain_gauge/` folder. The notebook assumes the `./notebooks/alertario/alertario_rain_gauge` folder exists with the alertario rain gauge data within it.
    2. Execute the `fuse_rain_gauge_and_era5.py` script:
        ```sh
        python src/fuse_rain_gauge_and_era5.py --dataset_file ./data/NWP/ERA5/RJ_1997_2024.nc
        ```
        This script creates a list of parquet files `[alto_da_boa_vista.parquet, anchieta.parquet, ..., vidigal.parquet]` in the `./data/ws/alertario/rain_gauge_era5_fused/` folder.

    TODO: automate the moving of .parquet files.

### Sirenes

Execute the

## Data preprocessing

### Inmet

- Preprocess INMET (only weather stations)
    ```sh
    python src/preprocess_ws.py -s A602
    ```
    This script creates the `./data/ws/inmet/A602_preprocessed.parquet.gzip` file

### AlertaRio

Preprocess AlertaRio

- AlertaRio weather stations

    Note it assumes the files `./data/ws/alertario/ws/sao_cristovao.parquet` (and `guaratiba.parquet`) exists
    ```sh
    python src/preprocess_ws.py -s sao_cristovao
    python src/preprocess_ws.py -s guaratiba
    ```
    These scripts create the `./data/ws/alertario/ws/sao_cristovao_preprocessed.parquet.gzip` (and `guaratiba_preprocessed.parquet.gzip`) files

- AlertaRio rain gauge stations

    Note it assumes the files in the `./data/ws/alertario/rain_gauge_era5_fused/` folder exist.
    ```sh
    python src/preprocess_gs.py --station_id jardim_botanico
    ```
    This script creates the `./data/ws/alertario/rain_gauge_era5_fused/jardim_botanico_preprocessed.parquet.gzip` file.

    TODO document --station_id all


### Sirenes

## Data building

This section describes the creation of training, validation and testing datasets for the INMET, AlertaRio and Sirenes entities.

### INMET

python src/build_datasets.py -s A652 --train_test_threshold "2021-11-12"
see src/globals.py

### AlertaRio

python src/build_datasets.py -s urca --train_test_threshold "2021-11-12"
see src/globals.py

### Sirenes

TODO

## Model training and evaluation

call src/train_model.py