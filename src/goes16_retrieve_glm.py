import s3fs
import numpy as np
import os
import shutil
from datetime import datetime, timedelta
from netCDF4 import Dataset
import xarray as xr
import sys
import argparse

# Definir limites de coordenadas de interesse (Rio de Janeiro)
lon_min, lon_max = -45.05290312102409, -42.35676996062447
lat_min, lat_max = -23.801876626302175, -21.699774257353113

# Diretórios de saída
output_directory = "data/goes16/glm_files/"
temp_directory = "data/goes16/temp_glm_files/"
final_directory = "data/goes16/aggregated_glm_files/"

def create_directory(directory):
    """Cria o diretório se ele não existir."""
    os.makedirs(directory, exist_ok=True)
    print(f"Diretório {directory} criado (ou já existia).")

def clear_directory(directory):
    """Limpa o diretório e cria novamente."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Diretório {directory} limpo.")
    create_directory(directory)

def download_files(start_date, end_date):
    """Baixa e processa os arquivos GLM para um intervalo de datas especificado."""
    current_date = start_date
    fs = s3fs.S3FileSystem(anon=True)

    clear_directory(temp_directory)
    create_directory(final_directory)

    temp_files = []
    while current_date <= end_date:
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday 
        bucket_path = f'noaa-goes16/GLM-L2-LCFA/{year}/{day_of_year:03d}/'

        print(f"Buscando arquivos para {current_date.strftime('%Y-%m-%d')} (dia {day_of_year})")

        for hour in range(24):
            hour_path = bucket_path + f"{hour:02d}/"
            hourly_files = []
            try:
                hourly_files = fs.ls(hour_path)
            except FileNotFoundError:
                print(f"Hora {hour:02d} não encontrada no bucket. Pulando...")

            for file in hourly_files:
                file_name = file.split('/')[-1]
                local_file_path = os.path.join(temp_directory, file_name)
                print(f"Baixando: {file} para {local_file_path}")
                fs.get(file, local_file_path)
                
                if not filter_by_coordinates(local_file_path):
                    open(local_file_path, 'w').close()  # Cria arquivo vazio se não há dados no Rio

                temp_files.append(local_file_path)

                if len(temp_files) == 20:
                    aggregate_files(temp_files, current_date, hour)
                    clear_directory(temp_directory)  # Limpar a pasta temporária para o próximo ciclo
                    temp_files.clear()

        current_date += timedelta(days=1)

def filter_by_coordinates(file_path):
    """Filtra os eventos GLM de um arquivo NetCDF com base nas coordenadas fornecidas."""
    try:
        dataset = Dataset(file_path, 'r')
        longitudes = dataset.variables['flash_lon'][:]
        latitudes = dataset.variables['flash_lat'][:]
        dataset.close()

        mask = (
            (longitudes >= lon_min) & (longitudes <= lon_max) &
            (latitudes >= lat_min) & (latitudes <= lat_max)
        )

        if np.sum(mask) == 0:
            print(f"Nenhum evento dentro do filtro encontrado em {file_path}. Removendo arquivo.")
            return False
        else:
            print(f"Eventos dentro do filtro encontrados no arquivo {file_path}.")
            return True
    except Exception as e:
        print(f"Erro ao filtrar o arquivo {file_path}: {e}")
        return False

def aggregate_files(files, current_date, hour):
    """Agrupa 30 arquivos válidos e salva como um único arquivo NetCDF."""
    datasets = []
    
    for file in files:
        if os.path.getsize(file) > 0:  # Ignorar arquivos vazios
            ds = xr.open_dataset(file)
            datasets.append(ds)

    if datasets:
        try:
            # Remover 'number_of_events' e 'number_of_groups' no momento da concatenação
            combined = xr.concat(
                [ds.drop_dims(['number_of_events', 'number_of_groups', 'number_of_flashes'], errors="ignore") for ds in datasets],
                dim='time',
                data_vars='minimal',
                coords='minimal',
                compat='override'
            )
            output_file_name = f"glm_agg_{current_date.strftime('%Y%m%d')}_{hour:02d}.nc"
            output_file_path = os.path.join(final_directory, output_file_name)
            combined.to_netcdf(output_file_path)
            print(f"Agrupamento salvo em {output_file_path}")
        except ValueError as e:
            print(f"Erro ao concatenar arquivos: {e}")
    else:
        print("Nenhum dado para agrupar nesta rodada.")


def main(argv):
    parser = argparse.ArgumentParser(description='Download e filtro de arquivos GLM por coordenadas.')
    parser.add_argument('-b', '--start_date', required=True, help='Data de início no formato YYYY-MM-DD')
    parser.add_argument('-e', '--end_date', required=True, help='Data de término no formato YYYY-MM-DD')
    args = parser.parse_args(argv[1:])

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    assert start_date <= end_date, "A data de início deve ser anterior ou igual à data de término."

    download_files(start_date, end_date)

if __name__ == "__main__":
    main(sys.argv)
