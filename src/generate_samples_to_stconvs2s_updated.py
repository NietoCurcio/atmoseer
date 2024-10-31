import argparse
from datetime import datetime
import glob
import numpy as np
import xarray as xr

# Função para checar o maior gap entre timestamps em um sample
def check_max_gap(sample):
    time_coords = sample['time'].values
    time_diffs = np.diff(time_coords)
    max_diff_ns = np.max(time_diffs).astype('timedelta64[ns]')
    max_diff_minutes = max_diff_ns / np.timedelta64(1, 'm')
    return max_diff_minutes

# Função principal
def main(path, max_gap, bands, output):
    TIMESTEP = 5

    # Diretórios de arquivos por banda e ano
    files_by_band = {}
    for band in bands:
        # Procurando recursivamente por arquivos em subpastas de cada banda
        band_path = f"{path}/band{band}/**/*.nc"
        files_by_band[band] = glob.glob(band_path, recursive=True)

    # Processamento e concatenação dos arquivos de cada banda
    data_by_band = []
    for band, files in files_by_band.items():
        files.sort()
        datasets = [xr.open_dataset(file) for file in files]
        
        # Concatenar ao longo da dimensão 'time' para cada banda
        band_data = xr.concat(datasets, dim='time')
        band_data = band_data.expand_dims(dim='channel', axis=-1)  # Expande a dimensão 'channel'
        data_by_band.append(band_data)

    # Concatenando os dados de todas as bandas ao longo da dimensão 'channel'
    combined_data = xr.concat(data_by_band, dim='channel')
    total_time = combined_data.sizes['time']

    X_samples = []
    Y_samples = []

    # Criando os samples para X e Y e checando gaps
    for i in range(total_time - TIMESTEP):
        X_sample = combined_data.isel(time=slice(i, i + TIMESTEP))
        max_gap_X = check_max_gap(X_sample)

        Y_sample = combined_data.isel(time=slice(i + 1, i + 1 + TIMESTEP))
        max_gap_Y = check_max_gap(Y_sample)

        if max_gap_X > max_gap:
            print(f'Timestamp faltando no X_sample: {X_sample.time}')
            continue
        elif max_gap_Y > max_gap:
            print(f'Timestamp faltando no Y_sample: {Y_sample.time}')
            continue

        X_samples.append(X_sample)
        Y_samples.append(Y_sample)

    # Concatenando os samples ao longo da dimensão 'sample'
    X_samples = xr.concat(X_samples, dim='sample')
    Y_samples = xr.concat(Y_samples, dim='sample')
    combined_samples = xr.Dataset({'x': X_samples, 'y': Y_samples})

    # Salvando o dataset combinado
    combined_samples.to_netcdf(output)
    print(f'Dataset salvo em {output}')

if __name__ == "__main__":
    # Configurando o argparse para aceitar argumentos
    parser = argparse.ArgumentParser(description="Processar arquivos NetCDF com janelas de tempo e checar gaps.")
    
    parser.add_argument('--path', type=str, required=True, help="Caminho dos arquivos NetCDF agrupados por banda e ano (ex: /dados).")
    parser.add_argument('--max-gap', type=float, default=10, help="Intervalo máximo de gap permitido entre os timestamps em minutos (default: 10).")
    parser.add_argument('--bands', type=int, nargs='+', default=[9, 13], help="Bandas a serem processadas como inteiros, ex: 9 13 (default: 9 13).")
    parser.add_argument('--output', type=str, required=True, help="Caminho do arquivo de saída NetCDF.")

    args = parser.parse_args()

    # Chamando a função principal com os argumentos
    main(args.path, args.max_gap, args.bands, args.output)
