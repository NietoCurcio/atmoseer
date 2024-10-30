import s3fs
import numpy as np
import os
import shutil

output_directory = "./data/goes16/glm"

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  
        print(f"Diretório {directory} limpo.")
    os.makedirs(directory, exist_ok=True)  

clear_directory(output_directory)

fs = s3fs.S3FileSystem(anon=True)

bucket_path = 'noaa-goes16/GLM-L2-LCFA/2023/322/'  # 322 corresponde ao dia do ano (18 de novembro de 2023)

files = []

# Itera pelas subpastas das horas (00 a 23)
for hour in range(24):
    hour_path = bucket_path + f"{hour:02d}/"  
    try:
        hourly_files = fs.ls(hour_path)
        files.extend(hourly_files)  
    except FileNotFoundError:
        print(f"Hora {hour:02d} não encontrada no bucket. Pulando...")

print(f"Total de arquivos encontrados para o dia: {len(files)}")

for file in files:
    file_name = file.split('/')[-1]
    
    local_file_path = os.path.join(output_directory, file_name)

    print(f"Baixando: {file} para {local_file_path}")
    fs.get(file, local_file_path)

print("Download concluído.")