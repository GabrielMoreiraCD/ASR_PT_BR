# src/utils_onedrive.py
from __future__ import annotations
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def transform_onedrive_url(share_url: str) -> str:
    # Se o usuário já passar o link com ?download=1 ou download.aspx, retornamos direto
    if "download.aspx" in share_url or "?download=1" in share_url:
        return share_url

    # Tentativa de transformação padrão para links "Anyone link"
    # Remove query params
    base_url = share_url.split("?")[0]
    
    # Se terminar com barra, remove
    if base_url.endswith("/"):
        base_url = base_url[:-1]
        
    # Adiciona o parametro de download
    download_url = f"{base_url}?download=1"
    return download_url

def download_and_extract_zip(url: str, dest_folder: Path):
    """
    Baixa o ZIP da pasta compartilhada e extrai em dest_folder.
    """
    dest_folder.mkdir(parents=True, exist_ok=True)
    zip_path = dest_folder / "dataset.zip"
    
    print(f"[OneDrive] Baixando dataset de: {url}")
    
    # Faz o download com stream para não estourar a memória
    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            print(f"Erro no download. Status: {r.status_code}")
            print("Conteúdo:", r.text[:200])
            r.raise_for_status()
            
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in r.iter_content(block_size):
                size = f.write(data)
                bar.update(size)

    print("[OneDrive] Extraindo arquivos...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print("[OneDrive] Extração concluída.")
    except zipfile.BadZipFile:
        print("ERRO: O arquivo baixado não é um ZIP válido. Verifique se o link do OneDrive está correto.")
        raise
    finally:
        if zip_path.exists():
            os.remove(zip_path)