#Script para para pegar as gravacoes do BD e salvar em cache por lote
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from azure.storage.blob import BlobClient


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_blob_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("blob://"):
        raise ValueError(f"Not a blob uri: {uri}")
    rest = uri[len("blob://"):]
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid blob uri: {uri}")
    return parts[0], parts[1]


def download_blob_to_cache(
    blob_uri: str,
    cache_dir: Path,
    connection_string: Optional[str] = None,
    account_url: Optional[str] = None,
    credential: Optional[object] = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    container, blob_path = parse_blob_uri(blob_uri)
    ext = Path(blob_path).suffix or ".wav"
    local_path = cache_dir / f"{_hash(blob_uri)}{ext}"

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    if connection_string:
        bc = BlobClient.from_connection_string(
            conn_str=connection_string,
            container_name=container,
            blob_name=blob_path,
        )
    else:
        if not account_url or credential is None:
            raise RuntimeError(
                "Blob download needs either AZURE_STORAGE_CONNECTION_STRING "
                "or (AZURE_STORAGE_ACCOUNT_URL + credential)."
            )
        bc = BlobClient(
            account_url=account_url,
            container_name=container,
            blob_name=blob_path,
            credential=credential,
        )

    with open(local_path, "wb") as f:
        stream = bc.download_blob()
        stream.readinto(f)

    return local_path
