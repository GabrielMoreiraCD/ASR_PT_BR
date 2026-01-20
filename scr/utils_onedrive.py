# src/utils_onedrive.py

from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
import requests


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_onedrive_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri.startswith("onedrive://"):
        raise ValueError(f"Not an onedrive uri: {uri}")
    return uri[len("onedrive://") :].lstrip("/")


def _get_env(name: str, required: bool = True) -> str:
    v = os.environ.get(name, "").strip()
    if required and not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def get_graph_token_client_secret() -> str:
    tenant_id = _get_env("MS_TENANT_ID")
    client_id = _get_env("MS_CLIENT_ID")
    client_secret = _get_env("MS_CLIENT_SECRET")

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
        "grant_type": "client_credentials",
    }
    r = requests.post(token_url, data=data, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def build_drive_item_download_url(drive_id: str, root_folder: str, rel_path: str) -> str:
    root_folder = (root_folder or "").strip().strip("/")
    rel_path = rel_path.strip().lstrip("/")

    if root_folder:
        full = f"{root_folder}/{rel_path}"
    else:
        full = rel_path

    return f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{full}:/content"


def download_onedrive_to_cache(
    onedrive_uri: str,
    cache_dir: Path,
    token: Optional[str] = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    rel_path = parse_onedrive_uri(onedrive_uri)
    ext = Path(rel_path).suffix or ".bin"
    local_path = cache_dir / f"{_sha1(onedrive_uri)}{ext}"

    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    drive_id = _get_env("ONEDRIVE_DRIVE_ID")
    root_folder = _get_env("ONEDRIVE_ROOT", required=False)

    if token is None:
        token = get_graph_token_client_secret()

    url = build_drive_item_download_url(drive_id, root_folder, rel_path)
    headers = {"Authorization": f"Bearer {token}"}

    with requests.get(url, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return local_path
