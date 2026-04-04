"""
RE-DOCK Google Drive Distributed Checkpoint Store
===================================================

Uses Google Drive API v3 as the shared state layer for all RE-DOCK workers.
Supports service account auth (headless for K8s/HF) and OAuth2 (local use).

GDrive folder structure::

    RE-DOCK/
    ├── campaigns/{campaign_id}/
    │   ├── config.json
    │   ├── checkpoints/gen_{N}.json
    │   ├── results/R{i}_gen{N}.json
    │   ├── exchanges/exchange_{N}.json
    │   ├── vanthoff/vanthoff_{N}.json
    │   └── poses/{target}_R{i}_best.pdb
    └── workers/{worker_id}_heartbeat.json

Authentication
--------------
- **Service account** (headless): Set ``GDRIVE_SERVICE_ACCOUNT_KEY`` env var
  to path of service account JSON key file. Used by K8s pods and HF Spaces.
- **OAuth2** (interactive): Set ``GDRIVE_CLIENT_SECRETS`` to path of OAuth2
  client secrets JSON. Opens browser for consent on first use, caches token.

Le Bonhomme Pharma / Najmanovich Research Group
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Google API imports (google-api-python-client, google-auth)
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
ROOT_FOLDER_NAME = "RE-DOCK"
TOKEN_CACHE_PATH = os.path.expanduser("~/.redock_gdrive_token.json")


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _authenticate_service_account(key_path: str) -> Any:
    """Authenticate with a GCP service account JSON key."""
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=SCOPES
    )
    return creds


def _authenticate_oauth2(client_secrets_path: str) -> Credentials:
    """Authenticate with OAuth2 (interactive, opens browser)."""
    creds = None

    if os.path.exists(TOKEN_CACHE_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_CACHE_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(TOKEN_CACHE_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def get_credentials() -> Any:
    """Get Google credentials from environment.

    Priority:
    1. GDRIVE_SERVICE_ACCOUNT_KEY (service account JSON path)
    2. GDRIVE_CLIENT_SECRETS (OAuth2 client secrets JSON path)
    """
    sa_key = os.environ.get("GDRIVE_SERVICE_ACCOUNT_KEY")
    if sa_key and os.path.exists(sa_key):
        return _authenticate_service_account(sa_key)

    client_secrets = os.environ.get("GDRIVE_CLIENT_SECRETS")
    if client_secrets and os.path.exists(client_secrets):
        return _authenticate_oauth2(client_secrets)

    raise RuntimeError(
        "No Google Drive credentials found. Set GDRIVE_SERVICE_ACCOUNT_KEY "
        "or GDRIVE_CLIENT_SECRETS environment variable."
    )


# ---------------------------------------------------------------------------
# GDriveStore
# ---------------------------------------------------------------------------

class GDriveStore:
    """Google Drive as distributed checkpoint store for RE-DOCK campaigns.

    Provides save/load operations for campaign state, results, exchange logs,
    Van't Hoff analyses, and best poses. Caches folder IDs to minimize API calls.
    """

    def __init__(self, campaign_id: str = "default"):
        self.campaign_id = campaign_id
        self._creds = get_credentials()
        self._service = build("drive", "v3", credentials=self._creds)
        self._folder_cache: Dict[str, str] = {}

    # -------------------------------------------------------------------
    # Folder management
    # -------------------------------------------------------------------

    def _find_folder(self, name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Find a folder by name under a parent. Returns folder ID or None."""
        query = (f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
                 f"and trashed=false")
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = self._service.files().list(
            q=query, spaces="drive", fields="files(id, name)", pageSize=1
        ).execute()

        files = results.get("files", [])
        return files[0]["id"] if files else None

    def _create_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Create a folder and return its ID."""
        metadata: Dict[str, Any] = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]

        folder = self._service.files().create(
            body=metadata, fields="id"
        ).execute()
        return folder["id"]

    def _ensure_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Find or create a folder, with caching."""
        cache_key = f"{parent_id or 'root'}/{name}"
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        folder_id = self._find_folder(name, parent_id)
        if not folder_id:
            folder_id = self._create_folder(name, parent_id)

        self._folder_cache[cache_key] = folder_id
        return folder_id

    def _get_campaign_folder(self, subfolder: Optional[str] = None) -> str:
        """Get the campaign folder path, creating if needed.

        RE-DOCK/campaigns/{campaign_id}/[subfolder]
        """
        root_id = self._ensure_folder(ROOT_FOLDER_NAME)
        campaigns_id = self._ensure_folder("campaigns", root_id)
        campaign_id = self._ensure_folder(self.campaign_id, campaigns_id)

        if subfolder:
            return self._ensure_folder(subfolder, campaign_id)
        return campaign_id

    def _get_workers_folder(self) -> str:
        """Get RE-DOCK/workers/ folder."""
        root_id = self._ensure_folder(ROOT_FOLDER_NAME)
        return self._ensure_folder("workers", root_id)

    # -------------------------------------------------------------------
    # File upload/download
    # -------------------------------------------------------------------

    def _upload_json(self, folder_id: str, filename: str, data: Any) -> str:
        """Upload a JSON file, replacing if it already exists. Returns file ID."""
        content = json.dumps(data, indent=2, default=str).encode("utf-8")

        # Check for existing file
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = self._service.files().list(
            q=query, spaces="drive", fields="files(id)", pageSize=1
        ).execute().get("files", [])

        media = MediaIoBaseUpload(
            io.BytesIO(content), mimetype="application/json", resumable=True
        )

        if existing:
            # Update existing
            file_id = existing[0]["id"]
            self._service.files().update(
                fileId=file_id, media_body=media
            ).execute()
            return file_id
        else:
            # Create new
            metadata = {"name": filename, "parents": [folder_id]}
            result = self._service.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute()
            return result["id"]

    def _upload_text(self, folder_id: str, filename: str, content: str,
                     mimetype: str = "text/plain") -> str:
        """Upload a text file (PDB, etc.)."""
        encoded = content.encode("utf-8")

        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = self._service.files().list(
            q=query, spaces="drive", fields="files(id)", pageSize=1
        ).execute().get("files", [])

        media = MediaIoBaseUpload(
            io.BytesIO(encoded), mimetype=mimetype, resumable=True
        )

        if existing:
            file_id = existing[0]["id"]
            self._service.files().update(
                fileId=file_id, media_body=media
            ).execute()
            return file_id
        else:
            metadata = {"name": filename, "parents": [folder_id]}
            result = self._service.files().create(
                body=metadata, media_body=media, fields="id"
            ).execute()
            return result["id"]

    def _download_json(self, folder_id: str, filename: str) -> Optional[dict]:
        """Download and parse a JSON file. Returns None if not found."""
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        results = self._service.files().list(
            q=query, spaces="drive", fields="files(id)", pageSize=1
        ).execute().get("files", [])

        if not results:
            return None

        file_id = results[0]["id"]
        request = self._service.files().get_media(fileId=file_id)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        buffer.seek(0)
        return json.loads(buffer.read().decode("utf-8"))

    def _list_files(self, folder_id: str, pattern: str = "") -> List[Dict[str, str]]:
        """List files in a folder, optionally filtered by name pattern."""
        query = f"'{folder_id}' in parents and trashed=false"
        if pattern:
            query += f" and name contains '{pattern}'"

        results = self._service.files().list(
            q=query, spaces="drive",
            fields="files(id, name, modifiedTime)",
            orderBy="modifiedTime desc",
            pageSize=100,
        ).execute()

        return results.get("files", [])

    # -------------------------------------------------------------------
    # Campaign operations
    # -------------------------------------------------------------------

    def save_checkpoint(self, generation: int, checkpoint_data: dict) -> str:
        """Save a campaign checkpoint.

        Path: RE-DOCK/campaigns/{id}/checkpoints/gen_{N}.json
        """
        folder = self._get_campaign_folder("checkpoints")
        filename = f"gen_{generation}.json"
        checkpoint_data["_saved_at"] = time.time()
        checkpoint_data["_generation"] = generation
        return self._upload_json(folder, filename, checkpoint_data)

    def load_latest_checkpoint(self) -> Optional[dict]:
        """Load the most recent checkpoint (highest generation number).

        Returns None if no checkpoints exist.
        """
        folder = self._get_campaign_folder("checkpoints")
        files = self._list_files(folder, "gen_")

        if not files:
            return None

        # Sort by generation number extracted from filename
        def gen_num(f):
            name = f["name"]
            try:
                return int(name.replace("gen_", "").replace(".json", ""))
            except ValueError:
                return -1

        files.sort(key=gen_num, reverse=True)
        latest = files[0]

        request = self._service.files().get_media(fileId=latest["id"])
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buffer.seek(0)
        return json.loads(buffer.read().decode("utf-8"))

    def save_replica_result(
        self, replica_index: int, generation: int, result: dict
    ) -> str:
        """Save a replica's docking result.

        Path: RE-DOCK/campaigns/{id}/results/R{i}_gen{N}.json
        """
        folder = self._get_campaign_folder("results")
        filename = f"R{replica_index}_gen{generation}.json"
        result["_replica_index"] = replica_index
        result["_generation"] = generation
        result["_saved_at"] = time.time()
        return self._upload_json(folder, filename, result)

    def load_replica_results(
        self, replica_index: Optional[int] = None
    ) -> List[dict]:
        """Load replica results, optionally filtered by replica index."""
        folder = self._get_campaign_folder("results")
        pattern = f"R{replica_index}_" if replica_index is not None else ""
        files = self._list_files(folder, pattern)

        results = []
        for f in files:
            data = self._download_json(folder, f["name"])
            if data:
                results.append(data)
        return results

    def save_exchange_log(self, round_num: int, exchange_data: dict) -> str:
        """Save an exchange round log.

        Path: RE-DOCK/campaigns/{id}/exchanges/exchange_{N}.json
        """
        folder = self._get_campaign_folder("exchanges")
        filename = f"exchange_{round_num}.json"
        exchange_data["_round"] = round_num
        exchange_data["_saved_at"] = time.time()
        return self._upload_json(folder, filename, exchange_data)

    def save_vanthoff(self, round_num: int, vanthoff_data: dict) -> str:
        """Save Van't Hoff analysis results.

        Path: RE-DOCK/campaigns/{id}/vanthoff/vanthoff_{N}.json
        """
        folder = self._get_campaign_folder("vanthoff")
        filename = f"vanthoff_{round_num}.json"
        vanthoff_data["_round"] = round_num
        vanthoff_data["_saved_at"] = time.time()
        return self._upload_json(folder, filename, vanthoff_data)

    def save_best_pose(
        self, target_id: str, replica_index: int, pdb_content: str
    ) -> str:
        """Save the best pose PDB for a target/replica.

        Path: RE-DOCK/campaigns/{id}/poses/{target}_R{i}_best.pdb
        """
        folder = self._get_campaign_folder("poses")
        filename = f"{target_id}_R{replica_index}_best.pdb"
        return self._upload_text(folder, filename, pdb_content,
                                 mimetype="chemical/x-pdb")

    def worker_heartbeat(self, worker_id: str, status: dict) -> str:
        """Update worker heartbeat.

        Path: RE-DOCK/workers/{worker_id}_heartbeat.json
        """
        folder = self._get_workers_folder()
        filename = f"{worker_id}_heartbeat.json"
        status["_worker_id"] = worker_id
        status["_timestamp"] = time.time()
        return self._upload_json(folder, filename, status)

    def get_worker_heartbeats(self) -> List[dict]:
        """Get all worker heartbeats."""
        folder = self._get_workers_folder()
        files = self._list_files(folder, "heartbeat")

        heartbeats = []
        for f in files:
            data = self._download_json(folder, f["name"])
            if data:
                heartbeats.append(data)
        return heartbeats


# ---------------------------------------------------------------------------
# Convenience function for workers
# ---------------------------------------------------------------------------

def quick_save(
    campaign_id: str,
    worker_id: str,
    replica_index: int,
    generation: int,
    result: dict,
    pdb_content: Optional[str] = None,
) -> Dict[str, str]:
    """One-call save for workers: result + heartbeat + optional best pose.

    Returns dict of GDrive file IDs for each saved item.

    Usage::

        from benchmarks.re_dock.gdrive_store import quick_save

        ids = quick_save(
            campaign_id="my_campaign",
            worker_id="hf-worker-0",
            replica_index=0,
            generation=5,
            result={"best_energy": -10.5, "energies": [...]},
            pdb_content="ATOM  1  CA ...",
        )
    """
    store = GDriveStore(campaign_id)
    saved = {}

    # Save result
    saved["result"] = store.save_replica_result(replica_index, generation, result)

    # Heartbeat
    saved["heartbeat"] = store.worker_heartbeat(worker_id, {
        "replica_index": replica_index,
        "generation": generation,
        "best_energy": result.get("best_energy", float("inf")),
        "status": "active",
    })

    # Best pose (optional)
    if pdb_content:
        target_id = result.get("pdb_id", result.get("target_id", "unknown"))
        saved["pose"] = store.save_best_pose(target_id, replica_index, pdb_content)

    return saved
