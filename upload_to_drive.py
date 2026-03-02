#!/usr/bin/env python3
"""
upload_to_drive.py
------------------
Uploads resources.zip to Google Drive and prints the resulting file ID.

ONE-TIME SETUP (only needed once):
  1. Go to  https://console.cloud.google.com/
  2. Create a project (or pick an existing one)
  3. Enable the Google Drive API:
       APIs & Services → Library → search "Google Drive API" → Enable
  4. Create OAuth credentials:
       APIs & Services → Credentials → + Create Credentials → OAuth client ID
       Application type: Desktop app  →  Create  →  Download JSON
  5. Save the downloaded file as  credentials.json  next to this script.

USAGE (first run — opens browser for Google sign-in):
  python3 upload_to_drive.py

  Subsequent runs reuse the saved token (token.json) without re-opening browser.

After upload, paste the printed FILE_ID into download_resources.py:
  RESOURCES_ZIP_ID = "<FILE_ID>"
Then commit and push.
"""

import os
import sys
from pathlib import Path

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:
    sys.exit("Missing packages. Run:\n  pip install google-api-python-client google-auth-oauthlib")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
UPLOAD_FILE = Path(__file__).parent / "resources.zip"
CREDENTIALS_FILE = Path(__file__).parent / "credentials.json"
TOKEN_FILE = Path(__file__).parent / "token.json"

# Optional: put the zip inside a specific Drive folder by its ID.
# Leave empty ("") to upload to root My Drive.
DRIVE_FOLDER_ID = ""


def get_creds():
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                sys.exit(
                    f"credentials.json not found at {CREDENTIALS_FILE}\n"
                    "See the ONE-TIME SETUP instructions at the top of this file."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def upload():
    if not UPLOAD_FILE.exists():
        sys.exit(f"Upload file not found: {UPLOAD_FILE}\nRun ./pack_resources.sh first.")

    size_gb = UPLOAD_FILE.stat().st_size / 1e9
    print(f"Uploading {UPLOAD_FILE.name}  ({size_gb:.1f} GB) ...")

    creds = get_creds()
    service = build("drive", "v3", credentials=creds)

    metadata = {"name": "resources.zip"}
    if DRIVE_FOLDER_ID:
        metadata["parents"] = [DRIVE_FOLDER_ID]

    media = MediaFileUpload(
        str(UPLOAD_FILE),
        mimetype="application/zip",
        resumable=True,
        chunksize=10 * 1024 * 1024,  # 10 MB chunks
    )

    request = service.files().create(body=metadata, media_body=media, fields="id,name,size")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"  {pct}% uploaded...", end="\r", flush=True)

    file_id = response["id"]
    print(f"\nUpload complete!")
    print(f"  File: {response['name']}")
    print(f"  Size: {int(response['size']) / 1e9:.2f} GB")
    print(f"  File ID: {file_id}")
    print()

    # Make it readable by anyone with the link
    service.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
    ).execute()
    print("Permissions set: anyone with the link can view.")
    print()
    print("Next steps:")
    print(f'  1. Open download_resources.py and set RESOURCES_ZIP_ID = "{file_id}"')
    print("  2. git add download_resources.py && git commit -m 'update resources zip id' && git push")

    return file_id


if __name__ == "__main__":
    upload()
