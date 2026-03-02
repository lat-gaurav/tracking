#!/usr/bin/env env zsh
# pack_resources.sh
# -----------------
# Zips the resources/ folder, ready for upload to Google Drive.
#
# Usage:
#   ./pack_resources.sh
#
# Then upload resources.zip to Google Drive:
#   1. Go to drive.google.com → New → File upload → resources.zip
#   2. Right-click the uploaded file → Share → Anyone with the link (Viewer)
#   3. Copy the file ID from the share URL:
#        https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#   4. Paste FILE_ID into download_resources.py  →  GDRIVE_FILE_ID = "..."
#   5. Commit and push:
#        git add download_resources.py
#        git commit -m "add gdrive resource id"
#        git push

set -e
cd "$(dirname "$0")"

OUTFILE="resources.zip"

echo "Zipping resources/ → $OUTFILE ..."
zip -r "$OUTFILE" resources/ -x "*.DS_Store"

SIZE=$(du -sh "$OUTFILE" | cut -f1)
echo "Done.  $OUTFILE  ($SIZE)"
echo
echo "Next steps:"
echo "  1. Upload $OUTFILE to Google Drive"
echo "  2. Share as 'Anyone with the link can view'"
echo "  3. Paste the file ID into download_resources.py"
