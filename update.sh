echo "[+]Starting update process..."
echo "[+]Adding all changes..."
git add .

echo "[+]Committing changes..."
echo "  [+]Common update for content by update.sh"
git commit -m "Common update for content triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"

echo "[+]Constructing navigation automatically..."
python3 ./auto_nav.py

echo "[+]Please update the last edition time in Windows."
python3 ./get_last_edition_time.py

echo "[+]Adding generated navigation and timestamps..."
git add mkdocs.yml docs/timestamps.json

echo "[+]Committing changes..."
echo "  [+]Common update for mkdocs.yml by update.sh"
git commit -m "Common update for mkdocs.yml triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"

echo "[+]Constructing navigation automatically..."
python3 ./auto_nav.py

echo "[+]Pushing to remote repo..."
git push origin main