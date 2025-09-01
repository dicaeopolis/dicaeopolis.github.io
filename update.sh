echo "[+]Starting update process..."

echo "[+]Constructing navigation automatically..."
python3 ./auto_nav.py

echo "[+]Please update the last edition time in Windows."
# python3 ./get_last_edition_time.py

echo "[+]Adding all changes..."
git add .

echo "[+]Committing changes..."
echo "  [+]Common update triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"
git commit -m "Common update triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"

echo "[+]Pushing to remote repo..."
git push origin main