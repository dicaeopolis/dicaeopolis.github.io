echo "Updating last edited time..."
python3 ./get_last_edition_time.py

echo "add all changes..."
git add .

echo "commit changes..."
echo "Common update triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"
git commit -m "Common update triggered at $(date +"%Y-%m-%d %H:%M:%S") by update.sh"

echo "push to remote repo..."
git push origin main