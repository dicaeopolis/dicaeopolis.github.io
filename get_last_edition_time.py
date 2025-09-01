import os
import json
import subprocess
from tqdm import tqdm

def main():
    docs_dir = 'docs'
    timestamp_data = {}
    project_root = os.getcwd()

    md_files_to_process = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                md_files_to_process.append(os.path.join(root, file))

    print(f"Found {len(md_files_to_process)} markdown files to process.")
    print("Using Git history (--follow) to determine the last modification time.")

    for full_path in tqdm(md_files_to_process, desc="Generating Timestamps"):
        rel_path_from_docs = os.path.relpath(full_path, docs_dir)
        git_path = os.path.join('docs', rel_path_from_docs)
        
        try:
            # 使用 --follow 来追溯文件重命名历史，这是最可靠的方法
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ct', '--follow', '--', git_path],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                timestamp = int(result.stdout.strip())
                timestamp_data[rel_path_from_docs] = timestamp
            else:
                # 如果 Git 找不到历史（例如，一个全新的、还未提交的文件），
                # 作为备选，使用文件系统时间。虽然它可能不准，但有总比没有好。
                fs_timestamp = int(os.path.getmtime(full_path))
                timestamp_data[rel_path_from_docs] = fs_timestamp
                print(f"\nWarning: Could not find Git history for {git_path}. Falling back to filesystem time.")

        except Exception as e:
            print(f"\nError processing {git_path}: {e}. Skipping.")

    timestamp_file = os.path.join(docs_dir, 'timestamps.json')
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        json.dump(timestamp_data, f, indent=2, sort_keys=True)

    print(f"\nGenerated timestamps.json with {len(timestamp_data)} entries")

if __name__ == '__main__':
    main()