import os
import json
import subprocess

def main():
    docs_dir = 'docs'
    timestamp_data = {}
    project_root = os.getcwd()  # 假设脚本在项目根目录运行

    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                # 相对于 docs/ 目录的路径
                rel_path_from_docs = os.path.relpath(full_path, docs_dir)
                # Git 路径（相对于项目根目录）
                git_path = os.path.join('docs', rel_path_from_docs)
                
                try:
                    # 获取最后一次提交时间
                    result = subprocess.run(
                        ['git', 'log', '-1', '--format=%ct', git_path],
                        cwd=project_root,
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        timestamp = int(result.stdout.strip())
                        timestamp_data[rel_path_from_docs] = timestamp
                except Exception as e:
                    print(f"Error processing {git_path}: {e}")

    # 将数据写入 timestamps.json
    timestamp_file = os.path.join(docs_dir, 'timestamps.json')
    with open(timestamp_file, 'w') as f:
        json.dump(timestamp_data, f, indent=2)

    print(f"Generated timestamps.json with {len(timestamp_data)} entries")

if __name__ == '__main__':
    main()