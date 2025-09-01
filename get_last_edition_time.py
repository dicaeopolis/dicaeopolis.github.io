import os
import json
import subprocess
from tqdm import tqdm # 导入 tqdm

def main():
    docs_dir = 'docs'
    timestamp_data = {}
    project_root = os.getcwd()  # 假设脚本在项目根目录运行

    # 1. 首先，收集所有需要处理的 markdown 文件路径
    md_files_to_process = []
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.md'):
                md_files_to_process.append(os.path.join(root, file))

    print(f"Found {len(md_files_to_process)} markdown files to process.")
    print("Using the earlier of Git commit time and file system modification time.")

    # 2. 使用 tqdm 遍历文件列表以显示进度
    for full_path in tqdm(md_files_to_process, desc="Generating Timestamps"):
        # 相对于 docs/ 目录的路径
        rel_path_from_docs = os.path.relpath(full_path, docs_dir)
        # Git 路径（相对于项目根目录）
        git_path = os.path.join('docs', rel_path_from_docs)

        # 获取系统记录的文件最后修改时间
        # os.path.getmtime 返回浮点数，转换为整数
        fs_timestamp = int(os.path.getmtime(full_path))
        
        # 默认使用文件系统的修改时间，如果 Git 时间更早，则更新
        chosen_timestamp = fs_timestamp
        
        try:
            # 获取最后一次提交时间
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%ct', '--', git_path], # 增加 '--' 提高文件名处理的安全性
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False
            )
            
            # 如果 git 命令成功执行并返回了时间戳
            if result.returncode == 0 and result.stdout.strip():
                git_timestamp = int(result.stdout.strip())
                # 取 Git 提交时间和文件系统修改时间中，更早（数值更小）的一个
                chosen_timestamp = min(git_timestamp, fs_timestamp)
            
            timestamp_data[rel_path_from_docs] = chosen_timestamp

        except Exception as e:
            # 如果发生异常，仍然使用文件系统的修改时间作为备选
            timestamp_data[rel_path_from_docs] = fs_timestamp
            print(f"\nError processing {git_path}: {e}. Falling back to filesystem timestamp.")

    # 将数据写入 timestamps.json
    timestamp_file = os.path.join(docs_dir, 'timestamps.json')
    with open(timestamp_file, 'w', encoding='utf-8') as f: # 增加 encoding='utf-8'
        json.dump(timestamp_data, f, indent=2, sort_keys=True) # 增加 sort_keys=True 使输出稳定

    print(f"\nGenerated timestamps.json with {len(timestamp_data)} entries")

if __name__ == '__main__':
    main()