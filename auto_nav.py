import re
import json
from pathlib import Path
from collections import defaultdict

# --- 配置 ---
DOCS_DIR = Path('docs')
CONFIG_FILE = Path('mkdocs.yml')
TIMESTAMPS_FILE = DOCS_DIR / 'timestamps.json'
# --- 结束配置 ---

# ... (所有辅助函数 get_md_meta, get_md_title, parse_md_file, load_timestamps, calculate_folder_timestamps, get_category_name 保持不变) ...
def get_md_meta(content: str) -> dict:
    meta_match = re.search(r'^\s*---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not meta_match:
        return {}
    meta_str = meta_match.group(1)
    meta = {}
    for line in meta_str.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip()
            if key in ['class', 'title', 'short_title', 'urlname']:
                 meta[key] = val.strip("'\"")
    return meta

def get_md_title(content: str, meta: dict, fallback_name: str) -> str:
    if meta.get('short_title'):
        return meta['short_title']
    if meta.get('title'):
        return meta['title']
    h1_match = re.search(r'^\s*#\s+(.*)', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    h1_tag_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.IGNORECASE)
    if h1_tag_match:
        return h1_tag_match.group(1).strip()
    return fallback_name

def parse_md_file(md_path: Path) -> dict:
    try:
        content = md_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        print(f"警告: 无法读取文件 {md_path}: {e}")
        return {'title': md_path.stem, 'class': None, 'urlname': None}
    meta = get_md_meta(content)
    title = get_md_title(content, meta, md_path.stem)
    return {'title': title, 'urlname': meta.get('urlname')}

def load_timestamps(file_path: Path) -> dict:
    if not file_path.is_file():
        print(f"警告: 时间戳文件 '{file_path}' 未找到，将按字母顺序排序。")
        return {}
    try:
        with file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return {DOCS_DIR / k: v for k, v in data.items()}
    except (json.JSONDecodeError, IOError) as e:
        print(f"警告: 无法读取或解析时间戳文件 '{file_path}': {e}。将按字母顺序排序。")
        return {}

def calculate_folder_timestamps(file_timestamps: dict) -> dict:
    folder_stamps = defaultdict(float)
    if not file_timestamps:
        return {}
    for file_path, timestamp in file_timestamps.items():
        # 从文件自身所在的目录开始，向上回溯
        for parent in file_path.parents:
            # 只要父目录在 DOCS_DIR 内部或就是 DOCS_DIR 本身，就更新
            if parent == DOCS_DIR or DOCS_DIR in parent.parents:
                 folder_stamps[parent] = max(folder_stamps.get(parent, 0.0), timestamp)
            else:
                # 一旦路径超出了 DOCS_DIR，就停止向上回溯
                break
    return dict(folder_stamps)

def get_category_name(dir_path: Path) -> str:
    for md_file in sorted(dir_path.glob('*.md')):
        try:
            content = md_file.read_text(encoding='utf-8')
            meta = get_md_meta(content)
            doc_class_str = meta.get('class', '').strip('[]')
            if doc_class_str:
                return doc_class_str.split(',')[0].strip()
        except Exception:
            continue
    return dir_path.name.replace('_', ' ').replace('-', ' ').title()

def generate_nav_tree(current_path: Path, file_timestamps: dict, folder_timestamps: dict) -> list:
    pages_with_ts = []
    categories_with_ts = []
    for child in current_path.iterdir():
        if child.is_dir():
            sub_nav = generate_nav_tree(child, file_timestamps, folder_timestamps)
            if sub_nav:
                category_name = get_category_name(child)
                timestamp = folder_timestamps.get(child, 0)
                nav_item = {category_name: sub_nav}
                categories_with_ts.append((nav_item, timestamp))
        elif child.is_file() and child.suffix == '.md':
            if child.name.lower() == 'index.md' and child.parent != DOCS_DIR:
                continue
            info = parse_md_file(child)
            relative_path = child.relative_to(DOCS_DIR)
            final_path = relative_path.with_name(f"{info['urlname']}.md") if info.get('urlname') else relative_path
            timestamp = file_timestamps.get(child, 0)
            nav_item = {info['title']: final_path.as_posix()}
            pages_with_ts.append((nav_item, timestamp))
    sorted_pages = sorted(pages_with_ts, key=lambda item: item[1], reverse=True)
    sorted_categories = sorted(categories_with_ts, key=lambda item: item[1], reverse=True)
    final_pages = [item[0] for item in sorted_pages]
    final_categories = [item[0] for item in sorted_categories]
    return final_pages + final_categories

def format_nav_to_yaml_string(nav_items: list, level=0) -> str:
    lines = []
    indent = '  ' * level
    for item in nav_items:
        key, value = list(item.items())[0]
        safe_key = f"'{key}'" if ':' in key or '#' in key else key
        if isinstance(value, str):
            lines.append(f"{indent}- {safe_key}: '{value}'")
        elif isinstance(value, list):
            lines.append(f"{indent}- {safe_key}:")
            lines.append(format_nav_to_yaml_string(value, level + 1))
    return "\n".join(lines)


# ----------------- main 函数已添加调试打印功能 -----------------
def main():
    print("开始为 mkdocs.yml 生成导航 (文件优先，内部分组时间排序模式)...")
    if not DOCS_DIR.is_dir() or not CONFIG_FILE.is_file():
        print(f"错误: '{DOCS_DIR}' 或 '{CONFIG_FILE}' 未找到。")
        return
    
    print(f"正在加载时间戳文件 '{TIMESTAMPS_FILE}'...")
    file_timestamps = load_timestamps(TIMESTAMPS_FILE)
    folder_timestamps = calculate_folder_timestamps(file_timestamps)

    # --- DEBUG: 打印计算出的文件夹时间戳 ---
    print("\n--- 调试: 计算出的文件夹时间戳 ---")
    if folder_timestamps:
        # 按路径字母顺序打印，方便查看
        for folder_path, timestamp in sorted(folder_timestamps.items(), key=lambda item: str(item[0])):
            # 为了可读性，只打印相对于 DOCS_DIR 的路径
            relative_folder = folder_path.relative_to(DOCS_DIR) if folder_path != DOCS_DIR else Path('.')
            print(f"  - {relative_folder.as_posix()}: {timestamp}")
    else:
        print("  (未计算出任何文件夹时间戳)")
    print("-------------------------------------\n")
    # --- DEBUG END ---

    print(f"正在扫描 '{DOCS_DIR}' 并构建导航...")
    nav_structure = generate_nav_tree(DOCS_DIR, file_timestamps, folder_timestamps)
    
    if not nav_structure:
        print("警告: 未生成任何导航条目。")
        nav_yaml_str = "nav: []"
    else:
        nav_yaml_str = "nav:\n" + format_nav_to_yaml_string(nav_structure, level=1)
    
    try:
        original_content = CONFIG_FILE.read_text(encoding='utf-8')
    except Exception as e:
        print(f"错误: 无法读取 '{CONFIG_FILE}': {e}")
        return
        
    nav_pattern = re.compile(r"^nav:.*?(?=\n^\S|\Z)", re.DOTALL | re.MULTILINE)
    if nav_pattern.search(original_content):
        print("找到现有的 'nav' 部分，正在替换...")
        new_content, _ = nav_pattern.subn(nav_yaml_str, original_content, count=1)
    else:
        print("未找到 'nav' 部分，正在追加到文件末尾...")
        new_content = original_content.rstrip() + '\n\n' + nav_yaml_str + '\n'
        
    try:
        CONFIG_FILE.write_text(new_content, encoding='utf-8')
        print(f"'{CONFIG_FILE}' 更新成功。")
    except Exception as e:
        print(f"错误: 无法写入 '{CONFIG_FILE}': {e}")

if __name__ == "__main__":
    main()