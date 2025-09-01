import re
from pathlib import Path

# --- 配置 ---
DOCS_DIR = Path('docs')
CONFIG_FILE = Path('mkdocs.yml')
# --- 结束配置 ---


def get_md_meta(content: str) -> dict:
    # (此函数无需修改)
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
    # (此函数无需修改)
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
    # (此函数无需修改)
    try:
        content = md_path.read_text(encoding='utf-8')
    except (IOError, UnicodeDecodeError) as e:
        print(f"警告: 无法读取文件 {md_path}: {e}")
        return {'title': md_path.stem, 'class': None, 'urlname': None}

    meta = get_md_meta(content)
    title = get_md_title(content, meta, md_path.stem)
    
    doc_class_str = meta.get('class', '').strip('[]')
    first_class = doc_class_str.split(',')[0].strip() if doc_class_str else None

    return {'title': title, 'class': first_class, 'urlname': meta.get('urlname')}

# ----------------- 核心逻辑: 文件夹结构 + 元数据命名 -----------------

def get_category_name(dir_path: Path) -> str:
    """
    确定目录的显示名称。
    它会查找目录中第一个定义了 'class' 元数据的文件，并使用该值。
    如果找不到，则使用格式化的目录名作为后备。
    """
    # 按名称排序以确保每次运行结果一致
    for md_file in sorted(dir_path.glob('*.md')):
        try:
            content = md_file.read_text(encoding='utf-8')
            meta = get_md_meta(content)
            doc_class_str = meta.get('class', '').strip('[]')
            if doc_class_str:
                # 找到第一个就立即返回
                return doc_class_str.split(',')[0].strip()
        except Exception:
            # 如果文件读取失败，跳过
            continue
    # 如果循环结束都没找到，使用目录名作为后备
    return dir_path.name.replace('_', ' ').replace('-', ' ').title()

def generate_nav_tree(current_path: Path) -> list:
    """
    递归地根据文件系统结构生成导航树。
    - 结构由文件夹决定。
    - 分类名由 get_category_name 决定。
    - 页面按字母顺序排列。
    """
    nav_items = []
    
    # 按名称字母顺序对所有子项（文件和目录）进行排序
    children = sorted(list(current_path.iterdir()), key=lambda p: p.name.lower())
    
    # 使用单一循环处理所有子项
    for child in children:
        # 情况A：如果子项是目录，递归处理
        if child.is_dir():
            sub_nav = generate_nav_tree(child)
            if sub_nav:
                category_name = get_category_name(child)
                nav_items.append({category_name: sub_nav})
        
        # 情况B：如果子项是 Markdown 文件
        elif child.is_file() and child.suffix == '.md':
            # 关键：只忽略子目录中的 index.md，不忽略根目录的
            if child.name.lower() == 'index.md' and child.parent != DOCS_DIR:
                continue

            info = parse_md_file(child)
            relative_path = child.relative_to(DOCS_DIR)
            final_path = relative_path.with_name(f"{info['urlname']}.md") if info.get('urlname') else relative_path
            path_str = final_path.as_posix()
            nav_items.append({info['title']: path_str})
            
    return nav_items

# ----------------- 结束核心逻辑 -----------------

def format_nav_to_yaml_string(nav_items: list, level=0) -> str:
    # (此函数无需修改)
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

def main():
    print("开始为 mkdocs.yml 生成导航 (混合模式)...")

    if not DOCS_DIR.is_dir():
        print(f"错误: '{DOCS_DIR}' 文件夹未找到。")
        return
    if not CONFIG_FILE.is_file():
        print(f"错误: '{CONFIG_FILE}' 文件未找到。")
        return

    print(f"正在扫描 '{DOCS_DIR}'...")
    nav_structure = generate_nav_tree(DOCS_DIR)
    
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
        new_content, count = nav_pattern.subn(nav_yaml_str, original_content, count=1)
    else:
        print("未找到 'nav' 部分，正在追加到文件末尾...")
        if not original_content.endswith('\n'):
            original_content += '\n'
        new_content = original_content + '\n' + nav_yaml_str + '\n'

    try:
        CONFIG_FILE.write_text(new_content, encoding='utf-8')
        print(f"'{CONFIG_FILE}' 更新成功。")
    except Exception as e:
        print(f"错误: 无法写入 '{CONFIG_FILE}': {e}")

if __name__ == "__main__":
    main()