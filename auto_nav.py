import re
from pathlib import Path

# --- 配置 ---
DOCS_DIR = Path('docs')
CONFIG_FILE = Path('mkdocs.yml')
# --- 结束配置 ---


def get_md_meta(content: str) -> dict:
    # 使用简单的 regex 来提取 front matter，避免 YAML 库
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
            # 简单处理，不涵盖所有 YAML 复杂性，但对元数据足够
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
    
    # 同样只做简单处理
    doc_class_str = meta.get('class', '').strip('[]')
    first_class = doc_class_str.split(',')[0].strip() if doc_class_str else None

    return {'title': title, 'class': first_class, 'urlname': meta.get('urlname')}

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

def generate_nav_tree(current_path: Path) -> list:
    nav_items = []
    children = sorted(list(current_path.iterdir()), key=lambda p: (not p.is_dir(), p.name.lower()))
    
    for child in children:
        if child.is_dir():
            sub_nav = generate_nav_tree(child)
            if sub_nav:
                category_name = get_category_name(child)
                nav_items.append({category_name: sub_nav})

    for child in children:
        if child.is_file() and child.suffix == '.md' and child.name.lower() != 'index.md':
            info = parse_md_file(child)
            relative_path = child.relative_to(DOCS_DIR)
            final_path = relative_path.with_name(f"{info['urlname']}.md") if info.get('urlname') else relative_path
            path_str = final_path.as_posix()
            nav_items.append({info['title']: path_str})
    return nav_items

def format_nav_to_yaml_string(nav_items: list, level=0) -> str:
    """将导航数据结构递归地格式化为YAML字符串。"""
    lines = []
    indent = '  ' * level
    for item in nav_items:
        key, value = list(item.items())[0]
        # 为包含特殊字符的键添加引号
        safe_key = f"'{key}'" if ':' in key or '#' in key else key

        if isinstance(value, str):  # 是一个文件条目
            lines.append(f"{indent}- {safe_key}: '{value}'")
        elif isinstance(value, list):  # 是一个目录/分类
            lines.append(f"{indent}- {safe_key}:")
            lines.append(format_nav_to_yaml_string(value, level + 1))
    return "\n".join(lines)

def main():
    print("开始为 mkdocs.yml 生成导航 (纯文本模式)...")

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

    # 正则表达式，匹配从行首 'nav:' 开始，直到下一个非空白字符开头的行或文件末尾
    # re.DOTALL 使 '.' 匹配换行符; re.MULTILINE 使 '^' 匹配每行开头
    nav_pattern = re.compile(r"^nav:.*?(?=\n^\S|\Z)", re.DOTALL | re.MULTILINE)

    if nav_pattern.search(original_content):
        print("找到现有的 'nav' 部分，正在替换...")
        new_content, count = nav_pattern.subn(nav_yaml_str, original_content, count=1)
    else:
        print("未找到 'nav' 部分，正在追加到文件末尾...")
        # 确保文件末尾有换行符
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