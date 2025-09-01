import re
import os
import json
from functools import lru_cache
from datetime import datetime

# --- 配置 ---

# 预编译正则表达式
# 在这里排除不需要统计和显示引用信息的文件：
EXCLUDE_PATTERNS = [
    re.compile(r'^index\.md$'),
    re.compile(r'^about/'),
    re.compile(r'^trip/index\.md$'),
    re.compile(r'^relax/index\.md$'),
    re.compile(r'^blog/indexblog\.md$'),
    re.compile(r'^blog/posts\.md$'),
    re.compile(r'^develop/index\.md$'),
    re.compile(r'waline\.md$'),
    re.compile(r'link\.md$'),
    re.compile(r'404\.md$'),
]

# 优化的字符统计正则表达式
CHINESE_CHARS_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
CODE_BLOCK_PATTERN = re.compile(r'```.*?```', re.DOTALL)

# 预定义排除类型
EXCLUDE_TYPES = frozenset({'landing', 'special', 'widget'})

# --- 核心函数 ---

@lru_cache(maxsize=256)
def extract_non_code_content(content_hash, markdown):
    """提取不在代码块中的内容"""
    content = CODE_BLOCK_PATTERN.sub('', markdown)
    return content

def count_code_lines(markdown):
    """统计代码行数 - 简化版本，只计算代码块中的行数"""
    code_blocks = CODE_BLOCK_PATTERN.findall(markdown)
    total_code_lines = 0
    for block in code_blocks:
        lines = block.split('\n')
        code_lines = [
            line for line in lines 
            if not line.strip().startswith('```') and line.strip()
        ]
        total_code_lines += len(code_lines)
    return total_code_lines

def calculate_reading_stats(markdown):
    """计算中文字符数和代码行数"""
    content_hash = hash(markdown)
    non_code_content = extract_non_code_content(content_hash, markdown)
    chinese_chars = len(CHINESE_CHARS_PATTERN.findall(non_code_content))
    code_lines = count_code_lines(markdown)
    reading_time = max(1, round(chinese_chars / 200))
    return reading_time, chinese_chars, code_lines

@lru_cache(maxsize=1)
def load_timestamps(docs_dir):
    """加载并缓存 timestamps.json 文件，避免重复读取"""
    timestamps_path = os.path.join(docs_dir, 'timestamps.json')
    if not os.path.exists(timestamps_path):
        print(f"Warning: timestamps.json not found at {timestamps_path}")
        return None
    try:
        with open(timestamps_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading or parsing timestamps.json: {e}")
        return None

def get_git_revision_date(page, config):
    """从 timestamps.json 获取文件的最后 Git 提交时间"""
    docs_dir = config['docs_dir']
    relative_path_key = page.file.src_path.replace(os.path.sep, '/')
    
    timestamps = load_timestamps(docs_dir)
    if timestamps and relative_path_key in timestamps:
        timestamp_value = timestamps[relative_path_key]
        return datetime.fromtimestamp(timestamp_value)
        
    return None

def get_file_modification_time(page, config):
    """获取文件的最后修改时间，优先使用 Git 时间"""
    last_modified = get_git_revision_date(page, config)
    
    if not last_modified:
        # 如果 Git 时间戳找不到，才回退到文件系统时间
        file_path = page.file.abs_src_path
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.fromtimestamp(mtime)

    # 如果两种方法都失败了，提供一个默认值（当前时间）
    if not last_modified:
        last_modified = datetime.now()
        
    return last_modified, bool(last_modified)

def generate_citation(page, config):
    """生成引用指引"""
    title = page.meta.get('title', page.title)
    author = 'Yan Li'
    
    mod_time, _ = get_file_modification_time(page, config)
    
    year = mod_time.year
    month_en = mod_time.strftime('%b')
    day = mod_time.day
    date_display = f"{month_en}. {day}, {year}"
    
    site_url = config.get('site_url', 'https://example.com').rstrip('/')
    page_url = page.url.rstrip('/')
    full_url = f"{site_url}/{page_url}"
    
    page_id = page_url.split('/')[-1] or 'index'
    
    citation = f"""
!!! info "📝 如果您需要引用本文"
    {author}. ({date_display}). {title} [Blog post]. Retrieved from {full_url}

    在 BibTeX 格式中：
    ```text
    @online{{{page_id},
        title={{{title}}},
        author={{{author}}},
        year={{{year}}},
        month={{{month_en}}},
        url={{\\url{{{full_url}}}}},
    }}
    ```
"""
    return citation

# --- MkDocs Hook 函数 ---

def on_page_markdown(markdown, *, page, config, **kwargs):
    """
    在每个 Markdown 页面渲染前执行此函数。
    """
    # 快速排除检查
    if page.meta.get('hide_reading_time', False):
        return markdown
    
    src_path = page.file.src_path
    if any(pattern.match(src_path) for pattern in EXCLUDE_PATTERNS):
        return markdown
    
    page_type = page.meta.get('type', '')
    if page_type in EXCLUDE_TYPES:
        return markdown
    
    # 快速预检查，避免对非常短的文件进行复杂计算
    if len(markdown) < 300:
        # 即使内容短，也可能需要添加引用
        if not page.meta.get('hide_citation', False):
            citation = generate_citation(page, config)
            return markdown + citation
        return markdown

    # 计算统计信息
    reading_time, chinese_chars, code_lines = calculate_reading_stats(markdown)
    
    # 如果中文字符过少，则不显示阅读信息，但可能仍显示引用
    if chinese_chars < 50:
        if not page.meta.get('hide_citation', False):
            citation = generate_citation(page, config)
            return markdown + citation
        return markdown
        
    # --- 构建信息块 ---
    
    # 检查是否有创作声明和是否隐藏信息
    creation_statement = page.meta.get('statement', '')
    no_info = page.meta.get('noinfo', False)

    reading_info_parts = [
        f"阅读时间约 **{reading_time}** 分钟",
        f"约 **{chinese_chars}** 字"
    ]
    if chinese_chars > 10000:
        reading_info_parts.append("⚠️ 万字长文，请慢慢阅读")
    if code_lines > 0:
        reading_info_parts.append(f"约 **{code_lines}** 行代码")
    else:
        reading_info_parts.append("没有代码，请放心食用")

    reading_info_content = "　|　".join(reading_info_parts)
    reading_info = f'!!! info "📖 阅读信息"\n    {reading_info_content}\n\n'

    if creation_statement:
        reading_info += f"    创作声明：{creation_statement}\n\n"

    if no_info:
        reading_info = ""

    # --- 插入信息块到 Markdown ---
    
    # 用正则找到第一个一级标题，并在其后插入阅读信息
    h1_pattern = re.compile(r'(^# .*\n)', re.MULTILINE)
    if h1_pattern.search(markdown):
        markdown = h1_pattern.sub(r'\1' + reading_info, markdown, count=1)
    else:
        # 没有一级标题就插在最前面
        markdown = reading_info + markdown
    
    # --- 添加引用指引 ---
    
    if not page.meta.get('hide_citation', False):
        citation = generate_citation(page, config)
        markdown += citation
    
    return markdown