import re
import os
import json
from functools import lru_cache
from datetime import datetime

# --- 调试开关 ---
# 这个全局变量确保我们只打印一次所有可用的键，避免刷屏
_TIMESTAMP_KEYS_PRINTED = False

# --- 配置 (保持不变) ---
EXCLUDE_PATTERNS = [re.compile(r'^index\.md$'), re.compile(r'^about/'), re.compile(r'^trip/index\.md$'), re.compile(r'^relax/index\.md$'), re.compile(r'^blog/indexblog\.md$'), re.compile(r'^blog/posts\.md$'), re.compile(r'^develop/index\.md$'), re.compile(r'waline\.md$'), re.compile(r'link\.md$'), re.compile(r'404\.md$')]
CHINESE_CHARS_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
CODE_BLOCK_PATTERN = re.compile(r'```.*?```', re.DOTALL)
EXCLUDE_TYPES = frozenset({'landing', 'special', 'widget'})

# --- 核心函数 (只增加调试) ---
@lru_cache(maxsize=256)
def extract_non_code_content(content_hash, markdown):
    return CODE_BLOCK_PATTERN.sub('', markdown)

def count_code_lines(markdown):
    code_blocks = CODE_BLOCK_PATTERN.findall(markdown)
    total_code_lines = 0
    for block in code_blocks:
        lines = block.split('\n')
        code_lines = [line for line in lines if not line.strip().startswith('```') and line.strip()]
        total_code_lines += len(code_lines)
    return total_code_lines

def calculate_reading_stats(markdown):
    content_hash = hash(markdown)
    non_code_content = extract_non_code_content(content_hash, markdown)
    chinese_chars = len(CHINESE_CHARS_PATTERN.findall(non_code_content))
    code_lines = count_code_lines(markdown)
    reading_time = max(1, round(chinese_chars / 200))
    return reading_time, chinese_chars, code_lines

# ###############################################################
# #               ↓ ↓ ↓ 唯一修改的函数在这里 ↓ ↓ ↓                 #
# ###############################################################

@lru_cache(maxsize=1)
def load_timestamps(docs_dir):
    """【调试版本】加载并缓存唯一的 timestamps.json 文件"""
    timestamps_path = os.path.join(docs_dir, 'timestamps.json')
    if not os.path.exists(timestamps_path):
        print(f"--- [DEBUG] --- [!] load_timestamps: File NOT FOUND at '{timestamps_path}'")
        return None
    try:
        with open(timestamps_path, 'r', encoding='utf-8') as f:
            # print(f"--- [DEBUG] --- [+] load_timestamps: File found and loaded from '{timestamps_path}'")
            return json.load(f)
    except Exception as e:
        print(f"--- [DEBUG] --- [!] load_timestamps: EXCEPTION while reading file: {e}")
        return None

def get_git_revision_date(page, config):
    """【调试版本】从唯一的 docs/timestamps.json 中获取时间"""
    global _TIMESTAMP_KEYS_PRINTED
    from datetime import datetime

    print("\n--- [DEBUG] Entering get_git_revision_date ---")
    
    # 1. 打印我们用来查找的 KEY 是什么
    relative_path_key = page.file.src_path.replace(os.path.sep, '/')
    print(f"[*] Page being processed: '{page.title}' (from file: {page.file.src_path})")
    print(f"[*] Generated lookup KEY: '{relative_path_key}'")

    # 2. 加载数据
    docs_dir = config['docs_dir']
    timestamps = load_timestamps(docs_dir)
    
    if timestamps is None:
        print("[!] FAILURE: timestamps.json could not be loaded or was not found. Aborting lookup.")
        print("--- [DEBUG] Exiting ---\n")
        return None

    # 3. 打印 JSON 文件中所有可用的 KEY (只打印一次)
    if not _TIMESTAMP_KEYS_PRINTED:
        print("[*] All available KEYS in timestamps.json (first 20 shown):")
        # 只显示前20个，避免刷屏
        print(sorted(list(timestamps.keys()))[:20])
        print(f"    ... (total {len(timestamps.keys())} keys)")
        _TIMESTAMP_KEYS_PRINTED = True

    # 4. 进行查找并报告结果
    if relative_path_key in timestamps:
        timestamp_value = timestamps[relative_path_key]
        print(f"[+] SUCCESS: Key '{relative_path_key}' found!")
        print("--- [DEBUG] Exiting ---\n")
        return datetime.fromtimestamp(timestamp_value)
    else:
        print(f"[!] FAILURE: Key '{relative_path_key}' was NOT FOUND in the available keys.")
        print("--- [DEBUG] Exiting ---\n")
        return None
        
# ###############################################################
# #               ↑ ↑ ↑ 唯一的修改到此为止 ↑ ↑ ↑                     #
# ###############################################################


def get_file_modification_time(page, config):
    last_modified = get_git_revision_date(page, config)
    if not last_modified:
        file_path = page.file.abs_src_path
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            last_modified = datetime.fromtimestamp(mtime)
    if not last_modified:
        last_modified = datetime.now()
    return last_modified, bool(last_modified)

def generate_citation(page, config):
    title = page.meta.get('title', page.title)
    author = 'Yan Li'
    mod_time, state = get_file_modification_time(page, config)
    year = mod_time.year
    month_en = mod_time.strftime('%b')
    day = mod_time.day
    date_display = f"{month_en}. {day}, {year}"
    site_url = 'https://dicaeopolis.github.io/'.rstrip('/')
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

def on_page_markdown(markdown, **kwargs):
    page = kwargs['page']
    config = kwargs['config']
    if page.meta.get('hide_reading_time', False): return markdown
    src_path = page.file.src_path
    for pattern in EXCLUDE_PATTERNS:
        if pattern.match(src_path): return markdown
    page_type = page.meta.get('type', '')
    if page_type in EXCLUDE_TYPES: return markdown
    if len(markdown) < 300: return markdown
    reading_time, chinese_chars, code_lines = calculate_reading_stats(markdown)
    creation_statement = page.meta.get('statement', '')
    no_info = page.meta.get('noinfo', '')
    reading_info = f"""!!! info "📖 阅读信息"
    阅读时间约 **{reading_time}** 分钟　|　约 **{chinese_chars}** 字"""
    if chinese_chars > 10000: reading_info += f"""　⚠️ 万字长文，请慢慢阅读"""
    if code_lines > 0: reading_info += f"""　|　约 **{code_lines}** 行代码

"""
    else: reading_info += f"""　|　没有代码，请放心食用

"""
    if creation_statement: reading_info += f"    创作声明：{creation_statement}\n\n"
    if no_info: reading_info = ""
    pattern = r'(^# .*\n)'
    if re.search(pattern, markdown, flags=re.MULTILINE): markdown = re.sub(pattern, r'\1' + reading_info, markdown, count=1, flags=re.MULTILINE)
    else: markdown = reading_info + markdown
    if chinese_chars < 50: return markdown
    if not page.meta.get('hide_citation', False): markdown += generate_citation(page, config)
    return markdown