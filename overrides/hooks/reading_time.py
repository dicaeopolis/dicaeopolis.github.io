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

def count_fomula(text: str) -> int:
    """
    统计文本中以 $...$（行内）与 $$...$$（行间）包裹的 LaTeX 公式数量。
    规则与处理：
      - 忽略代码块内容：支持 ``` 与 ~~~ 围栏代码块（fenced code block）
      - 忽略行内代码：`...`（支持任意数量反引号作为定界符）
      - 处理转义：被反斜杠转义的 $（如 \$）不作为定界符
      - $$...$$ 可跨行，$...$ 也可跨行（若未闭合则不计数）
      - 仅统计使用 $ 或 $$ 作为定界符的公式，其他如 KATEX_INLINE_OPEN KATEX_INLINE_CLOSE 不计

    参数:
        text: 原始 Markdown 文本

    返回:
        int: 公式总数
    """
    n = len(text)
    i = 0
    count = 0

    in_fenced = False
    fence_char = ''
    fence_len = 0

    in_inline_code = False
    inline_tick_len = 0

    in_math_inline = False
    in_math_display = False

    def is_escaped(pos: int) -> bool:
        # 判断 text[pos] 是否被奇数个反斜杠转义
        bs = 0
        j = pos - 1
        while j >= 0 and text[j] == '\\':
            bs += 1
            j -= 1
        return (bs % 2) == 1

    while i < n:
        sol = (i == 0 or text[i - 1] == '\n')  # start of line

        # 1) 已在围栏代码块中：仅在行首检查关闭围栏
        if in_fenced:
            if sol:
                j = i
                # 跳过最多 3 个空白（CommonMark 允许最多 3 个缩进）
                spaces = 0
                while j < n and text[j] in ' \t' and spaces < 3:
                    j += 1
                    spaces += 1
                if j < n and text[j] == fence_char:
                    k = j
                    while k < n and text[k] == fence_char:
                        k += 1
                    if (k - j) >= fence_len:
                        # 关闭围栏：跳到本行行尾
                        while k < n and text[k] != '\n':
                            k += 1
                        i = k + 1 if k < n else k
                        in_fenced = False
                        continue
            # 未遇到关闭围栏，逐字符跳过
            i += 1
            continue

        # 2) 行首检查开启围栏代码块
        if not in_inline_code and not in_math_inline and not in_math_display and sol:
            j = i
            spaces = 0
            while j < n and text[j] in ' \t' and spaces < 3:
                j += 1
                spaces += 1
            if j < n and text[j] in ('`', '~'):
                c = text[j]
                k = j
                while k < n and text[k] == c:
                    k += 1
                run = k - j
                if run >= 3:
                    # 开启围栏：记录围栏信息并跳到本行末
                    in_fenced = True
                    fence_char = c
                    fence_len = run
                    while k < n and text[k] != '\n':
                        k += 1
                    i = k + 1 if k < n else k
                    continue

        # 3) 行内代码 `...`（支持可变数量反引号）
        if in_inline_code:
            if text[i] == '`':
                k = i
                while k < n and text[k] == '`':
                    k += 1
                run = k - i
                if run == inline_tick_len:
                    in_inline_code = False
                    i = k
                    continue
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue
        else:
            if text[i] == '`' and not in_math_inline and not in_math_display:
                k = i
                while k < n and text[k] == '`':
                    k += 1
                inline_tick_len = k - i
                in_inline_code = True
                i = k
                continue

        # 4) 数学模式
        if not in_math_inline and not in_math_display:
            if text[i] == '$' and not is_escaped(i):
                # 统计连续 $ 的个数
                k = i
                while k < n and text[k] == '$':
                    k += 1
                run = k - i
                if run >= 2:
                    # $$ 开启行间数学
                    in_math_display = True
                    i += 2
                    continue
                else:
                    # $ 开启行内数学
                    in_math_inline = True
                    i += 1
                    continue
            else:
                i += 1
                continue

        # 行内数学：寻找单个未转义的 $ 作为闭合
        if in_math_inline:
            if text[i] == '$' and not is_escaped(i):
                in_math_inline = False
                count += 1
                i += 1
                continue
            else:
                i += 1
                continue

        # 行间数学：寻找未转义的 $$ 作为闭合
        if in_math_display:
            if text[i] == '$' and not is_escaped(i):
                if i + 1 < n and text[i + 1] == '$':
                    in_math_display = False
                    count += 1
                    i += 2
                    continue
                else:
                    # 单个 $ 在行间数学内视作普通字符
                    i += 1
                    continue
            else:
                i += 1
                continue

    return count

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
    fomula_count = count_fomula(markdown)
    creation_statement = page.meta.get('statement', '')
    no_info = page.meta.get('noinfo', '')
    reading_info = f"""!!! info "📖 阅读信息"
    阅读时间约 **{reading_time+fomula_count//10+code_lines//100}** 分钟　|　约 **{chinese_chars}** 字"""
    if chinese_chars > 10000: reading_info += f"""　⚠️ 万字长文，请慢慢阅读"""
    if fomula_count > 0: reading_info += f"""　|　约 **{fomula_count}** 个公式"""
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