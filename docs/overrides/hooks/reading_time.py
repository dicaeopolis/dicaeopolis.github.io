import re
from functools import lru_cache

# 预编译正则表达式
# 在这里排除不需要统计的文件：
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

@lru_cache(maxsize=256)
def extract_non_code_content(content_hash, markdown):
    """提取不在代码块中的内容"""
    # 移除所有代码块
    content = CODE_BLOCK_PATTERN.sub('', markdown)
    return content

def count_code_lines(markdown):
    """统计代码行数 - 简化版本，只计算代码块中的行数"""
    code_blocks = CODE_BLOCK_PATTERN.findall(markdown)
    total_code_lines = 0
    
    for block in code_blocks:
        # 移除代码块的标记行（开头和结尾的```）
        lines = block.split('\n')
        
        # 过滤掉代码块标记行和空行
        code_lines = [
            line for line in lines 
            if not line.strip().startswith('```') and line.strip()
        ]
        
        total_code_lines += len(code_lines)
    
    return total_code_lines

def calculate_reading_stats(markdown):
    """计算中文字符数和代码行数"""
    # 生成内容哈希用于缓存
    content_hash = hash(markdown)
    
    # 提取不在代码块中的内容
    non_code_content = extract_non_code_content(content_hash, markdown)
    
    # 统计中文字符（不在代码块中的）
    chinese_chars = len(CHINESE_CHARS_PATTERN.findall(non_code_content))
    
    # 统计代码行数
    code_lines = count_code_lines(markdown)
    
    # 计算阅读时间（中文：200字/分钟）
    reading_time = max(1, round(chinese_chars / 200))
    
    return reading_time, chinese_chars, code_lines

def on_page_markdown(markdown, **kwargs):
    page = kwargs['page']
    
    # 快速排除检查
    if page.meta.get('hide_reading_time', False):
        return markdown
    
    # 保持原有的EXCLUDE_PATTERNS循环检查方式
    src_path = page.file.src_path
    for pattern in EXCLUDE_PATTERNS:
        if pattern.match(src_path):
            return markdown
    
    # 优化类型检查
    page_type = page.meta.get('type', '')
    if page_type in EXCLUDE_TYPES:
        return markdown
    
    # 快速预检查
    if len(markdown) < 300:
        return markdown
    
    # 计算统计信息
    reading_time, chinese_chars, code_lines = calculate_reading_stats(markdown)
    
    # 过滤太短的内容
    if chinese_chars < 50:
        return markdown
    
    # 生成阅读信息
    
    reading_info = f"""!!! info "📖 阅读信息"
阅读时间约 **{reading_time}** 分钟　|　约 **{chinese_chars}** 字"""
    
    if chinese_chars > 10000:
        reading_info += f"""　⚠️⚠️⚠️万字长文，请慢慢阅读"""

    if code_lines > 0:
        reading_info += f"""　|　约 **{code_lines}** 行代码

"""
    else:
        reading_info += f"""　|　没有代码，请放心食用

"""

    # 用正则找到第一个一级标题，并在其后插入阅读信息
    pattern = r'(^# .*\n)'
    if re.search(pattern, markdown, flags=re.MULTILINE):
        markdown = re.sub(pattern, r'\1' + reading_info, markdown, count=1, flags=re.MULTILINE)
        return markdown
    else:
        # 没有一级标题就插在最前面
        return reading_info + markdown