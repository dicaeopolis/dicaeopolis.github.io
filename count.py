import os
import re

def count_chinese_characters_in_file(file_path):
    """统计单个文件中的汉字数量"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # 使用正则表达式匹配汉字（Unicode范围：\u4e00-\u9fff）
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
            
            return len(chinese_chars)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0

def count_chinese_in_md_files(folder_path='docs'):
    """统计文件夹中所有.md文件的汉字总数"""
    total_chinese_count = 0
    file_count = 0
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    print(f"正在扫描文件夹: {folder_path}")
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                chinese_count = count_chinese_characters_in_file(file_path)
                
                total_chinese_count += chinese_count
                file_count += 1
                
                print(f"文件: {file_path} - 汉字数: {chinese_count}")
    
    # 输出统计结果
    print("\n" + "="*50)
    print(f"扫描完成！")
    print(f"共处理 {file_count} 个 .md 文件")
    print(f"总汉字数: {total_chinese_count}")
    print("="*50)
    
    return total_chinese_count

# 增强版：显示文件大小和统计详情
def detailed_count_chinese_in_md_files(folder_path='docs'):
    """详细统计，包含文件大小等信息"""
    total_chinese_count = 0
    total_file_size = 0
    file_count = 0
    file_stats = []
    
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    print(f"正在详细扫描文件夹: {folder_path}\n")
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                
                try:
                    # 获取文件大小
                    file_size = os.path.getsize(file_path)
                    
                    # 读取内容并统计汉字
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    chinese_chars = re.findall(r'[\u4e00-\u9fff]', content)
                    chinese_count = len(chinese_chars)
                    
                    total_chinese_count += chinese_count
                    total_file_size += file_size
                    file_count += 1
                    
                    file_stats.append({
                        'path': file_path,
                        'chinese_count': chinese_count,
                        'file_size': file_size
                    })
                    
                    print(f"📄 {file}")
                    print(f"   路径: {file_path}")
                    print(f"   大小: {file_size} 字节")
                    print(f"   汉字数: {chinese_count}")
                    print(f"   {'─'*40}")
                    
                except Exception as e:
                    print(f"❌ 处理文件 {file_path} 时出错: {e}")
    
    # 输出详细统计结果
    print("\n" + "="*60)
    print("📊 统计摘要")
    print("="*60)
    print(f"📁 扫描文件夹: {folder_path}")
    print(f"📄 文件数量: {file_count} 个 .md 文件")
    print(f"💾 总文件大小: {total_file_size} 字节 ({total_file_size/1024:.2f} KB)")
    print(f"🔤 总汉字数: {total_chinese_count}")
    
    if file_stats:
        # 找出汉字最多的文件
        max_chinese_file = max(file_stats, key=lambda x: x['chinese_count'])
        print(f"🏆 汉字最多的文件: {max_chinese_file['path']}")
        print(f"   汉字数: {max_chinese_file['chinese_count']}")
    
    print("="*60)
    
    return total_chinese_count

if __name__ == "__main__":
    # 使用基本版本
    print("基本版本统计：")
    count_chinese_in_md_files('docs')
    
    print("\n\n" + "★"*60 + "\n")
    
    # 使用详细版本
    print("详细版本统计：")
    detailed_count_chinese_in_md_files('docs')
