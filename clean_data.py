"""
数据清洗脚本
对原始翻译数据进行清洗和标准化处理
"""

import os
import re
import argparse
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor

def clean_text(text, lowercase=True, max_length=50):
    """
    增强的文本清洗函数
    
    参数:
        text: 原始文本
        lowercase: 是否转为小写
        max_length: 最大token数（None表示不限制）
    
    返回:
        str: 清洗后的文本或 None（如果过滤）
    """
    if not isinstance(text, str) or not text.strip():
        return None
    
    # 基础清洗
    text = text.strip()
    
    # 移除多余的空白
    text = re.sub(r'\s+', ' ', text)
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 标准化引号
    text = re.sub(r'[""„"‚'']', '"', text)
    text = re.sub(r'[''‛']', "'", text)
    
    # 标准化破折号
    text = re.sub(r'[–—―]', '-', text)
    
    # 移除多个连续的标点符号（保留省略号）
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'([,;:]){2,}', r'\1', text)
    
    # 长度过滤
    if max_length and len(text.split()) > max_length:
        return None
    
    # 转换为小写（可选）
    if lowercase:
        text = text.lower()
    
    # 最终检查
    if len(text.strip()) < 3:  # 太短的句子
        return None
    
    return text

def clean_parallel_data(src_file, tgt_file, src_out, tgt_out, max_length=50, min_length=3):
    """
    清洗平行语料数据
    
    参数:
        src_file: 源语言文件路径
        tgt_file: 目标语言文件路径
        src_out: 清洗后源语言文件路径
        tgt_out: 清洗后目标语言文件路径
        max_length: 最大句子长度（单词数）
        min_length: 最小句子长度（单词数）
    """
    print(f"开始清洗平行数据: {src_file} + {tgt_file}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(src_out) if os.path.dirname(src_out) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(tgt_out) if os.path.dirname(tgt_out) else '.', exist_ok=True)
    
    total_lines = 0
    kept_lines = 0
    seen_pairs = set()  # 用于去重
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
         open(src_out, 'w', encoding='utf-8') as f_src_out, \
         open(tgt_out, 'w', encoding='utf-8') as f_tgt_out:
        
        for src_line, tgt_line in zip(f_src, f_tgt):
            total_lines += 1
            
            # 清洗源语言和目标语言
            clean_src = clean_text(src_line.strip(), lowercase=True, max_length=max_length)
            clean_tgt = clean_text(tgt_line.strip(), lowercase=True, max_length=max_length)
            
            # 检查清洗结果
            if clean_src is None or clean_tgt is None:
                continue
            
            # 长度检查
            src_words = len(clean_src.split())
            tgt_words = len(clean_tgt.split())
            
            if src_words < min_length or tgt_words < min_length:
                continue
            
            if src_words > max_length or tgt_words > max_length:
                continue
            
            # 长度比例检查（避免长度差异过大）
            length_ratio = max(src_words, tgt_words) / min(src_words, tgt_words)
            if length_ratio > 3.0:  # 长度比例不能超过3:1
                continue
            
            # 去重检查
            pair_hash = md5(f"{clean_src}|{clean_tgt}".encode()).hexdigest()
            if pair_hash in seen_pairs:
                continue
            seen_pairs.add(pair_hash)
            
            # 语言特定检查
            if not is_valid_german(clean_src) or not is_valid_english(clean_tgt):
                continue
            
            # 写入清洗后的数据
            f_src_out.write(clean_src + '\n')
            f_tgt_out.write(clean_tgt + '\n')
            kept_lines += 1
    
    print(f"清洗完成: {kept_lines}/{total_lines} 行保留 ({kept_lines/total_lines*100:.1f}%)")

def is_valid_german(text):
    """检查是否为有效的德语文本"""
    # 德语特征检查
    german_chars = set('äöüßÄÖÜ')
    common_german_words = {'der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'ich', 'du', 'er', 'sie', 'es'}
    
    # 检查德语特殊字符
    has_german_chars = any(char in german_chars for char in text)
    
    # 检查常见德语词汇
    words = set(text.lower().split())
    has_german_words = len(words & common_german_words) > 0
    
    # 至少满足一个条件，或者长度较短时放宽要求
    return has_german_chars or has_german_words or len(words) <= 5

def is_valid_english(text):
    """检查是否为有效的英语文本"""
    # 英语常见词汇
    common_english_words = {'the', 'and', 'is', 'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    
    # 检查是否包含非英语字符（排除基本标点）
    non_english_chars = set('äöüßÄÖÜàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ')
    has_non_english = any(char in non_english_chars for char in text)
    
    if has_non_english:
        return False
    
    # 检查常见英语词汇
    words = set(text.lower().split())
    has_english_words = len(words & common_english_words) > 0
    
    return has_english_words or len(words) <= 5

def process_dataset(data_dir, output_dir):
    """处理整个数据集"""
    print(f"开始处理数据集: {data_dir} -> {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有语言对文件
    language_pairs = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.de'):
            base_name = file[:-3]  # 移除 .de 后缀
            en_file = f"{base_name}.en"
            
            if os.path.exists(os.path.join(data_dir, en_file)):
                language_pairs.append(base_name)
    
    print(f"发现 {len(language_pairs)} 个语言对文件")
    
    # 处理每个语言对
    for base_name in language_pairs:
        src_file = os.path.join(data_dir, f"{base_name}.de")
        tgt_file = os.path.join(data_dir, f"{base_name}.en")
        src_out = os.path.join(output_dir, f"{base_name}.de.cleaned")
        tgt_out = os.path.join(output_dir, f"{base_name}.en.cleaned")
        
        clean_parallel_data(src_file, tgt_file, src_out, tgt_out)

def create_combined_training_data(cleaned_dir):
    """合并所有训练数据为一个文件"""
    print("合并训练数据...")
    
    # 查找所有训练文件
    train_files_de = []
    train_files_en = []
    
    for file in os.listdir(cleaned_dir):
        if file.startswith('train') and file.endswith('.de.cleaned'):
            train_files_de.append(file)
            en_file = file.replace('.de.cleaned', '.en.cleaned')
            if os.path.exists(os.path.join(cleaned_dir, en_file)):
                train_files_en.append(en_file)
    
    # 合并文件
    combined_de = os.path.join(cleaned_dir, 'all_train.de')
    combined_en = os.path.join(cleaned_dir, 'all_train.en')
    
    with open(combined_de, 'w', encoding='utf-8') as out_de, \
         open(combined_en, 'w', encoding='utf-8') as out_en:
        
        for de_file, en_file in zip(sorted(train_files_de), sorted(train_files_en)):
            with open(os.path.join(cleaned_dir, de_file), 'r', encoding='utf-8') as f_de, \
                 open(os.path.join(cleaned_dir, en_file), 'r', encoding='utf-8') as f_en:
                
                for de_line, en_line in zip(f_de, f_en):
                    out_de.write(de_line)
                    out_en.write(en_line)
    
    print(f"合并完成: {combined_de}, {combined_en}")

def main():
    parser = argparse.ArgumentParser(description='清洗机器翻译数据')
    parser.add_argument('--data_dir', default='data', help='原始数据目录')
    parser.add_argument('--output_dir', default='cleaned_data', help='输出目录')
    parser.add_argument('--max_length', type=int, default=50, help='最大句子长度')
    parser.add_argument('--min_length', type=int, default=3, help='最小句子长度')
    
    args = parser.parse_args()
    
    # 处理数据集
    process_dataset(args.data_dir, args.output_dir)
    
    # 合并训练数据
    create_combined_training_data(args.output_dir)
    
    print("数据清洗完成!")

if __name__ == "__main__":
    main()
