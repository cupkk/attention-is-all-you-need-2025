"""
构建BPE词汇表
"""

from torchtext.vocab import build_vocab_from_iterator
import torch
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def yield_tokens(file_path):
    """
    逐行读取 tokenized 文件并生成 token 列表
    
    参数:
        file_path: 文件路径（如 train_part1.tok.de）
    
    返回:
        generator: 每次yield一个token列表
    """
    logger.info(f"读取文件: {file_path}")
    
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                tokens = line.split()
                yield tokens
            
            if line_num % 10000 == 0:
                logger.info(f"已处理 {line_num} 行")

def collect_all_tokens(file_paths):
    """
    从多个文件收集所有tokens
    """
    all_tokens = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            logger.info(f"收集tokens: {file_path}")
            for tokens in yield_tokens(file_path):
                all_tokens.extend(tokens)
        else:
            logger.warning(f"跳过不存在的文件: {file_path}")
    
    return all_tokens

def build_vocabulary(token_files, min_freq=2, max_tokens=32000):
    """
    构建词汇表
    
    参数:
        token_files: tokenized文件列表
        min_freq: 最小出现频率
        max_tokens: 最大词汇数量
    
    返回:
        vocab: 词汇表对象
    """
    logger.info(f"开始构建词汇表，最小频率: {min_freq}, 最大词汇数: {max_tokens}")
    
    # 定义特殊token
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    
    # 收集所有tokens
    all_tokens = collect_all_tokens(token_files)
    logger.info(f"总共收集到 {len(all_tokens)} 个tokens")
    
    # 统计词频
    token_counter = Counter(all_tokens)
    logger.info(f"唯一tokens数量: {len(token_counter)}")
    
    # 过滤低频词
    filtered_tokens = {token: count for token, count in token_counter.items() 
                      if count >= min_freq}
    logger.info(f"过滤后tokens数量: {len(filtered_tokens)}")
    
    # 按频率排序并限制词汇表大小
    sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
    if max_tokens:
        sorted_tokens = sorted_tokens[:max_tokens - len(special_tokens)]
    
    # 创建token迭代器
    def token_iterator():
        for token, _ in sorted_tokens:
            yield [token]
    
    # 构建词汇表
    vocab = build_vocab_from_iterator(
        token_iterator(),
        specials=special_tokens,
        special_first=True
    )
    
    # 设置默认索引为<unk>
    vocab.set_default_index(vocab['<unk>'])
    
    logger.info(f"词汇表构建完成，大小: {len(vocab)}")
    
    # 显示词频统计
    top_tokens = sorted_tokens[:20]
    logger.info("Top 20 tokens:")
    for token, count in top_tokens:
        logger.info(f"  {token}: {count}")
    
    return vocab

def save_vocab_as_dict(vocab, output_path):
    """
    将vocab保存为简单的dict格式以便translate_api.py使用
    """
    vocab_dict = {}
    
    # 获取所有tokens
    if hasattr(vocab, 'get_itos'):
        # torchtext 0.10+
        tokens = vocab.get_itos()
        for idx, token in enumerate(tokens):
            vocab_dict[token] = idx
    elif hasattr(vocab, 'itos'):
        # torchtext < 0.10
        for idx, token in enumerate(vocab.itos):
            vocab_dict[token] = idx
    else:
        # 其他情况，尝试从stoi获取
        vocab_dict = dict(vocab.get_stoi()) if hasattr(vocab, 'get_stoi') else dict(vocab.stoi)
    
    # 保存为字典
    torch.save(vocab_dict, output_path)
    logger.info(f"词汇表已保存为字典格式: {output_path}")
    
    return vocab_dict

def main():
    """主函数"""
    # 设置路径
    data_dir = "multi30k_processed_bpe"
    
    # 德语词汇表
    logger.info("构建德语词汇表...")
    de_files = [
        os.path.join(data_dir, "train_part1.tok.de"),
        os.path.join(data_dir, "val.tok.de"),
    ]
    
    vocab_de = build_vocabulary(de_files, min_freq=2, max_tokens=32000)
    de_dict = save_vocab_as_dict(vocab_de, os.path.join(data_dir, "vocab_de.pth"))
    
    # 英语词汇表
    logger.info("构建英语词汇表...")
    en_files = [
        os.path.join(data_dir, "train_part1.tok.en"),
        os.path.join(data_dir, "val.tok.en"),
    ]
    
    vocab_en = build_vocabulary(en_files, min_freq=2, max_tokens=32000)
    en_dict = save_vocab_as_dict(vocab_en, os.path.join(data_dir, "vocab_en.pth"))
    
    # 显示词汇表信息
    logger.info(f"德语词汇表大小: {len(de_dict)}")
    logger.info(f"英语词汇表大小: {len(en_dict)}")
    
    # 验证特殊token
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    for token in special_tokens:
        if token in de_dict and token in en_dict:
            logger.info(f"特殊token {token}: DE={de_dict[token]}, EN={en_dict[token]}")
        else:
            logger.warning(f"特殊token {token} 缺失!")
    
    logger.info("词汇表构建完成!")

if __name__ == "__main__":
    main()
