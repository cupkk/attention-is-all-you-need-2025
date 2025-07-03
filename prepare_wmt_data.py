import pandas as pd
import os
import re
from hashlib import md5
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text, lowercase=True, max_length=50, multilingual=False):
    """
    清洗单句文本，并统计被过滤的句子数量
    
    参数:
        text (str): 原始句子
        lowercase (bool): 是否转为小写
        max_length (int): 最大 token 长度
        multilingual (bool): 是否启用多语言支持
    
    返回:
        str: 清洗后的句子或 None（如果超过长度）
    """

    if not isinstance(text, str) or not text.strip():
        return None
    

    # 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)

    # 保留更多符号（包括括号类和数学符号）
    if multilingual:
        pattern = r'[^\w\s.,!?;:\'"$€£¥¢¡¿¿«»《》…+\-×÷\u0080-\U0010ffff]'
    else:
        pattern = r'[^\w\s.,!?;:\'"$€£¥¢¡¿+\-×÷]'
        
    text = re.sub(pattern, '', text)
    
    # 规范空白符
    text = ' '.join(text.strip().split())
    
    # 小写转换
    if lowercase:
        text = text.lower()
        
    # 控制最大长度
    if max_length and len(text.split()) > max_length:
        return None
    
    return text


def process_parquet_file(parquet_path, output_de, output_en, 
                        max_length=50, lowercase=True, multilingual=False):
    """
    处理 .parquet 文件，提取 de/en 句子并清洗保存为 .de/.en 文件
    
    参数:
        parquet_path: 输入 Parquet 文件路径
        output_de: 输出德语文件路径
        output_en: 输出英语文件路径
        max_length: 最大句子长度
        lowercase: 是否统一转为小写
        multilingual: 是否启用多语言支持
    """
    logger.info(f"📖 正在处理: {parquet_path}")
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"读取失败: {parquet_path} - {str(e)}")
        return
        
    # 自动检测翻译字段
    if 'translation' in df.columns:
        de_sentences = [item['de'] if isinstance(item, dict) else '' for item in df['translation']]
        en_sentences = [item['en'] if isinstance(item, dict) else '' for item in df['translation']]
    else:
        logger.error(f"❌ 缺失 translation 字段: {parquet_path}")
        return

    logger.info(f"共读取 {len(de_sentences)} 对句子")
    logger.info(f"开始清洗，最大长度: {max_length}, 小写化: {lowercase}, 多语言: {multilingual}")

    # 创建输出目录
    Path(output_de).parent.mkdir(parents=True, exist_ok=True)
    Path(output_en).parent.mkdir(parents=True, exist_ok=True)

    counter = {'total': 0, 'too_long': 0, 'valid': 0}
    seen_hashes = set()

    with open(output_de, "w", encoding="utf-8") as f_de, \
         open(output_en, "w", encoding="utf-8") as f_en:

        for de_line, en_line in zip(de_sentences, en_sentences):
            counter['total'] += 1
            
            # 清洗文本
            de_cleaned = clean_text(de_line, lowercase=lowercase, 
                                 max_length=max_length, multilingual=multilingual)
            en_cleaned = clean_text(en_line, lowercase=lowercase, 
                                 max_length=max_length, multilingual=multilingual)
            
            # 空句或清洗失败
            if not de_cleaned or not en_cleaned:
                counter['too_long'] += 1
                continue
                
            # 去重逻辑
            pair_hash = md5(f"{de_cleaned}|||{en_cleaned}".encode()).hexdigest()
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)
            
            # 写入文件
            f_de.write(de_cleaned + "\n")
            f_en.write(en_cleaned + "\n")
            counter['valid'] += 1
            
            # 显示进度
            if counter['total'] % 10000 == 0:
                logger.debug(f"已处理 {counter['total']} 句，有效 {counter['valid']} 句")

    logger.info(f"✅ 清洗完成: {parquet_path} -> {output_de} & {output_en}")
    logger.info(f"📊 统计: 总={counter['total']} 有效={counter['valid']} 过长={counter['too_long']}")


if __name__ == "__main__":
    data_dir = "data"
    output_dir = "data_dir"
    
    # 自动检测所有 Parquet 文件
    import glob
    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理训练集（仅处理 train_part1）
    for i in range(1, 2):  # 仅处理 train_part1
        train_parquet = os.path.join(data_dir, f"train_part{i}.parquet")
        if not os.path.exists(train_parquet):
            logger.warning(f"⚠️ 跳过缺失文件: {train_parquet}")
            continue
            
        process_parquet_file(
            train_parquet,
            os.path.join(output_dir, f"train_part{i}.de"),
            os.path.join(output_dir, f"train_part{i}.en"),
            max_length=50,
            lowercase=True,
            multilingual=True
        )
    
    # 处理验证集
    val_parquet = os.path.join(data_dir, "validation.parquet")
    if os.path.exists(val_parquet):
        process_parquet_file(
            val_parquet,
            os.path.join(output_dir, "validation.de"),
            os.path.join(output_dir, "validation.en"),
            max_length=50,
            lowercase=True,
            multilingual=True
        )
    
    # 处理测试集（确保清洗逻辑一致）
    test_parquet = os.path.join(data_dir, "test.parquet")
    if os.path.exists(test_parquet):
        process_parquet_file(
            test_parquet,
            os.path.join(output_dir, "test.de"),
            os.path.join(output_dir, "test.en"),
            max_length=50,
            lowercase=True,
            multilingual=True
        )
