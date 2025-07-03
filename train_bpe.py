import sentencepiece as spm
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_bpe(input_file, model_prefix, vocab_size=32000):
    """
    训练 BPE 分词器
    
    参数:
        input_file: 输入文本文件路径（每行一句）
        model_prefix: 输出模型前缀（如 bpe_de）
        vocab_size: 词表大小（建议 32000）
    """
    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            character_coverage=1.0,
            max_sentence_length=4192000
        )
        logger.info(f"✅ BPE 分词器已训练完成：{model_prefix}.model")
    except Exception as e:
        logger.error(f"BPE 分词器训练失败: {str(e)}")
        raise

def merge_train_files(train_files, output_path):
    """
    合并多个训练文件到一个文件
    
    参数:
        train_files: 训练文件列表
        output_path: 输出文件路径
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        for fname in train_files:
            file_path = os.path.join(cleaned_dir, fname)
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 文件不存在: {file_path}")
                continue
            logger.info(f"Reading {fname} ...")
            with open(file_path, "r", encoding="utf-8") as fin:
                fout.write(fin.read())
    logger.info(f"📚 已合并所有训练数据到 {output_path}")

if __name__ == "__main__":
    cleaned_dir = "cleaned_data"
    bpe_dir = "bpe_models"
    os.makedirs(bpe_dir, exist_ok=True)

    # 合并所有训练数据作为 BPE 模型训练语料
    all_train_path = os.path.join(cleaned_dir, "all_train.txt")

    # 支持多个训练文件
    train_files = [
        "train_part1.de.cleaned",
        "train_part1.en.cleaned",
        "validation.de.cleaned",
        "validation.en.cleaned",
        "test.de.cleaned",
        "test.en.cleaned"
    ]

    merge_train_files(train_files, all_train_path)

    # 训练德语 BPE 分词器
    train_bpe(
        input_file=all_train_path,
        model_prefix=os.path.join(bpe_dir, "bpe_de"),
        vocab_size=32000
    )

    # 训练英语 BPE 分词器
    train_bpe(
        input_file=all_train_path,
        model_prefix=os.path.join(bpe_dir, "bpe_en"),
        vocab_size=32000
    )

    logger.info("✅ BPE 分词器训练完成，路径:", bpe_dir)