"""
使用BPE对数据进行分词处理
"""

import sentencepiece as spm
import os
import logging
import argparse

def tokenize_parallel_with_bpe(src_in, tgt_in, src_out, tgt_out, sp_src, sp_tgt, max_length=50, min_length=3):
    """
    保证输出的 src_out 和 tgt_out 行数严格一致。
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(src_out), exist_ok=True)
    os.makedirs(os.path.dirname(tgt_out), exist_ok=True)

    kept = 0
    total = 0
    with open(src_in, "r", encoding="utf-8") as fsrc, \
         open(tgt_in, "r", encoding="utf-8") as ftgt, \
         open(src_out, "w", encoding="utf-8") as fsrc_out, \
         open(tgt_out, "w", encoding="utf-8") as ftgt_out:

        for src_line, tgt_line in zip(fsrc, ftgt):
            total += 1
            src_clean = src_line.strip()
            tgt_clean = tgt_line.strip()

            if not src_clean or not tgt_clean:
                continue

            # 使用SentencePiece分词
            src_tokens = sp_src.encode_as_pieces(src_clean)
            tgt_tokens = sp_tgt.encode_as_pieces(tgt_clean)

            # 长度过滤
            if len(src_tokens) < min_length or len(src_tokens) > max_length:
                continue
            if len(tgt_tokens) < min_length or len(tgt_tokens) > max_length:
                continue

            # 长度比例检查
            length_ratio = max(len(src_tokens), len(tgt_tokens)) / min(len(src_tokens), len(tgt_tokens))
            if length_ratio > 2.5:
                continue

            # 写入分词结果
            fsrc_out.write(" ".join(src_tokens) + "\n")
            ftgt_out.write(" ".join(tgt_tokens) + "\n")
            kept += 1

    logger.info(f"分词完成: {src_in} -> {src_out}, 保留 {kept}/{total} 行")
    return kept

def process_all_files():
    """处理所有数据文件"""
    # 加载BPE模型
    sp_de = spm.SentencePieceProcessor()
    sp_de.Load("bpe_models/bpe_de.model")
    
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load("bpe_models/bpe_en.model")
    
    # 创建输出目录
    output_dir = "multi30k_processed_bpe"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义文件映射
    file_pairs = [
        ("cleaned_data/train_part1.de.cleaned", "cleaned_data/train_part1.en.cleaned", 
         f"{output_dir}/train_part1.tok.de", f"{output_dir}/train_part1.tok.en"),
        ("cleaned_data/val.de.cleaned", "cleaned_data/val.en.cleaned",
         f"{output_dir}/val.tok.de", f"{output_dir}/val.tok.en"),
        ("cleaned_data/test.de.cleaned", "cleaned_data/test.en.cleaned",
         f"{output_dir}/test.tok.de", f"{output_dir}/test.tok.en"),
    ]
    
    total_kept = 0
    for src_in, tgt_in, src_out, tgt_out in file_pairs:
        if os.path.exists(src_in) and os.path.exists(tgt_in):
            kept = tokenize_parallel_with_bpe(src_in, tgt_in, src_out, tgt_out, sp_de, sp_en)
            total_kept += kept
        else:
            print(f"跳过不存在的文件: {src_in} 或 {tgt_in}")
    
    print(f"总共处理了 {total_kept} 对句子")

def test_bpe_models():
    """测试BPE模型"""
    try:
        sp_de = spm.SentencePieceProcessor()
        sp_de.Load("bpe_models/bpe_de.model")
        
        sp_en = spm.SentencePieceProcessor()
        sp_en.Load("bpe_models/bpe_en.model")
        
        # 测试德语分词
        test_de = "Ein Mann sitzt auf einer Bank."
        tokens_de = sp_de.encode_as_pieces(test_de)
        print(f"德语测试: {test_de}")
        print(f"分词结果: {tokens_de}")
        
        # 测试英语分词
        test_en = "A man sits on a bench."
        tokens_en = sp_en.encode_as_pieces(test_en)
        print(f"英语测试: {test_en}")
        print(f"分词结果: {tokens_en}")
        
        return True
    except Exception as e:
        print(f"BPE模型测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='使用BPE对数据进行分词')
    parser.add_argument('--test', action='store_true', help='只测试BPE模型')
    
    args = parser.parse_args()
    
    if args.test:
        test_bpe_models()
    else:
        # 首先测试模型
        if test_bpe_models():
            print("BPE模型测试通过，开始处理数据...")
            process_all_files()
        else:
            print("BPE模型测试失败，请先训练BPE模型")

if __name__ == "__main__":
    main()
