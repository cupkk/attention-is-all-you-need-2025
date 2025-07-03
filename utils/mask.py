import torch

def generate_square_subsequent_mask(sz):
    """
    生成大小为 (sz, sz) 的方形后续掩码矩阵。
    下三角区域为 False，上三角（表示未来位置）为 True，以屏蔽解码器中未来的词。
    """
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return mask

def create_mask(src, tgt, pad_idx):
    """
    为源序列和目标序列生成掩码。
    返回: src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    """
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)
    # 源序列不需要未来掩码，使用全 False 的矩阵
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool)
    # 目标序列的后续掩码
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    # 填充掩码：标记出输入中为 <pad> 的位置（值为 True 表示需要被mask）
    src_padding_mask = (src == pad_idx)
    tgt_padding_mask = (tgt == pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
