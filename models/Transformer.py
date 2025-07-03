"""
Transformer 主模型文件
整合编码器、解码器和完整的 Seq2Seq Transformer 模型
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到 Python 路径以确保正确导入
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.Encoder import Encoder
from models.Decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    这是一个包装类，用于兼容现有的训练和推理代码
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8, 
                 ffn_hidden_dim=2048, num_layers=6, dropout=0.1, padding_idx=1,
                 norm_first=True, activation='relu'):  # 默认使用norm_first=True
        super().__init__()
        
        # 创建编码器和解码器
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            norm_first=norm_first,
            activation=activation
        )
        
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            padding_idx=padding_idx,
            norm_first=norm_first,
            activation=activation
        )
        
        # 保存参数以便调试
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_idx = padding_idx
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                return_attention=False):
        """
        前向传播
        Args:
            src: 源序列 (batch, src_len)
            tgt: 目标序列 (batch, tgt_len)
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            src_key_padding_mask: 源序列填充掩码
            tgt_key_padding_mask: 目标序列填充掩码
            return_attention: 是否返回注意力权重
        Returns:
            output: (batch, tgt_len, tgt_vocab_size)
            attention_weights: 注意力权重 (可选)
        """
        # 编码
        if return_attention:
            memory, enc_attention = self.encoder(
                src, src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            # 解码
            output, dec_attention = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                return_attention=True
            )
            return output, {
                'encoder_attention': enc_attention,
                'decoder_self_attention': dec_attention[0],
                'decoder_cross_attention': dec_attention[1]
            }
        else:
            memory = self.encoder(
                src, src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                return_attention=False
            )
            # 解码
            output = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask,
                return_attention=False
            )
            return output


class Seq2SeqTransformer(nn.Module):
    """
    Seq2Seq Transformer 模型
    与 train.py 中的模型保持兼容
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
        """
        前向传播 - 与训练时的接口保持一致
        """
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return output
