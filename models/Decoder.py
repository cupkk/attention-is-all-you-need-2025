import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PositionalEncoding import PositionalEncoding
from models.MultiHeadAttention import MultiHeadAttention

class DecoderLayer(nn.Module):
    """
    优化的解码器层
    - 支持Pre-LayerNorm和Post-LayerNorm
    - 添加GELU激活函数选项
    - 改进的权重初始化
    """
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout, 
                 activation='relu', norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        
        # 注意力层
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout=dropout)
        self.multihead_attn = MultiHeadAttention(embed_dim, num_heads, attn_dropout=dropout)  # 交叉注意力
        
        # 前馈网络
        self.linear1 = nn.Linear(embed_dim, ffn_hidden_dim)
        self.linear2 = nn.Linear(ffn_hidden_dim, embed_dim)
        
        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.relu if activation == 'relu' else F.gelu
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, return_attention=False):
        """
        支持Pre-LayerNorm和Post-LayerNorm的前向传播
        """
        if self.norm_first:
            # Pre-LayerNorm
            # 掩码自注意力
            norm_x = self.norm1(x)
            self_attn_out, self_attn_weights = self.self_attn(
                norm_x, norm_x, norm_x, 
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask,
                return_attention=return_attention
            )
            x = x + self.dropout1(self_attn_out)
            
            # 编码器-解码器注意力
            norm_x = self.norm2(x)
            cross_attn_out, cross_attn_weights = self.multihead_attn(
                norm_x, memory, memory, 
                attn_mask=None, 
                key_padding_mask=memory_key_padding_mask,
                return_attention=return_attention
            )
            x = x + self.dropout2(cross_attn_out)
            
            # 前馈网络
            norm_x = self.norm3(x)
            ff_out = self.linear2(self.activation(self.linear1(norm_x)))
            x = x + self.dropout3(ff_out)
        else:
            # Post-LayerNorm (原论文)
            # 掩码自注意力
            self_attn_out, self_attn_weights = self.self_attn(
                x, x, x, 
                attn_mask=tgt_mask, 
                key_padding_mask=tgt_key_padding_mask,
                return_attention=return_attention
            )
            x = x + self.dropout1(self_attn_out)
            x = self.norm1(x)
            
            # 编码器-解码器注意力
            cross_attn_out, cross_attn_weights = self.multihead_attn(
                x, memory, memory, 
                attn_mask=None, 
                key_padding_mask=memory_key_padding_mask,
                return_attention=return_attention
            )
            x = x + self.dropout2(cross_attn_out)
            x = self.norm2(x)
            
            # 前馈网络
            ff_out = self.linear2(self.activation(self.linear1(x)))
            x = x + self.dropout3(ff_out)
            x = self.norm3(x)
        
        if return_attention:
            return x, (self_attn_weights, cross_attn_weights)
        return x

class Decoder(nn.Module):
    """
    优化的Transformer解码器
    - 支持Pre-LayerNorm和Post-LayerNorm
    - 改进的嵌入初始化
    - 支持注意力权重返回
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ffn_hidden_dim, 
                 num_layers, dropout, padding_idx, norm_first=False, activation='relu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm_first = norm_first
        
        # 嵌入层
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout,
                        activation=activation, norm_first=norm_first)
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, vocab_size)
        
        # 移除final_norm，因为实际模型中没有
        self.final_norm = None
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        # 嵌入层初始化
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        # 输出投影初始化
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None, return_attention=False):
        """
        前向传播
        Args:
            tgt: (batch, tgt_len)
            memory: 编码器输出 (batch, src_len, embed_dim)
            tgt_mask: 目标序列掩码
            tgt_key_padding_mask: 目标填充掩码
            memory_key_padding_mask: 源序列填充掩码
            return_attention: 是否返回注意力权重
        Returns:
            output: (batch, tgt_len, vocab_size)
            attention_weights: 各层注意力权重 (可选)
        """
        # 嵌入 + 位置编码
        x = self.token_embed(tgt) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        
        self_attention_weights = []
        cross_attention_weights = []
        
        # 通过解码器层
        for layer in self.layers:
            if return_attention:
                x, (self_attn, cross_attn) = layer(
                    x, memory, 
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=True
                )
                self_attention_weights.append(self_attn)
                cross_attention_weights.append(cross_attn)
            else:
                x = layer(
                    x, memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_attention=False
                )
        
        # 最终LayerNorm (仅对Pre-LayerNorm)
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # 输出投影
        output = self.out_proj(x)  # (batch, tgt_len, vocab_size)
        
        if return_attention:
            return output, (self_attention_weights, cross_attention_weights)
        return output