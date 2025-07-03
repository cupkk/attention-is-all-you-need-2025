import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    优化的多头自注意力机制
    - 支持论文标准的scaled dot-product attention
    - 添加权重初始化策略
    - 支持注意力权重返回用于可视化
    - 优化计算效率
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)  # 预计算缩放因子
        
        # 使用Xavier初始化，符合论文建议
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot初始化，符合原论文"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, return_attention=False):
        """
        优化的前向传播
        Args:
            query, key, value: (batch, seq_len, embed_dim)
            attn_mask: 注意力掩码
            key_padding_mask: 填充掩码
            return_attention: 是否返回注意力权重
        Returns:
            output: (batch, seq_len, embed_dim)
            attn_weights: 注意力权重 (可选)
        """
        B, Q_len, _ = query.shape
        K_len = key.shape[1]
        
        # 线性投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # 重塑为多头格式: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        # 应用填充掩码
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Q_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output, None