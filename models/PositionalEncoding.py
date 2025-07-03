import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    优化的正弦位置编码
    - 支持可学习位置编码选项
    - 改进的缓存机制
    - 支持不同的位置编码策略
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000, learnable=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.learnable = learnable
        
        if learnable:
            # 可学习位置编码
            self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        else:
            # 标准正弦位置编码
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            # 改进的计算方式，避免数值不稳定
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                               (-math.log(10000.0) / embed_dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            if embed_dim % 2 == 1:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            
            pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        优化的前向传播
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            x + positional encoding with dropout applied
        """
        seq_len = x.size(1)
        
        if self.learnable:
            # 可学习位置编码
            pos_encoding = self.pe[:, :seq_len, :]
        else:
            # 标准正弦位置编码
            pos_encoding = self.pe[:, :seq_len, :].to(x.device)
        
        x = x + pos_encoding
        return self.dropout(x)
    
    def get_encoding(self, seq_len, device=None):
        """
        获取位置编码（用于可视化或其他用途）
        """
        if self.learnable:
            return self.pe[:, :seq_len, :].detach()
        else:
            pe = self.pe[:, :seq_len, :]
            if device is not None:
                pe = pe.to(device)
            return pe.detach()