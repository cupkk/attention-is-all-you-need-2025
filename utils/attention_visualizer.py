import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional

class AttentionVisualizer:
    """
    注意力权重可视化工具
    - 支持自注意力和交叉注意力可视化
    - 支持多头注意力平均或单头显示
    - 支持保存图片和交互式显示
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        
    def visualize_self_attention(self, attention_weights: torch.Tensor, 
                                tokens: List[str], 
                                layer_idx: int = -1,
                                head_idx: Optional[int] = None,
                                save_path: Optional[str] = None,
                                title: Optional[str] = None):
        """
        可视化自注意力权重
        
        Args:
            attention_weights: 注意力权重 (num_layers, batch, num_heads, seq_len, seq_len)
            tokens: 词汇列表
            layer_idx: 要可视化的层索引，-1表示最后一层
            head_idx: 要可视化的头索引，None表示平均所有头
            save_path: 保存路径
            title: 图标题
        """
        # 选择层
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
        attn = attention_weights[layer_idx]  # (batch, num_heads, seq_len, seq_len)
        
        # 取第一个样本
        attn = attn[0]  # (num_heads, seq_len, seq_len)
        
        # 选择头或平均
        if head_idx is not None:
            attn = attn[head_idx]  # (seq_len, seq_len)
            head_info = f"Head {head_idx}"
        else:
            attn = attn.mean(dim=0)  # (seq_len, seq_len)
            head_info = "Average of all heads"
        
        # 转换为numpy
        attn = attn.detach().cpu().numpy()
        
        # 创建图形
        plt.figure(figsize=self.figsize)
        sns.heatmap(attn, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues', 
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Attention Weight'})
        
        if title is None:
            title = f"Self-Attention Layer {layer_idx} - {head_info}"
        plt.title(title)
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_cross_attention(self, attention_weights: torch.Tensor,
                                 src_tokens: List[str],
                                 tgt_tokens: List[str],
                                 layer_idx: int = -1,
                                 head_idx: Optional[int] = None,
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None):
        """
        可视化交叉注意力权重
        
        Args:
            attention_weights: 注意力权重 (num_layers, batch, num_heads, tgt_len, src_len)
            src_tokens: 源序列词汇
            tgt_tokens: 目标序列词汇
            layer_idx: 要可视化的层索引
            head_idx: 要可视化的头索引
            save_path: 保存路径
            title: 图标题
        """
        # 选择层
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
        attn = attention_weights[layer_idx]  # (batch, num_heads, tgt_len, src_len)
        
        # 取第一个样本
        attn = attn[0]  # (num_heads, tgt_len, src_len)
        
        # 选择头或平均
        if head_idx is not None:
            attn = attn[head_idx]  # (tgt_len, src_len)
            head_info = f"Head {head_idx}"
        else:
            attn = attn.mean(dim=0)  # (tgt_len, src_len)
            head_info = "Average of all heads"
        
        # 转换为numpy
        attn = attn.detach().cpu().numpy()
        
        # 创建图形
        plt.figure(figsize=self.figsize)
        sns.heatmap(attn,
                   xticklabels=src_tokens,
                   yticklabels=tgt_tokens,
                   cmap='Reds',
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Attention Weight'})
        
        if title is None:
            title = f"Cross-Attention Layer {layer_idx} - {head_info}"
        plt.title(title)
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_attention_heads(self, attention_weights: torch.Tensor,
                                 tokens: List[str],
                                 layer_idx: int = -1,
                                 save_path: Optional[str] = None):
        """
        可视化一层中所有注意力头
        
        Args:
            attention_weights: 注意力权重
            tokens: 词汇列表
            layer_idx: 层索引
            save_path: 保存路径
        """
        if layer_idx == -1:
            layer_idx = len(attention_weights) - 1
        attn = attention_weights[layer_idx][0]  # (num_heads, seq_len, seq_len)
        
        num_heads = attn.shape[0]
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for head in range(num_heads):
            row, col = head // cols, head % cols
            ax = axes[row, col]
            
            head_attn = attn[head].detach().cpu().numpy()
            sns.heatmap(head_attn,
                       xticklabels=tokens,
                       yticklabels=tokens,
                       cmap='Blues',
                       ax=ax,
                       cbar=False,
                       annot=False)
            ax.set_title(f'Head {head}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        
        # 隐藏多余的子图
        for head in range(num_heads, rows * cols):
            row, col = head // cols, head % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'All Attention Heads - Layer {layer_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_attention_patterns(self, attention_weights: torch.Tensor,
                                  tokens: List[str]) -> dict:
        """
        分析注意力模式
        
        Returns:
            dict: 包含各种注意力统计信息
        """
        # 平均所有层和头
        avg_attn = torch.stack(attention_weights).mean(dim=(0, 2))  # (batch, seq_len, seq_len)
        avg_attn = avg_attn[0].detach().cpu().numpy()  # (seq_len, seq_len)
        
        analysis = {
            'max_attention_per_token': np.max(avg_attn, axis=1),
            'entropy_per_token': [-np.sum(row * np.log(row + 1e-8)) for row in avg_attn],
            'self_attention_strength': np.diag(avg_attn),
            'most_attended_tokens': np.argmax(avg_attn, axis=1),
            'attention_dispersion': np.std(avg_attn, axis=1)
        }
        
        return analysis

def tokens_to_words(token_ids: List[int], vocab: dict) -> List[str]:
    """
    将token ID转换为词汇
    """
    id_to_token = {v: k for k, v in vocab.items()}
    return [id_to_token.get(tid, f'<unk_{tid}>') for tid in token_ids]
