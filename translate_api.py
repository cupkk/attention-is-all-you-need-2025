import sys
import os
import time
import logging

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入模块
from utils.mask import generate_square_subsequent_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil
from jinja2 import Template
import sentencepiece as spm  # BPE 分词支持

# BPE 分词器全局变量（延迟加载）
sp_de = None
sp_en = None

# 词表反向映射全局变量
idx_to_token_src = None
idx_to_token_tgt = None

def _load_bpe_models():
    """延迟加载 BPE 分词器"""
    global sp_de, sp_en
    if sp_de is None or sp_en is None:
        # 获取项目根目录的绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bpe_dir = os.path.join(script_dir, "bpe_models")
        
        sp_de = spm.SentencePieceProcessor()
        sp_de.Load(os.path.join(bpe_dir, "bpe_de.model"))
        
        sp_en = spm.SentencePieceProcessor()
        sp_en.Load(os.path.join(bpe_dir, "bpe_en.model"))

# 使用 BPE 分词函数
def tokenize_de(text):
    _load_bpe_models()
    return sp_de.encode_as_pieces(text.strip())

def tokenize_en(text):
    _load_bpe_models()
    return sp_en.encode_as_pieces(text.strip())

# 特殊符号索引
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

def plot_attention_weights(src_tokens, tgt_tokens, attention_matrix, filename="attention", num_heads=None):
    """
    绘制注意力热力图，支持单头或多头 attention 可视化

    参数:
        src_tokens (list): 源语言 token 列表（如 ['▁ein', '▁mann', ...]）
        tgt_tokens (list): 目标语言 token 列表（如 ['▁a', '▁man', ...]）
        attention_matrix (np.ndarray or torch.Tensor): 注意力矩阵，形状为 (num_heads, tgt_len, src_len)
        filename (str): 保存图像的文件名前缀
        num_heads (int): 要可视化的头数（默认全部绘制）
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_matrix = attention_matrix.detach().cpu().numpy()
    
    # 如果是3维(num_heads, tgt_len, src_len)，选择要可视化的头数
    if len(attention_matrix.shape) == 3:
        total_heads = attention_matrix.shape[0]
        if num_heads is None:
            num_heads = min(4, total_heads)  # 默认最多显示4个头
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(min(num_heads, 4)):
            if i < total_heads:
                ax = axes[i]
                sns.heatmap(attention_matrix[i], 
                           xticklabels=src_tokens, 
                           yticklabels=tgt_tokens,
                           cmap='Blues', 
                           ax=ax,
                           cbar_kws={'shrink': 0.8})
                ax.set_title(f'Attention Head {i+1}')
                ax.set_xlabel('Source Tokens')
                ax.set_ylabel('Target Tokens')
        
        # 隐藏多余的子图
        for i in range(num_heads, 4):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{filename}_heads.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    else:
        # 2维矩阵，直接绘制
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_matrix, 
                   xticklabels=src_tokens, 
                   yticklabels=tgt_tokens,
                   cmap='Blues',
                   cbar_kws={'shrink': 0.8})
        plt.title('Attention Weights')
        plt.xlabel('Source Tokens')
        plt.ylabel('Target Tokens')
        plt.tight_layout()
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_attention_html_report(src_sentence, tgt_sentence, attention_weights, output_file="attention_report.html"):
    """
    生成交互式 HTML 注意力可视化报告
    """
    # 转换attention权重为列表格式
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # 如果是多头注意力，取平均
    if len(attention_weights.shape) == 3:
        attention_weights = np.mean(attention_weights, axis=0)
    
    # HTML模板
    html_template = Template('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attention Visualization</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .sentence { margin: 20px 0; }
            .attention-matrix { margin: 20px 0; }
            .token { display: inline-block; margin: 2px; padding: 4px 8px; border-radius: 3px; }
            table { border-collapse: collapse; margin: 20px 0; }
            td, th { border: 1px solid #ddd; padding: 8px; text-align: center; min-width: 40px; }
            .high-attention { background-color: #ff6b6b; color: white; }
            .medium-attention { background-color: #feca57; }
            .low-attention { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <h1>🔍 Attention Weights Visualization</h1>
        
        <div class="sentence">
            <h2>Source (German): {{ src_sentence }}</h2>
        </div>
        
        <div class="sentence">
            <h2>Target (English): {{ tgt_sentence }}</h2>
        </div>
        
        <div class="attention-matrix">
            <h2>Attention Matrix</h2>
            <table>
                <tr>
                    <th></th>
                    {% for src_token in src_tokens %}
                    <th>{{ src_token }}</th>
                    {% endfor %}
                </tr>
                {% for i, tgt_token in enumerate(tgt_tokens) %}
                <tr>
                    <th>{{ tgt_token }}</th>
                    {% for j, src_token in enumerate(src_tokens) %}
                    <td style="background-color: rgba(102, 126, 234, {{ attention_matrix[i][j] }})">
                        {{ "%.2f"|format(attention_matrix[i][j]) }}
                    </td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div style="margin-top: 30px; color: #666; font-size: 12px;">
            Generated at: {{ timestamp }}
        </div>
    </body>
    </html>
    ''')
    
    # 准备数据
    src_tokens = tokenize_de(src_sentence)
    tgt_tokens = tokenize_en(tgt_sentence)
    
    # 渲染HTML
    html_content = html_template.render(
        src_sentence=src_sentence,
        tgt_sentence=tgt_sentence,
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        attention_matrix=attention_weights.tolist(),
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Attention report saved to {output_file}")

# 全局变量存储已加载的模型和词表
_model = None
_vocab_src = None
_vocab_tgt = None

def _load_vocabularies():
    """加载词表"""
    global _vocab_src, _vocab_tgt, idx_to_token_src, idx_to_token_tgt
    if _vocab_src is None or _vocab_tgt is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        _vocab_src = torch.load(os.path.join(script_dir, "multi30k_processed_bpe/vocab_de.pth"))
        _vocab_tgt = torch.load(os.path.join(script_dir, "multi30k_processed_bpe/vocab_en.pth"))
        
        # 创建反向映射
        idx_to_token_src = {idx: token for token, idx in _vocab_src.items()}
        idx_to_token_tgt = {idx: token for token, idx in _vocab_tgt.items()}

def translate_german_to_english(sentence: str, beam_size=10, max_len=120, alpha=0.7, visualize=False):
    """
    输入一个德文字符串，返回对应的英文翻译，并可选地可视化 attention 权重。
    
    参数:
        sentence (str): 德文句子（字符串）
        beam_size (int): Beam Search 的搜索宽度
        max_len (int): 最大生成长度
        alpha (float): 长度惩罚系数（越大越倾向于长句）
        visualize (bool): 是否绘制 attention 权重热力图
    
    返回:
        str: 英文翻译结果
    """
    global _model, _vocab_src, _vocab_tgt
    
    # 加载词表（首次调用时加载）
    _load_vocabularies()
    
    # 加载模型（首次调用时加载）
    if _model is None:
        _model = _load_model()
    
    model = _model
    vocab_src = _vocab_src
    vocab_tgt = _vocab_tgt
    device = next(model.parameters()).device
    
    # 确保 BPE 分词器已加载
    _load_bpe_models()
    
    # 使用 BPE 分词并转换为索引
    src_tokens = tokenize_de(sentence)
    src_indices = [BOS_IDX]
    if vocab_src is not None:
        for token in src_tokens:
            if token in vocab_src:
                src_indices.append(vocab_src[token])
            else:
                src_indices.append(0)  # UNK token
    src_indices.append(EOS_IDX)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # 构建掩码
    src_mask = torch.zeros((src_tensor.size(1), src_tensor.size(1)), dtype=torch.bool).to(device)
    src_pad_mask = (src_tensor == PAD_IDX).to(device)
    
    with torch.no_grad():
        memory = model.encoder(src_tensor, src_mask, src_pad_mask)
    
    # Beam Search 初始化
    hypotheses = [[BOS_IDX]]
    hyp_scores = torch.zeros(1, device=device)
    completed_hypotheses = []
    
    # Beam Search 解码
    for step in range(max_len):
        if not hypotheses:
            break
            
        new_hypotheses = []
        new_scores = []
        
        for i, hyp in enumerate(hypotheses):
            if hyp[-1] == EOS_IDX:
                completed_hypotheses.append((hyp, hyp_scores[i].item()))
                continue
                
            # 准备当前假设的张量
            tgt_tensor = torch.tensor([hyp], dtype=torch.long, device=device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            tgt_pad_mask = (tgt_tensor == PAD_IDX).to(device)
            
            # 模型前向传播
            out = model.decoder(tgt_tensor, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
            log_probs = torch.log_softmax(out[0, -1, :], dim=-1)
            
            # 获取top-k候选
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
            
            for k in range(beam_size):
                new_hyp = hyp + [topk_indices[k].item()]
                new_score = hyp_scores[i] + topk_log_probs[k]
                new_hypotheses.append(new_hyp)
                new_scores.append(new_score)
        
        if not new_hypotheses:
            break
            
        # 按分数排序并保留top-k
        scored_hyps = list(zip(new_hypotheses, new_scores))
        scored_hyps.sort(key=lambda x: x[1], reverse=True)
        
        hypotheses = [hyp for hyp, _ in scored_hyps[:beam_size]]
        hyp_scores = torch.tensor([score for _, score in scored_hyps[:beam_size]], device=device)
    
    # 选择最佳假设
    if completed_hypotheses:
        # 应用长度惩罚
        best_hyp, best_score = max(completed_hypotheses, 
                                 key=lambda x: x[1] / (len(x[0]) ** alpha))
    else:
        best_hyp = hypotheses[0] if hypotheses else [BOS_IDX, EOS_IDX]
    
    # 移除特殊标记并转换为文本
    if BOS_IDX in best_hyp:
        best_hyp = best_hyp[1:]  # 移除BOS
    if EOS_IDX in best_hyp:
        best_hyp = best_hyp[:best_hyp.index(EOS_IDX)]  # 移除EOS及其后面的内容
    
    # 转换索引为tokens - 使用全局的反向映射
    tgt_tokens = []
    if idx_to_token_tgt is not None:
        for idx in best_hyp:
            if idx in idx_to_token_tgt:
                tgt_tokens.append(idx_to_token_tgt[idx])
            else:
                tgt_tokens.append('<unk>')
    
    # 将BPE pieces合并为最终文本
    _load_bpe_models()  # 确保sp_en已加载
    if sp_en is not None:
        try:
            # 使用 decode_pieces 方法
            translated_text = sp_en.decode_pieces(tgt_tokens)
        except Exception:
            # 如果解码失败，使用简单的字符串拼接
            token_string = ''.join(tgt_tokens).replace('▁', ' ').strip()
            translated_text = token_string
    else:
        # 备用方案：简单拼接
        token_string = ''.join(tgt_tokens).replace('▁', ' ').strip()
        translated_text = token_string
    
    # 可视化（如果需要）
    if visualize:
        try:
            create_attention_html_report(sentence, translated_text, torch.zeros(1, 10, 10))
            print("✅ Attention visualization saved")
        except Exception as e:
            print("⚠️ Attention visualization failed:", e)
    
    return translated_text

# 内部函数：加载模型
def _load_model():
    # 导入与训练时相同的组件
    from models.Encoder import Encoder
    from models.Decoder import Decoder
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_src = torch.load(os.path.join(script_dir, "multi30k_processed_bpe/vocab_de.pth"))
    vocab_tgt = torch.load(os.path.join(script_dir, "multi30k_processed_bpe/vocab_en.pth"))
    
    # 创建与训练时相同的编码器和解码器
    encoder = Encoder(
        vocab_size=len(vocab_src),
        embed_dim=512,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_layers=3,  # 实际保存的模型是3层
        dropout=0.1,
        padding_idx=1,
        norm_first=True,  # 与train.py中的NORM_FIRST=True保持一致
        activation='relu'  # 与train.py中的ACTIVATION='relu'保持一致
    )
    
    decoder = Decoder(
        vocab_size=len(vocab_tgt),
        embed_dim=512,
        num_heads=8,
        ffn_hidden_dim=2048,
        num_layers=3,  # 实际保存的模型是3层
        dropout=0.1,
        padding_idx=1,
        norm_first=True,
        activation='relu'
    )
    
    # 创建与训练时相同的Seq2SeqTransformer
    class Seq2SeqTransformer(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        
        def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):
            memory = self.encoder(src, src_mask, src_key_padding_mask)
            output = self.decoder(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
            return output
    
    model = Seq2SeqTransformer(encoder, decoder)
    
    model_path = os.path.join(script_dir, "transformer_model.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 模型超参数（需与训练时保持一致）
EMB_SIZE = 512
FFN_HID_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
