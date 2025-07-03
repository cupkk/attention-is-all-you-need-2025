import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from models.Encoder import Encoder
from models.Decoder import Decoder
from utils.mask import create_mask
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.clip_grad import clip_grad_norm
# 修复AMP导入 - 使用正确的API
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


# 固定随机种子
import random
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. 加载词表
vocab_src = torch.load("multi30k_processed_bpe/vocab_de.pth")
vocab_tgt = torch.load("multi30k_processed_bpe/vocab_en.pth")

PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_tgt)

# 原论文标准配置
EMB_SIZE = 512
FFN_HID_DIM = 2048
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1

# 新增优化参数
NORM_FIRST = True  # True为Pre-LayerNorm，False为Post-LayerNorm（原论文）
ACTIVATION = 'relu'  # 'relu' 或 'gelu'
LABEL_SMOOTHING = 0.1  # Label smoothing参数
WARMUP_STEPS = 4000  # 学习率预热步数（原论文标准）

# 训练参数（考虑显存限制的梯度累积方案）
BATCH_SIZE = 8   # 减小实际batch size避免OOM
ACCUMULATE_GRAD_BATCHES = 4  # 梯度累积4步，有效batch_size = 8*4 = 32（等同原论文）
MAX_SEQ_LEN = 100  # 限制序列长度

# 2. 数据集定义
class ParallelTextDataset(Dataset):
    def __init__(self, src_path, tgt_path, max_samples=None):
        with open(src_path, encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f]
        with open(tgt_path, encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f]
        assert len(self.src_lines) == len(self.tgt_lines)
        if max_samples is not None:
            self.src_lines = self.src_lines[:max_samples]
            self.tgt_lines = self.tgt_lines[:max_samples]
    def __len__(self):
        return len(self.src_lines)
    def __getitem__(self, idx):
        return self.src_lines[idx], self.tgt_lines[idx]

def tokenize_bpe(text):
    return text.strip().split()

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokens = [BOS_IDX] + [vocab_src.get(tok, vocab_src.get('<unk>', 0)) for tok in tokenize_bpe(src_sample)] + [EOS_IDX]
        tgt_tokens = [BOS_IDX] + [vocab_tgt.get(tok, vocab_tgt.get('<unk>', 0)) for tok in tokenize_bpe(tgt_sample)] + [EOS_IDX]
        
        # 限制序列长度防止OOM
        if len(src_tokens) > MAX_SEQ_LEN:
            src_tokens = src_tokens[:MAX_SEQ_LEN-1] + [EOS_IDX]
        if len(tgt_tokens) > MAX_SEQ_LEN:
            tgt_tokens = tgt_tokens[:MAX_SEQ_LEN-1] + [EOS_IDX]
            
        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_tokens, dtype=torch.long))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

train_dataset = ParallelTextDataset(
    "multi30k_processed_bpe/train_part1.tok.de",
    "multi30k_processed_bpe/train_part1.tok.en",
    max_samples=30000
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

val_dataset = ParallelTextDataset(
    "multi30k_processed_bpe/val.tok.de",
    "multi30k_processed_bpe/val.tok.en"
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

encoder = Encoder(SRC_VOCAB_SIZE, EMB_SIZE, NUM_HEADS, FFN_HID_DIM, NUM_ENCODER_LAYERS, 
                  DROPOUT, padding_idx=PAD_IDX, norm_first=NORM_FIRST, activation=ACTIVATION)
decoder = Decoder(TGT_VOCAB_SIZE, EMB_SIZE, NUM_HEADS, FFN_HID_DIM, NUM_DECODER_LAYERS, 
                  DROPOUT, padding_idx=PAD_IDX, norm_first=NORM_FIRST, activation=ACTIVATION)
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
# 添加Label Smoothing损失函数
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing损失函数 - 原论文中提到的正则化技术
    """
    def __init__(self, num_classes, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        pred: (N, C) logits
        target: (N,) targets
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_(target.unsqueeze(1) == self.ignore_index, 0)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# 添加Warmup学习率调度器
class WarmupLRScheduler:
    """
    原论文中的学习率调度策略: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 使用Label Smoothing损失函数
criterion = LabelSmoothingCrossEntropy(TGT_VOCAB_SIZE, smoothing=LABEL_SMOOTHING, ignore_index=PAD_IDX)

# 使用原论文推荐的Adam参数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

# 使用Warmup学习率调度
warmup_scheduler = WarmupLRScheduler(optimizer, EMB_SIZE, WARMUP_STEPS)

def accuracy_fn(pred, target, pad_idx=PAD_IDX):
    pred = pred.argmax(-1)
    mask = target != pad_idx
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct, total

def greedy_decode(model, src, src_mask, src_pad_mask, max_len=50, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    # src: [1, src_len]
    memory = model.encoder(src, src_mask, src_pad_mask)
    ys = torch.ones(1, 1).fill_(bos_idx).type_as(src)
    for i in range(max_len-1):
        tgt_mask, _, _, tgt_pad_mask = create_mask(ys, ys, PAD_IDX)
        out = model.decoder(ys, memory, tgt_mask.to(src.device), tgt_pad_mask.to(src.device), src_pad_mask.to(src.device))
        prob = out[:, -1, :]
        next_word = prob.argmax(-1).unsqueeze(0)
        ys = torch.cat([ys, next_word], dim=1)
        if next_word.item() == eos_idx:
            break
    return ys.squeeze(0).tolist()[1:]  # 去掉BOS

def beam_search_decode(model, src, src_mask, src_pad_mask, beam_size=4, max_len=50, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    device = src.device
    memory = model.encoder(src, src_mask, src_pad_mask)
    ys = torch.ones(1, 1).fill_(bos_idx).long().to(device)
    sequences = [(ys, 0.0)]
    for _ in range(max_len - 1):
        all_candidates = []
        for seq, score in sequences:
            if seq[0, -1].item() == eos_idx:
                all_candidates.append((seq, score))
                continue
            tgt_mask, _, _, tgt_pad_mask = create_mask(seq, seq, PAD_IDX)
            out = model.decoder(seq, memory, tgt_mask.to(device), tgt_pad_mask.to(device), src_pad_mask.to(device))
            log_probs = torch.log_softmax(out[:, -1, :], dim=-1)
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
            for k in range(beam_size):
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, k].item()
                all_candidates.append((new_seq, new_score))
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_size]
        if all(seq[0][0, -1].item() == eos_idx for seq in sequences):
            break
    best_seq = sequences[0][0].squeeze(0).tolist()
    if bos_idx in best_seq:
        best_seq = best_seq[1:]
    if eos_idx in best_seq:
        best_seq = best_seq[:best_seq.index(eos_idx)]
    return best_seq

print("Using device:", device)
print("Model on:", next(model.parameters()).device)
def evaluate(model, data_loader, vocab_tgt, device, use_beam=True, beam_size=4):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    refs = []
    hyps = []
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, PAD_IDX)
            logits = model(src, tgt_input, src_mask.to(device), tgt_mask.to(device), src_pad_mask.to(device), tgt_pad_mask.to(device))
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            correct, tokens = accuracy_fn(logits, tgt_output)
            total_correct += correct
            total_tokens += tokens
            # 解码BLEU
            for i in range(src.size(0)):
                src_sent = src[i].unsqueeze(0)
                src_mask_i = (src_sent != PAD_IDX).unsqueeze(1)
                src_pad_mask_i = (src_sent == PAD_IDX)
                if use_beam:
                    hyp = beam_search_decode(model, src_sent, src_mask_i, src_pad_mask_i, beam_size=beam_size)
                else:
                    hyp = greedy_decode(model, src_sent, src_mask_i, src_pad_mask_i)
                ref = tgt[i, 1:].tolist()
                if EOS_IDX in ref:
                    ref = ref[:ref.index(EOS_IDX)]
                if EOS_IDX in hyp:
                    hyp = hyp[:hyp.index(EOS_IDX)]
                refs.append([ref])
                hyps.append(hyp)
    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_tokens if total_tokens > 0 else 0
    bleu_score = corpus_bleu(refs, hyps)
    # 确保BLEU分数是float类型
    if isinstance(bleu_score, (list, tuple)):
        bleu_score = float(bleu_score[0]) if len(bleu_score) > 0 else 0.0
    else:
        bleu_score = float(bleu_score)
    return avg_loss, acc, bleu_score

def train_one_epoch(model, data_loader, optimizer, warmup_scheduler, device, clip=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    current_lr = 0.0  # 初始化变量
    
    # 梯度累积相关变量
    accumulate_loss = 0.0
    
    for batch_idx, (src, tgt) in enumerate(data_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, PAD_IDX)
        
        with autocast():
            logits = model(src, tgt_input, src_mask.to(device), tgt_mask.to(device), src_pad_mask.to(device), tgt_pad_mask.to(device))
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            # 梯度累积：损失要除以累积步数
            loss = loss / ACCUMULATE_GRAD_BATCHES
        
        scaler.scale(loss).backward()
        accumulate_loss += loss.item()
        
        # 每ACCUMULATE_GRAD_BATCHES步更新一次参数
        if (batch_idx + 1) % ACCUMULATE_GRAD_BATCHES == 0 or (batch_idx + 1) == len(data_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # 更新学习率
            current_lr = warmup_scheduler.step()
        
        # 统计时要乘回累积步数，保持loss尺度一致
        total_loss += accumulate_loss * ACCUMULATE_GRAD_BATCHES
        correct, tokens = accuracy_fn(logits, tgt_output)
        total_correct += correct
        total_tokens += tokens
        
        # 重置累积损失
        if (batch_idx + 1) % ACCUMULATE_GRAD_BATCHES == 0:
            accumulate_loss = 0.0
        
    avg_loss = total_loss / len(data_loader)
    acc = total_correct / total_tokens if total_tokens > 0 else 0
    return avg_loss, acc, current_lr

# TensorBoard日志
writer = SummaryWriter(log_dir="runs/transformer_exp")

# ...existing code...

num_epochs = 15  # 减少训练轮数，避免过长时间

# 显示当前配置信息
print("="*60)
print("🚀 Transformer训练配置 (优化后防止OOM)")
print("="*60)
print(f"📊 数据配置:")
print(f"   - 训练样本数: {len(train_dataset):,}")
print(f"   - 验证样本数: {len(val_dataset):,}")
print(f"   - 德语词汇量: {SRC_VOCAB_SIZE:,}")
print(f"   - 英语词汇量: {TGT_VOCAB_SIZE:,}")

print(f"\n🏗️  模型配置 (减小规模防止OOM):")
print(f"   - 嵌入维度: {EMB_SIZE}")
print(f"   - FFN隐藏层: {FFN_HID_DIM}")
print(f"   - 编码器层数: {NUM_ENCODER_LAYERS}")
print(f"   - 解码器层数: {NUM_DECODER_LAYERS}")
print(f"   - 注意力头数: {NUM_HEADS}")
print(f"   - Dropout: {DROPOUT}")

print(f"\n⚙️  训练配置:")
print(f"   - Batch Size: {BATCH_SIZE} (减小防止OOM)")
print(f"   - 梯度累积步数: {ACCUMULATE_GRAD_BATCHES}")
print(f"   - 有效Batch Size: {BATCH_SIZE * ACCUMULATE_GRAD_BATCHES}")
print(f"   - 学习率: {optimizer.param_groups[0]['lr']}")
print(f"   - Warmup步数: {WARMUP_STEPS}")
print(f"   - 训练轮数: {num_epochs}")
print(f"   - Label Smoothing: {LABEL_SMOOTHING}")

print(f"\n💻 硬件配置:")
print(f"   - 设备: {device}")
if device.type == 'cuda':
    print(f"   - GPU名称: {torch.cuda.get_device_name()}")
    print(f"   - 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   - 混合精度: 启用 (AMP)")

print("="*60)
print("开始训练...")
print("="*60)

best_bleu = 0.0
beam_eval_interval = 5  # 每N轮用beam search评估一次

for epoch in range(num_epochs):
    train_loss, train_acc, current_lr = train_one_epoch(model, train_loader, optimizer, warmup_scheduler, device)
    # 训练时只用greedy解码评估
    val_loss, val_acc, val_bleu = evaluate(model, val_loader, vocab_tgt, device, use_beam=False)
    
    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {current_lr:.6f}")
    print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, BLEU (greedy): {val_bleu:.4f}")
    writer.add_scalar("Loss/train", train_loss, epoch+1)
    writer.add_scalar("Loss/val", val_loss, epoch+1)
    writer.add_scalar("Accuracy/train", train_acc, epoch+1)
    writer.add_scalar("Accuracy/val", val_acc, epoch+1)
    writer.add_scalar("BLEU/val_greedy", val_bleu, epoch+1)
    writer.add_scalar("Learning_Rate", current_lr, epoch+1)
    
    # 每N轮用beam search评估BLEU
    if (epoch + 1) % beam_eval_interval == 0 or (epoch + 1) == num_epochs:
        val_loss_beam, val_acc_beam, val_bleu_beam = evaluate(model, val_loader, vocab_tgt, device, use_beam=True, beam_size=4)
        print(f"  Val BLEU (beam search): {val_bleu_beam:.4f}")
        writer.add_scalar("BLEU/val_beam", val_bleu_beam, epoch+1)
        # 保存最佳beam BLEU模型
        if val_bleu_beam > best_bleu:
            best_bleu = val_bleu_beam
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print(f"  Best model saved at epoch {epoch+1} (BLEU={val_bleu_beam:.4f})")
# 最终模型
torch.save(model.state_dict(), "transformer_model.pth")
writer.close()
print("训练完成，模型参数已保存至 transformer_model.pth")