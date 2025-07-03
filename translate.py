import os
import torch
import torch.nn as nn
from torchtext.data.metrics import bleu_score

# 根据当前模块化结构调整导入路径
from models.Transformer import Transformer
from utils.mask import generate_square_subsequent_mask


# 配置参数
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

# 模型超参数（需与训练时保持一致）
EMB_SIZE = 512
FFN_HID_DIM = 2048
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1

# 加载词表
vocab_src = torch.load("multi30k_processed/vocab_de.pth")
vocab_tgt = torch.load("multi30k_processed/vocab_en.pth")

SRC_VOCAB_SIZE = len(vocab_src)
TGT_VOCAB_SIZE = len(vocab_tgt)

# 构建模型
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    embed_dim=EMB_SIZE,
    num_heads=NUM_HEADS,
    ffn_hidden_dim=FFN_HID_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    padding_idx=PAD_IDX
)

# 加载模型权重
try:
    model.load_state_dict(torch.load("transformer_model.pth", map_location='cpu'))
except FileNotFoundError:
    raise RuntimeError("模型权重文件 'transformer_model.pth' 未找到，请确认路径是否正确")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 从本地文件加载测试集函数
def read_parallel_sentences(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        src_lines = [line.strip().split() for line in f_src]
        tgt_lines = [line.strip().split() for line in f_tgt]
    return list(zip(src_lines, tgt_lines))

processed_dir = "multi30k_processed"
test_data = read_parallel_sentences(
    os.path.join(processed_dir, "test.tok.de"),
    os.path.join(processed_dir, "test.tok.en")
)

# Beam Search 翻译函数
def translate_sentence_beam_search(sentence_tokens, beam_size=10, max_len=120, alpha=0.7):
    model.eval()
    src_tokens = [BOS_IDX] + [vocab_src[token] for token in sentence_tokens] + [EOS_IDX]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(device)

    src_mask = torch.zeros((src_tensor.size(1), src_tensor.size(1)), dtype=torch.bool).to(device)
    src_pad_mask = (src_tensor == PAD_IDX).to(device)

    with torch.no_grad():
        memory = model.encoder(src_tensor, src_mask, src_pad_mask)

    hypotheses = [[BOS_IDX]]
    hyp_scores = torch.zeros(1, device=device)
    completed_hypotheses = []

    for _ in range(max_len):
        new_hypotheses = []
        new_hyp_scores = []

        for i in range(len(hypotheses)):
            tgt_tensor = torch.tensor([hypotheses[i]], dtype=torch.long, device=device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            tgt_pad_mask = (tgt_tensor == PAD_IDX).to(device)

            with torch.no_grad():
                output = model.decoder(tgt_tensor, memory, tgt_mask, tgt_pad_mask, src_pad_mask)

            logits = output[0]
            log_probs = torch.nn.functional.log_softmax(logits[-1], dim=-1)

            topk_log_probs, topk_indices = log_probs.topk(beam_size)

            for log_p, idx in zip(topk_log_probs, topk_indices):
                new_seq = hypotheses[i] + [idx.item()]
                new_score = hyp_scores[i] + log_p.item()

                if idx.item() == EOS_IDX:
                    completed_hypotheses.append((new_seq, new_score / ((len(new_seq) ** alpha) / (5 ** alpha))))
                else:
                    new_hypotheses.append(new_seq)
                    new_hyp_scores.append(new_score)

        if len(completed_hypotheses) >= beam_size:
            break

        if not new_hypotheses:
            break

# 确保 k 不超过候选数量
        k = min(beam_size, len(new_hyp_scores))
        topk_indices = torch.tensor(new_hyp_scores).topk(k).indices.tolist()

        hypotheses = [new_hypotheses[i] for i in topk_indices]
        hyp_scores = torch.tensor([new_hyp_scores[i] for i in topk_indices], device=device)

    # 添加剩余假设
    for seq, score in zip(hypotheses, hyp_scores):
        completed_hypotheses.append((seq, score / (len(seq) ** alpha)))

    # 按得分排序
    completed_hypotheses.sort(key=lambda x: x[1], reverse=True)
    best_seq = completed_hypotheses[0][0]

    pred_tokens = []
    for idx in best_seq[1:]:
        if idx == EOS_IDX:
            break
        if hasattr(vocab_tgt, "get_itos"):
            token = vocab_tgt.get_itos()[idx]
        else:
            token = vocab_tgt.lookup_token(idx)
        pred_tokens.append(token)

    return " ".join(pred_tokens)


# 遍历测试集进行翻译，并计算 BLEU 分数
pred_corpus = []
ref_corpus = []

for src_tokens, tgt_tokens in test_data:
    pred_text = translate_sentence_beam_search(src_tokens, beam_size=10)
    pred_tokens = pred_text.split()
    pred_corpus.append(pred_tokens)
    ref_corpus.append([tgt_tokens])  # 每个参考翻译用子列表包裹

bleu = bleu_score(pred_corpus, ref_corpus)
print(f"BLEU score = {bleu:.4f}")

