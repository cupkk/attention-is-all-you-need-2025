import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

class ModelAnalyzer:
    """
    模型分析工具
    - 参数统计
    - 性能分析
    - BLEU分数详细分析
    - 模型复杂度分析
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def count_parameters(self) -> Dict[str, int]:
        """
        统计模型参数数量
        """
        param_counts = {}
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            param_counts[name] = param_count
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        param_counts['total_parameters'] = total_params
        param_counts['trainable_parameters'] = trainable_params
        param_counts['non_trainable_parameters'] = total_params - trainable_params
        
        return param_counts
    
    def analyze_parameter_distribution(self):
        """
        分析参数分布
        """
        component_params = defaultdict(int)
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                component_params['encoder'] += param.numel()
            elif 'decoder' in name:
                component_params['decoder'] += param.numel()
            elif 'embed' in name:
                component_params['embeddings'] += param.numel()
            else:
                component_params['other'] += param.numel()
        
        # 可视化
        labels = list(component_params.keys())
        sizes = list(component_params.values())
        
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Parameter Distribution by Component')
        
        plt.subplot(1, 2, 2)
        plt.bar(labels, sizes)
        plt.title('Parameter Count by Component')
        plt.ylabel('Number of Parameters')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return component_params
    
    def estimate_memory_usage(self, batch_size: int = 32, seq_len: int = 100) -> Dict[str, float]:
        """
        估算内存使用量 (MB)
        """
        # 模型参数内存
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        
        # 激活内存 (粗略估计)
        # 假设 FP32，每个元素4字节
        embed_dim = 512  # 从模型配置获取
        
        # 编码器激活
        encoder_activation = batch_size * seq_len * embed_dim * 6 * 4 / 1024**2  # 6层
        
        # 解码器激活
        decoder_activation = batch_size * seq_len * embed_dim * 6 * 4 / 1024**2  # 6层
        
        # 注意力权重
        attention_memory = batch_size * 8 * seq_len * seq_len * 6 * 2 * 4 / 1024**2  # 8头，6层编码器+解码器
        
        total_activation = encoder_activation + decoder_activation + attention_memory
        
        return {
            'model_parameters_mb': model_memory,
            'encoder_activation_mb': encoder_activation,
            'decoder_activation_mb': decoder_activation,
            'attention_weights_mb': attention_memory,
            'total_activation_mb': total_activation,
            'estimated_total_mb': model_memory + total_activation
        }
    
    def analyze_gradient_flow(self, data_loader, criterion):
        """
        分析梯度流
        """
        self.model.train()
        gradient_norms = defaultdict(list)
        
        # 运行一个batch来分析梯度
        src, tgt = next(iter(data_loader))
        src, tgt = src.to(self.device), tgt.to(self.device)
        
        # 前向传播
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        from utils.mask import create_mask
        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt_input, 1)  # PAD_IDX=1
        
        logits = self.model(src, tgt_input, 
                           src_mask.to(self.device), tgt_mask.to(self.device),
                           src_pad_mask.to(self.device), tgt_pad_mask.to(self.device))
        
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # 反向传播
        loss.backward()
        
        # 收集梯度统计
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[name].append(grad_norm)
        
        self.model.zero_grad()
        
        # 可视化梯度分布
        names = list(gradient_norms.keys())
        norms = [gradient_norms[name][0] for name in names]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(names)), norms)
        plt.xlabel('Layer')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Flow Analysis')
        plt.xticks(range(len(names)), [name.split('.')[-1] for name in names], rotation=45)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
        return gradient_norms
    
    def benchmark_inference_speed(self, batch_sizes: List[int] = [1, 8, 16, 32], 
                                 seq_len: int = 50, num_runs: int = 100) -> Dict[int, float]:
        """
        测试推理速度
        """
        import time
        
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for batch_size in batch_sizes:
                # 创建随机输入
                src = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                tgt = torch.randint(0, 1000, (batch_size, seq_len)).to(self.device)
                
                from utils.mask import create_mask
                src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(src, tgt, 1)
                
                # 预热
                for _ in range(10):
                    _ = self.model(src, tgt, 
                                  src_mask.to(self.device), tgt_mask.to(self.device),
                                  src_pad_mask.to(self.device), tgt_pad_mask.to(self.device))
                
                # 计时
                start_time = time.time()
                for _ in range(num_runs):
                    _ = self.model(src, tgt,
                                  src_mask.to(self.device), tgt_mask.to(self.device),
                                  src_pad_mask.to(self.device), tgt_pad_mask.to(self.device))
                end_time = time.time()
                
                avg_time = (end_time - start_time) / num_runs
                results[batch_size] = avg_time
                
                print(f"Batch size {batch_size}: {avg_time:.4f}s per batch, "
                      f"{avg_time/batch_size:.4f}s per sample")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(list(results.keys()), list(results.values()), 'bo-')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Batch (s)')
        plt.title('Inference Time vs Batch Size')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        samples_per_sec = [bs / time for bs, time in results.items()]
        plt.plot(list(results.keys()), samples_per_sec, 'ro-')
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.title('Throughput vs Batch Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def detailed_bleu_analysis(self, predictions: List[str], references: List[str]) -> Dict:
        """
        详细的BLEU分析
        """
        from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
        from nltk.translate.bleu_score import SmoothingFunction
        
        smooth = SmoothingFunction()
        
        # 分词
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        # 计算各种BLEU分数
        bleu_scores = {}
        
        # 整体BLEU分数
        bleu_scores['corpus_bleu_1'] = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0))
        bleu_scores['corpus_bleu_2'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_scores['corpus_bleu_3'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0))
        bleu_scores['corpus_bleu_4'] = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        
        # 句子级BLEU分数分布
        sentence_bleus = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            try:
                score = sentence_bleu(ref, pred, smoothing_function=smooth.method1)
                sentence_bleus.append(score)
            except:
                sentence_bleus.append(0.0)
        
        bleu_scores['sentence_bleu_mean'] = np.mean(sentence_bleus)
        bleu_scores['sentence_bleu_std'] = np.std(sentence_bleus)
        bleu_scores['sentence_bleu_min'] = np.min(sentence_bleus)
        bleu_scores['sentence_bleu_max'] = np.max(sentence_bleus)
        
        # 长度分析
        pred_lengths = [len(pred) for pred in pred_tokens]
        ref_lengths = [len(ref[0]) for ref in ref_tokens]
        
        bleu_scores['avg_pred_length'] = np.mean(pred_lengths)
        bleu_scores['avg_ref_length'] = np.mean(ref_lengths)
        bleu_scores['length_ratio'] = np.mean(pred_lengths) / np.mean(ref_lengths)
        
        # 可视化BLEU分数分布
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(sentence_bleus, bins=50, alpha=0.7)
        plt.xlabel('Sentence BLEU Score')
        plt.ylabel('Count')
        plt.title('BLEU Score Distribution')
        plt.axvline(np.mean(sentence_bleus), color='red', linestyle='--', label=f'Mean: {np.mean(sentence_bleus):.3f}')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.scatter(ref_lengths, pred_lengths, alpha=0.5)
        plt.xlabel('Reference Length')
        plt.ylabel('Prediction Length')
        plt.title('Length Comparison')
        plt.plot([0, max(ref_lengths)], [0, max(ref_lengths)], 'r--', label='Perfect Match')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        n_grams = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
        scores = [bleu_scores['corpus_bleu_1'], bleu_scores['corpus_bleu_2'], 
                 bleu_scores['corpus_bleu_3'], bleu_scores['corpus_bleu_4']]
        plt.bar(n_grams, scores)
        plt.ylabel('BLEU Score')
        plt.title('N-gram BLEU Scores')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return bleu_scores
    
    def save_analysis_report(self, save_path: str):
        """
        保存分析报告
        """
        report = {
            'model_info': {
                'architecture': 'Transformer',
                'parameters': self.count_parameters(),
                'memory_usage': self.estimate_memory_usage()
            },
            'timestamp': str(torch.datetime.now())
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to {save_path}")
