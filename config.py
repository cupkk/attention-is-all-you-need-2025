"""
Transformer模型配置文件
包含论文原始配置和现代优化配置
"""

# 原论文Base Model配置
ORIGINAL_BASE_CONFIG = {
    'model_dim': 512,
    'ffn_dim': 2048,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'activation': 'relu',
    'norm_first': False,  # Post-LayerNorm
    'max_seq_len': 512,
    'vocab_size_src': 37000,
    'vocab_size_tgt': 37000
}

# 原论文Big Model配置
ORIGINAL_BIG_CONFIG = {
    'model_dim': 1024,
    'ffn_dim': 4096,
    'num_heads': 16,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dropout': 0.3,
    'attention_dropout': 0.1,
    'activation': 'relu',
    'norm_first': False,
    'max_seq_len': 512,
    'vocab_size_src': 37000,
    'vocab_size_tgt': 37000
}

# 现代优化配置
MODERN_OPTIMIZED_CONFIG = {
    'model_dim': 512,
    'ffn_dim': 2048,
    'num_heads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dropout': 0.1,
    'attention_dropout': 0.1,
    'activation': 'gelu',  # GELU激活函数
    'norm_first': True,   # Pre-LayerNorm
    'max_seq_len': 512,
    'vocab_size_src': 37000,
    'vocab_size_tgt': 37000,
    'use_learnable_pe': True,  # 是否使用可学习位置编码
    'weight_sharing': True,     # 编码器-解码器权重共享
    'layer_drop': 0.05        # LayerDrop概率
}

# 自定义配置示例
CUSTOM_CONFIG = {
    'model_dim': 768,              # 增大模型维度
    'ffn_dim': 3072,              # 对应增大FFN
    'num_heads': 12,              # 增加注意力头
    'num_encoder_layers': 8,       # 增加编码器层数
    'num_decoder_layers': 8,       # 增加解码器层数
    'dropout': 0.15,              # 增加dropout防过拟合
    'attention_dropout': 0.1,
    'activation': 'gelu',         # 使用GELU激活
    'norm_first': True,           # 使用Pre-LayerNorm
    'max_seq_len': 256,           # 减少序列长度节省内存
    'vocab_size_src': 37000,
    'vocab_size_tgt': 37000,
    'use_learnable_pe': True,     # 使用可学习位置编码
    'weight_sharing': False,      # 关闭权重共享
    'layer_drop': 0.05           # 添加LayerDrop
}

# 训练配置 - 原论文
ORIGINAL_TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'clip_grad_norm': 0.0,  # 原论文没有梯度裁剪
    'weight_decay': 0.0
}

# 训练配置 - 现代优化
MODERN_TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9,
    'warmup_steps': 4000,
    'label_smoothing': 0.1,
    'clip_grad_norm': 1.0,      # 梯度裁剪
    'weight_decay': 0.01,       # 权重衰减
    'use_amp': True,            # 混合精度训练
    'scheduler_type': 'warmup_cosine',  # 余弦退火调度
    'save_best_model': True,
    'early_stopping_patience': 10
}

# 自定义训练配置
CUSTOM_TRAINING_CONFIG = {
    'batch_size': 16,             # 减小批大小适应大模型
    'learning_rate': 8e-5,        # 降低学习率
    'beta1': 0.9,
    'beta2': 0.98,
    'eps': 1e-9,
    'warmup_steps': 6000,         # 增加warmup步数
    'label_smoothing': 0.15,      # 增加标签平滑
    'clip_grad_norm': 0.5,        # 更严格的梯度裁剪
    'weight_decay': 0.02,         # 增加权重衰减
    'use_amp': True,
    'scheduler_type': 'warmup_cosine',
    'save_best_model': True,
    'early_stopping_patience': 15
}

# 解码配置
DECODING_CONFIG = {
    'beam_size': 4,
    'max_decode_length': 100,
    'length_penalty': 0.6,
    'coverage_penalty': 0.0,
    'repetition_penalty': 1.0,
    'no_repeat_ngram_size': 0,
    'min_decode_length': 1
}

# 数据配置
DATA_CONFIG = {
    'max_src_length': 100,
    'max_tgt_length': 100,
    'share_vocab': False,
    'bpe_vocab_size': 37000,
    'min_freq': 2,
    'special_tokens': {
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'bos_token': '<s>',
        'eos_token': '</s>'
    }
}

def get_config(config_name: str = 'modern') -> dict:
    """
    获取指定配置
    
    Args:
        config_name: 配置名称 ('original_base', 'original_big', 'modern', 'custom')
    
    Returns:
        完整的配置字典
    """
    if config_name == 'original_base':
        model_config = ORIGINAL_BASE_CONFIG
        training_config = ORIGINAL_TRAINING_CONFIG
    elif config_name == 'original_big':
        model_config = ORIGINAL_BIG_CONFIG
        training_config = ORIGINAL_TRAINING_CONFIG
    elif config_name == 'modern':
        model_config = MODERN_OPTIMIZED_CONFIG
        training_config = MODERN_TRAINING_CONFIG
    elif config_name == 'custom':
        model_config = CUSTOM_CONFIG
        training_config = CUSTOM_TRAINING_CONFIG
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    return {
        'model': model_config,
        'training': training_config,
        'decoding': DECODING_CONFIG,
        'data': DATA_CONFIG
    }

def print_config_comparison():
    """
    打印配置对比
    """
    configs = ['original_base', 'original_big', 'modern', 'custom']
    
    print("=" * 80)
    print("TRANSFORMER CONFIGURATION COMPARISON")
    print("=" * 80)
    
    for config_name in configs:
        config = get_config(config_name)
        print(f"\n{config_name.upper().replace('_', ' ')} CONFIGURATION:")
        print("-" * 50)
        
        model_config = config['model']
        for key, value in model_config.items():
            print(f"  {key:20} : {value}")
        
        print("\nTraining Configuration:")
        training_config = config['training']
        for key, value in training_config.items():
            print(f"  {key:20} : {value}")

if __name__ == "__main__":
    print_config_comparison()
