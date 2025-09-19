"""
超参数配置文件
"""

# 数据预处理参数
DATA_CONFIG = {
    'data_dir': './data/formatted',
    'max_length': 64,
    'train_test_split_ratio': 0.1,
    'random_seed': 42
}

# 训练参数
TRAINING_CONFIG = {
    'num_epochs': 2,
    'base_batch_size': 64,
    'learning_rate': 3e-4, 
    'weight_decay': 1e-4,
    'num_workers': 64,
    'pin_memory': True,
    'gpu_ids': [2],
   
}


# ColFormerDynamicSGP模型配置
COLFORMER_DYNAMIC_SGP_CONFIG = {
    'input_dim': 8,
    'feature_dim': 512,
    'nhead': 8,
    'intra_message_layers': 2,
    'num_labels': 2,
    'dropout': 0.1,
    'enable_structure_guidance': True,  # 是否启用结构引导（调制方式）
    'use_structure_features': True,     # 是否使用结构特征
    'enable_ipc': True,                 # 是否启用包间协作机制（IPC）
    'structure_features': ['abc', 'entropy'],  # 可选: 'abc', 'entropy',
}



# ===== 序列模型选择配置 =====
SEQUENCE_MODEL_SELECTION = {
    'model_type': 'dynamic_sgp',  # 默认序列模型
    'dynamic_sgp_config': COLFORMER_DYNAMIC_SGP_CONFIG,
}


# ===== 兼容性配置 (保持向后兼容) =====
MODEL_SELECTION = SEQUENCE_MODEL_SELECTION  # 默认使用序列模型配置




# 文件路径
PATH_CONFIG = {
    'best_model_path': 'best_model.pth',  # 这个会在运行时动态更新
    'log_dir': 'logs',
    'checkpoint_dir': 'checkpoints'
}

# # 协议列表 mbtcp
# PROTOCOLS = ['mbtcp','icmp','smb','ntp','arp',
#              'tcp',
#              'udp','s7comm','dns','nbns']

PROTOCOLS = ['arp', 'dns', 'icmp', 'mbtcp', 'nbns', 'ntp', 's7comm', 'smb', 'tcp', 'udp']


# # 协议列表 modbus
# PROTOCOLS = ['modbus','icmp','smb','ntp','arp',
#              'tcp',
#              'udp','s7comm','dns','nbns']

# 实验配置 - 留一法交叉验证
EXPERIMENT_CONFIG = {
    # 'target_protocols': ['arp'],  # 只测试单个协议
    'target_protocols': PROTOCOLS,  # 测试所有协议
}

# 随机种子配置
REPRODUCIBILITY_CONFIG = {
    'seed': 42,
    'deterministic': True,
    'benchmark': False,
    'use_deterministic_algorithms': True
}


