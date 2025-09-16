import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from typing import List, Dict, Tuple, Iterator
from collections import defaultdict
import random
import re
import zlib
from config import DATA_CONFIG, PROTOCOLS, TRAINING_CONFIG

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_worker(worker_id):
    """DataLoader worker初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class NetworkPacketDataset(Dataset):
    """网络数据包数据集类"""
    
    def __init__(self, bit_matrices: List[np.ndarray], labels: List[np.ndarray], 
                 protocols: List[str], max_length: int, structure_features: List[np.ndarray] = None):
        self.bit_matrices = bit_matrices
        self.labels = labels
        self.protocols = protocols
        self.max_length = max_length
        self.structure_features = structure_features  # 新增：预计算的结构特征
        
        # 移除了 protocol_eval_lengths，不再限制评估长度
        
    def __len__(self):
        return len(self.bit_matrices)
    
    def __getitem__(self, idx):
        protocol = self.protocols[idx]
        
        # 不再限制评估长度，使用完整数据
        result = {
            'bit_matrix': torch.FloatTensor(self.bit_matrices[idx]),
            'labels': torch.LongTensor(self.labels[idx]),  # 使用完整标签
            'protocol': protocol,
            'eval_length': self.max_length  # 使用最大长度
        }
        
        # 添加预计算的结构特征
        if self.structure_features is not None:
            result['structure_features'] = torch.FloatTensor(self.structure_features[idx])
        
        return result

class NetworkPacketPreprocessor:
    """网络数据包预处理器"""
    
    def __init__(self, data_dir: str = None, max_length: int = None, enable_truncation: bool = True):
        self.data_dir = Path(data_dir or DATA_CONFIG['data_dir'])
        self.max_length = max_length or DATA_CONFIG['max_length']
        self.enable_truncation = enable_truncation  # 新增：是否启用截断
        self.protocols = PROTOCOLS
        self.dataset_cache_dir = Path("data/dataset")
        self.structure_window_size = 5  # 结构特征窗口大小
        
       
        
        # 协议特定的固定模式填充处理配置
        self.protocol_hash_configs = {
            'icmp': {'start_bit': 64, 'end_bit': 512},
            'dns' : {'start_bit': 96, 'end_bit': 232},
            # 'nbns' : {'start_bit': 96, 'end_bit': 368}
        }
        
        # 协议特定的有效长度限制配置
        self.protocol_length_limits = {
            'smb': 32,  # SMB协议限制有效长度为32字节
        }
        
        # 更新的协议边界标签 - 只包含header部分的分割点
        self.protocol_true_labels = {
            'udp': [1, 3, 5, 7],
            'tcp': [1, 3, 7, 11, 13, 15, 17, 19],
            's7comm': [0, 1, 3, 5, 7, 9, 10, 11, 19],  # 只保留header部分
            'arp': [1, 3, 4, 5, 7, 13, 17, 23, 27],
            'ntp': [0, 1, 2, 3, 7, 11, 15, 23, 31, 39, 47],
            'icmp': [0, 1, 3, 5, 7, 15],
            # 'modbus': [0, 2, 4],
            'mbtcp': [1, 3, 5, 6, 7, 11],
            # 'smb': [3, 4, 5, 6, 8, 9, 11, 13, 21, 23, 25, 27, 29, 31, 32, 34, 36, 38, 40, 41, 42, 44, 48, 50, 52, 54, 56, 58, 59, 60, 62, 63, 64, 66, 68],
            'smb': [3, 4, 5, 6, 8, 9, 11, 13, 21, 23, 25, 27, 29, 31],
            'dns': [1, 3, 5, 7, 9, 11, 23],
            'nbns': [1, 3, 5, 7, 9, 11, 49],
        }
        
        # 可选：保留header长度信息供参考（但不用于限制）
        self.protocol_header_info = {
            'udp': {'header_end': 8, 'description': 'UDP header固定8字节'},
            'tcp': {'header_end': 20, 'description': 'TCP基础header 20字节'},
            's7comm': {'header_end': 12, 'description': 'S7COMM header约12字节'},
            'arp': {'header_end': 28, 'description': 'ARP固定28字节'},
            'ntp': {'header_end': 48, 'description': 'NTP基础48字节'},
            'icmp': {'header_end': 8, 'description': 'ICMP header 8字节'},
            'mbtcp': {'header_end': 8, 'description': 'Modbus TCP header 8字节'},
            'smb': {'header_end': 32, 'description': 'SMB header 32字节'},
            'dns': {'header_end': 12, 'description': 'DNS header 12字节'},
            'nbns': {'header_end': 12, 'description': 'NBNS header 12字节'},
            'modbus': {'header_end': 5, 'description': 'Modbus基础5字节'},
        }
        
        # 预计算BC查找表
        self.bc_lookup_table = self._precompute_bc_table()
    
    def _precompute_bc_table(self):
        """预计算所有字节对的BC值"""
        table = np.zeros((256, 256), dtype=np.float32)
        for i in range(256):
            for j in range(256):
                xor_result = i ^ j
                same_bits = 8 - bin(xor_result).count('1')
                table[i, j] = same_bits / 8.0
        return table
    
    def compute_structure_features(self, bit_matrix: np.ndarray, feature_types: List[str] = None) -> np.ndarray:
        """预计算结构特征 - 支持选择性计算"""
        if feature_types is None:
            feature_types = ['abc', 'entropy', 'compression']
        
        seq_len, bit_dim = bit_matrix.shape
        
        # 转换为字节值
        powers = np.array([128, 64, 32, 16, 8, 4, 2, 1])
        byte_values = np.sum(bit_matrix * powers, axis=1).astype(np.int32)
        
        features = []
        
        # 根据配置选择性计算特征
        if 'abc' in feature_types:
            abc_seq = self._compute_abc_sequence(byte_values)
            features.append(abc_seq)
        
        if 'entropy' in feature_types:
            entropy_seq = self._compute_entropy_sequence(byte_values)
            features.append(entropy_seq)
            
        if 'compression' in feature_types:
            compression_seq = self._compute_compression_ratio_sequence(byte_values)
            features.append(compression_seq)
        
        # 如果没有选择任何特征，返回零特征
        if not features:
            return np.zeros((seq_len, 1), dtype=np.float32)
        
        # 堆叠为多维特征 [L, num_features]
        structure_features = np.stack(features, axis=-1)
        
        return structure_features.astype(np.float32)
    
    def _compute_abc_sequence(self, byte_values: np.ndarray) -> np.ndarray:
        """计算ABC序列"""
        seq_len = len(byte_values)
        
        if seq_len < 3:
            return np.zeros(seq_len, dtype=np.float32)
        
        # 使用查找表计算BC序列
        b1 = byte_values[:-1]
        b2 = byte_values[1:]
        bc_sequence = self.bc_lookup_table[b1, b2]
        
        # 计算ABC序列（BC的一阶差分）
        if len(bc_sequence) < 2:
            abc_sequence = np.zeros(seq_len, dtype=np.float32)
        else:
            abc_diff = bc_sequence[1:] - bc_sequence[:-1]
            abc_sequence = np.concatenate([
                np.zeros(1, dtype=np.float32),
                abc_diff,
                np.zeros(1, dtype=np.float32)
            ])
        
        return abc_sequence
    
    def _compute_entropy_sequence(self, byte_values: np.ndarray) -> np.ndarray:
        """计算局部滑动窗口熵序列"""
        seq_len = len(byte_values)
        entropy_sequence = np.zeros(seq_len, dtype=np.float32)
        
        for i in range(seq_len):
            # 定义滑动窗口
            start = max(0, i - self.structure_window_size // 2)
            end = min(seq_len, i + self.structure_window_size // 2 + 1)
            
            # 提取窗口内的字节
            window_bytes = byte_values[start:end]
            
            if len(window_bytes) > 0:
                # 计算字节值的概率分布
                unique_values, counts = np.unique(window_bytes, return_counts=True)
                probabilities = counts.astype(np.float32) / counts.sum()
                
                # 计算熵
                log_probs = np.log2(probabilities + 1e-8)
                local_entropy = -np.sum(probabilities * log_probs)
                entropy_sequence[i] = local_entropy
        
        # 归一化到 [0, 1]
        entropy_sequence = entropy_sequence / 8.0  # 最大熵为log2(256) = 8
        
        return entropy_sequence
    
    def _compute_compression_ratio_sequence(self, byte_values: np.ndarray) -> np.ndarray:
        """计算局部滑动窗口压缩比序列"""
        seq_len = len(byte_values)
        compression_sequence = np.zeros(seq_len, dtype=np.float32)
        
        for i in range(seq_len):
            # 定义滑动窗口
            start = max(0, i - self.structure_window_size // 2)
            end = min(seq_len, i + self.structure_window_size // 2 + 1)
            
            # 提取窗口内的字节
            window_bytes = byte_values[start:end].astype(np.uint8)
            
            if len(window_bytes) > 0:
                # 计算压缩比
                original_size = len(window_bytes)
                try:
                    compressed_size = len(zlib.compress(window_bytes.tobytes(), level=6))
                    compression_ratio = compressed_size / original_size
                except:
                    compression_ratio = 1.0  # 压缩失败时设为1
                
                compression_sequence[i] = compression_ratio
        
        return compression_sequence

    def get_eval_length(self, protocol: str) -> int:
        """获取评估长度 - 现在总是返回最大长度"""
        return self.max_length
    
    def load_and_merge_data(self) -> List[Dict]:
        """加载与合并数据"""
        logger.info("开始加载和合并数据...")
        all_data = []
        
        for protocol in self.protocols:
            file_path = self.data_dir / f"{protocol}_data.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        protocol_data = json.load(f)
                    logger.info(f"加载 {protocol} 协议数据: {len(protocol_data)} 条记录")
                    all_data.extend(protocol_data)
                except Exception as e:
                    logger.error(f"加载 {protocol} 数据失败: {e}")
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        logger.info(f"总共加载 {len(all_data)} 条数据记录")
        return all_data
    
    def apply_protocol_hash(self, item: dict) -> dict:
        """对特定协议的指定区间进行固定模式填充处理，同时更新相关字段"""
        protocol = item['protocol']
        if protocol not in self.protocol_hash_configs:
            return item
        
        config = self.protocol_hash_configs[protocol]
        start_bit, end_bit = config['start_bit'], config['end_bit']
        
        # 转换为字节位置
        start_byte = start_bit // 8
        end_byte = min((end_bit + 7) // 8, item['data_len'])
        
        if start_byte >= item['data_len'] or start_byte >= end_byte:
            return item
        
        # 处理十六进制数据
        hex_string = item['data'].replace(' ', '').replace('\n', '')
        if len(hex_string) % 2 != 0:
            hex_string = '0' + hex_string
        
        if len(hex_string) < end_byte * 2:
            return item
        
        # 分割数据并用固定模式填充
        prefix = hex_string[:start_byte * 2]
        hash_part = hex_string[start_byte * 2:end_byte * 2]
        suffix = hex_string[end_byte * 2:]
        
        # 使用固定的模式 "2045454542464143" 填充8字节
        pattern = "2045454542464143"  # 16个十六进制字符，正好8字节
        hash_hex = pattern[:16]  # 使用完整模式填充8字节(16个十六进制字符)
        
        # 计算新的数据长度
        old_hash_bytes = (end_byte - start_byte)
        new_hash_bytes = 8
        new_data_len = item['data_len'] - old_hash_bytes + new_hash_bytes
        
        # 更新数据和长度
        new_item = item.copy()
        new_item['data'] = prefix + hash_hex + suffix
        new_item['data_len'] = new_data_len
        
        # 更新分割位置
        if 'split_positions' in item:
            old_b_pos = item['split_positions']['b_pos'][:]
            old_e_pos = item['split_positions']['e_pos'][:]
            
            new_b_pos = []
            new_e_pos = []
            
            bit_offset = (new_hash_bytes - old_hash_bytes) * 8  # 位偏移量
            
            for b, e in zip(old_b_pos, old_e_pos):
                if e <= start_bit:
                    # 填充区间之前的位置不变
                    new_b_pos.append(b)
                    new_e_pos.append(e)
                elif b >= end_bit:
                    # 填充区间之后的位置需要调整
                    new_b_pos.append(b + bit_offset)
                    new_e_pos.append(e + bit_offset)
                elif b >= start_bit and e <= end_bit:
                    # 完全在填充区间内的字段，用填充区间替代
                    if b == start_bit and e == end_bit:
                        # 如果字段正好是整个填充区间，替换为新的填充区间
                        new_b_pos.append(start_bit)
                        new_e_pos.append(start_bit + new_hash_bytes * 8)
                    # 其他完全在填充区间内的小字段被移除（因为填充后无法保持原有结构）
                elif b < start_bit and e > start_bit and e <= end_bit:
                    # 跨越填充区间开始的字段，截断到填充区间开始
                    new_b_pos.append(b)
                    new_e_pos.append(start_bit)
                elif b >= start_bit and b < end_bit and e > end_bit:
                    # 跨越填充区间结束的字段，调整到填充区间结束后
                    new_b_pos.append(start_bit + new_hash_bytes * 8)
                    new_e_pos.append(e + bit_offset)
                elif b < start_bit and e > end_bit:
                    # 完全包含填充区间的字段，分割为两部分
                    new_b_pos.append(b)
                    new_e_pos.append(start_bit)
                    new_b_pos.append(start_bit + new_hash_bytes * 8)
                    new_e_pos.append(e + bit_offset)
            
            new_item['split_positions'] = {
                'b_pos': new_b_pos,
                'e_pos': new_e_pos
            }
        
        return new_item
    
    def apply_protocol_length_limit(self, item: dict) -> dict:
        """对特定协议应用有效长度限制"""
        protocol = item['protocol']
        if protocol not in self.protocol_length_limits:
            return item
        
        max_length = self.protocol_length_limits[protocol]
        
        if item['data_len'] > max_length:
            # 限制数据长度
            new_item = item.copy()
            new_item['data_len'] = max_length
            
            # 截断十六进制数据
            hex_string = item['data'].replace(' ', '').replace('\n', '')
            if len(hex_string) % 2 != 0:
                hex_string = '0' + hex_string
            
            # 截断到指定长度（字节数 * 2 = 十六进制字符数）
            truncated_hex = hex_string[:max_length * 2]
            new_item['data'] = truncated_hex
            
            # 更新边界标签，移除超出长度限制的标签
            if 'byte_split' in item:
                new_byte_split = [pos for pos in item['byte_split'] if pos < max_length]
                new_item['byte_split'] = new_byte_split
            
            logger.debug(f"协议 {protocol} 长度从 {item['data_len']} 截断到 {max_length}")
            return new_item
        
        return item
    
    def hex_to_bit_matrix(self, hex_string: str, data_len: int, protocol: str = None) -> np.ndarray:
        """将十六进制data转换为比特矩阵，支持协议特定固定模式填充处理"""
        try:
            # 对特定协议进行固定模式填充处理
            if protocol:
                hex_string = self.apply_protocol_hash(hex_string, data_len, protocol)
            
            hex_string = hex_string.replace(' ', '').replace('\n', '')
            if len(hex_string) % 2 != 0:
                hex_string = '0' + hex_string
            
            byte_array = bytes.fromhex(hex_string)
            
            if len(byte_array) > data_len:
                byte_array = byte_array[:data_len]
            elif len(byte_array) < data_len:
                byte_array = byte_array + b'\x00' * (data_len - len(byte_array))
            
            bit_array = np.unpackbits(np.frombuffer(byte_array, dtype=np.uint8))
            bit_matrix = bit_array.reshape(len(byte_array), 8)
            
            return bit_matrix.astype(np.float32)
            
        except Exception as e:
            logger.error(f"转换十六进制字符串失败: {hex_string[:20]}..., 错误: {e}")
            return np.zeros((data_len, 8), dtype=np.float32)
    
    def generate_boundary_labels_from_true_labels(self, protocol: str, data_len: int) -> np.ndarray:
        """根据预定义的真实标签生成边界标签 - 不再限制评估长度"""
        labels = np.zeros(data_len, dtype=np.int64)
        
        # 使用预定义的真实标签
        if protocol in self.protocol_true_labels:
            true_boundary_positions = self.protocol_true_labels[protocol]
            
            for pos in true_boundary_positions:
                if 0 <= pos < data_len:  # 只要在数据长度内就设置
                    labels[pos] = 1
            
            logger.debug(f"协议 {protocol} 使用预定义标签: {true_boundary_positions}")
        else:
            logger.warning(f"协议 {protocol} 没有预定义的真实标签")
        
        return labels
    
    def generate_boundary_labels(self, split_positions: Dict, data_len: int, protocol: str = None) -> np.ndarray:
        """根据split_positions生成边界标签 - 兼容性方法"""
        labels = np.zeros(data_len, dtype=np.int64)

        if 'e_pos' in split_positions and split_positions['e_pos']:
            e_positions = split_positions['e_pos']

            for pos in e_positions:
                if pos > 0 and pos <= data_len * 8:  # 使用完整数据长度
                    byte_index = (pos - 1) // 8
                    if 0 <= byte_index < data_len:
                        labels[byte_index] = 1

        return labels
    
    def pad_and_truncate(self, matrix: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """填充与截断 - 支持多维矩阵"""
        current_length = matrix.shape[0]
        
        if current_length < self.max_length:
            pad_length = self.max_length - current_length
            
            if matrix.ndim == 2:  # bit_matrix [L, 8]
                matrix_padded = np.vstack([
                    matrix, 
                    np.zeros((pad_length, matrix.shape[1]), dtype=matrix.dtype)
                ])
            elif matrix.ndim == 2 and matrix.shape[1] == 3:  # structure_features [L, 3]
                matrix_padded = np.vstack([
                    matrix,
                    np.zeros((pad_length, 3), dtype=matrix.dtype)
                ])
            else:
                matrix_padded = matrix
                
            labels_padded = np.concatenate([
                labels, 
                np.full(pad_length, -100, dtype=np.int64)
            ])
        else:
            matrix_padded = matrix[:self.max_length]
            labels_padded = labels[:self.max_length]
        
        return matrix_padded, labels_padded
    
    def generate_boundary_labels_from_byte_split(self, byte_split: List[int], data_len: int, protocol: str = None) -> np.ndarray:
        """根据预保存的byte_split生成边界标签 - 不再限制评估长度"""
        labels = np.zeros(data_len, dtype=np.int64)
        
        for pos in byte_split:
            if 0 <= pos < data_len:  # 使用完整数据长度
                labels[pos] = 1
        
        logger.debug(f"协议 {protocol} 使用预保存的byte_split: {byte_split}")
        
        return labels

    def save_cross_validation_datasets(self):
        """为留一法交叉验证预先划分并保存所有数据集"""
        logger.info("开始为留一法交叉验证预处理和保存数据集...")
        logger.info("使用更新的协议边界标签（只包含header部分）")
        logger.info(f"截断设置: {'启用' if self.enable_truncation else '禁用'}")
        
        all_data = self.load_and_merge_data()
        
        # 对需要固定模式填充处理的协议进行预处理
        processed_data = []
        for item in all_data:
            # 简化数据结构，只保留必要字段
            simplified_item = {
                'protocol': item['protocol'],
                'data': item['data'],
                'data_len': item['data_len']
            }
            
            # 从预定义的真实标签获取byte_split
            if item['protocol'] in self.protocol_true_labels:
                simplified_item['byte_split'] = self.protocol_true_labels[item['protocol']].copy()
            else:
                simplified_item['byte_split'] = []
                logger.warning(f"协议 {item['protocol']} 没有预定义的真实标签")
            
            # 对需要固定模式填充处理的协议进行预处理
            if item['protocol'] in self.protocol_hash_configs:
                # 创建完整的临时item用于固定模式填充处理
                temp_item = item.copy()
                processed_temp = self.apply_protocol_hash(temp_item)
                
                # 更新简化item的相关字段
                simplified_item['data'] = processed_temp['data']
                simplified_item['data_len'] = processed_temp['data_len']
                
                logger.debug(f"对 {item['protocol']} 协议数据应用固定模式填充处理")
            
            # 应用协议长度限制
            simplified_item = self.apply_protocol_length_limit(simplified_item)
            
            processed_data.append(simplified_item)
        
        for test_protocol in self.protocols:
            logger.info(f"处理测试协议: {test_protocol}")
            
            train_val_data = []
            test_data = []
            
            for item in processed_data:
                if item['protocol'] == test_protocol:
                    test_data.append(item)
                else:
                    train_val_data.append(item)
            
            if not test_data:
                logger.warning(f"协议 {test_protocol} 没有数据，跳过")
                continue
            
            if train_val_data:
                train_data, val_data = train_test_split(
                    train_val_data, 
                    test_size=DATA_CONFIG['train_test_split_ratio'], 
                    random_state=DATA_CONFIG['random_seed'],
                    stratify=[item['protocol'] for item in train_val_data]
                )
            else:
                train_data, val_data = [], []
            
            protocol_dir = self.dataset_cache_dir / test_protocol
            protocol_dir.mkdir(parents=True, exist_ok=True)
            
            datasets = {
                'train': train_data,
                'val': val_data,
                'test': test_data
            }
            
            for split_name, split_data in datasets.items():
                save_path = protocol_dir / f"{split_name}.json"
                
                # 先生成 JSON 字符串
                json_str = json.dumps(split_data, ensure_ascii=False, indent=2)
                
                # 使用正则表达式将 byte_split 从多行变为单行，并移除多余空格
                json_str = re.sub(
                    r'("byte_split":\s*\[)([\d\s,]+)(\])', 
                    lambda m: m.group(1) + ' ' + re.sub(r'\s+', ' ', m.group(2).strip()) + ' ' + m.group(3),
                    json_str
                )
                
                # 写入文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                    
                logger.info(f"保存 {test_protocol}/{split_name}: {len(split_data)} 样本")
        
        logger.info("所有交叉验证数据集保存完成")

    def load_cross_validation_dataset(self, test_protocol: str):
        """加载指定测试协议的交叉验证数据集"""
        protocol_dir = self.dataset_cache_dir / test_protocol
        
        if not protocol_dir.exists():
            logger.error(f"数据集目录不存在: {protocol_dir}")
            return None, None, None
        
        # 检查是否需要计算结构特征
        from models import get_model_name
        from config import COLFORMER_DYNAMIC_SGP_CONFIG
        need_structure_features = get_model_name() == 'ColFormerDynamicSGP'
        
        # 获取结构特征配置
        structure_feature_types = COLFORMER_DYNAMIC_SGP_CONFIG.get('structure_features', ['abc', 'entropy', 'compression'])
        
        datasets = {}
        for split_name in ['train', 'val', 'test']:
            split_path = protocol_dir / f"{split_name}.json"
            if split_path.exists():
                with open(split_path, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                bit_matrices, labels, protocols = [], [], []
                structure_features = [] if need_structure_features else None
                
                for item in split_data:
                    try:
                        # 使用简化的数据结构
                        bit_matrix = self.hex_to_bit_matrix(
                            item['data'], item['data_len']
                        )
                        
                        # 从预保存的byte_split生成边界标签
                        boundary_labels = self.generate_boundary_labels_from_byte_split(
                            item['byte_split'], item['data_len'], item['protocol']
                        )
                        
                        bit_matrix_processed, labels_processed = self.pad_and_truncate(
                            bit_matrix, boundary_labels
                        )
                        
                        bit_matrices.append(bit_matrix_processed)
                        labels.append(labels_processed)
                        protocols.append(item['protocol'])
                        
                        # 只在需要时预计算结构特征，使用配置的特征类型
                        if need_structure_features:
                            structure_feat = self.compute_structure_features(bit_matrix, structure_feature_types)
                            structure_feat_processed, _ = self.pad_and_truncate(
                                structure_feat, np.zeros(structure_feat.shape[0])  # 占位符
                            )
                            structure_features.append(structure_feat_processed)
                            
                    except Exception as e:
                        logger.error(f"处理数据失败: {e}")
                        continue
                
                datasets[split_name] = NetworkPacketDataset(
                    bit_matrices, labels, protocols, self.max_length, structure_features
                )
                
                logger.info(f"加载 {test_protocol}/{split_name}: {len(datasets[split_name])} 样本")
            else:
                logger.warning(f"文件不存在: {split_path}")
                datasets[split_name] = None
        
        return datasets.get('train'), datasets.get('val'), datasets.get('test')

# ProtocolGroupSampler 和 create_data_loaders 保持不变，它们不需要修改

class ProtocolGroupSampler(Sampler):
    """协议分组采样器 - 保持不变"""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 按协议类型分组索引
        self.protocol_groups = defaultdict(list)
        for idx in range(len(dataset)):
            protocol = dataset.protocols[idx]
            self.protocol_groups[protocol].append(idx)
        
        # 计算每个协议的batch数量
        self.protocol_batches = {}
        self.total_batches = 0
        
        for protocol, indices in self.protocol_groups.items():
            num_samples = len(indices)
            if self.drop_last:
                num_batches = num_samples // self.batch_size
            else:
                num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            
            self.protocol_batches[protocol] = num_batches
            self.total_batches += num_batches
    
    def __iter__(self) -> Iterator[List[int]]:
        all_batches = []
        
        for protocol, indices in self.protocol_groups.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                
                all_batches.append(batch_indices)
        
        if self.shuffle:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self) -> int:
        return self.total_batches

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=None):
    """创建数据加载器 - 保持不变"""
    batch_size = batch_size or TRAINING_CONFIG['base_batch_size']
    num_workers = TRAINING_CONFIG['num_workers']
    pin_memory = TRAINING_CONFIG['pin_memory']
    
    g = torch.Generator()
    g.manual_seed(DATA_CONFIG['random_seed'])
    
    def colformer_collate_fn(batch):
        bit_matrices = torch.stack([item['bit_matrix'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        protocols = [item['protocol'] for item in batch]
        eval_lengths = [item['eval_length'] for item in batch]
        
        result = {
            'bit_matrix': bit_matrices,
            'labels': labels,
            'protocols': protocols,
            'eval_lengths': eval_lengths
        }
        
        from models import get_model_name
        if (get_model_name() == 'ColFormerDynamicSGP' and 
            'structure_features' in batch[0] and 
            batch[0]['structure_features'] is not None):
            structure_features = torch.stack([item['structure_features'] for item in batch])
            result['structure_features'] = structure_features
        
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ProtocolGroupSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=colformer_collate_fn
    ) if train_dataset else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=ProtocolGroupSampler(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=colformer_collate_fn
    ) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=ProtocolGroupSampler(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        collate_fn=colformer_collate_fn
    ) if test_dataset else None
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # 使用示例，支持配置是否启用截断
    preprocessor = NetworkPacketPreprocessor(enable_truncation=True)  # 可以设置为False禁用截断
    preprocessor.save_cross_validation_datasets()
