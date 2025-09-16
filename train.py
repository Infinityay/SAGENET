import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime
import random
import torch.nn.functional as F
from models import create_model, get_model_name
from preparation import NetworkPacketPreprocessor, create_data_loaders
from config import (
    DATA_CONFIG, TRAINING_CONFIG, PATH_CONFIG,
    COLFORMER_DYNAMIC_SGP_CONFIG  
)


def set_seed(seed=42):
    """设置全局随机种子以确保可重现性"""
    # Python随机数
    random.seed(seed)
    
    # NumPy随机数
    np.random.seed(seed)
    
    # PyTorch随机数
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 使用确定性算法 - 但对不支持的操作只发出警告
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"随机种子已设置为: {seed}")

# 设置日志 - 同时输出到控制台和文件
def setup_logging():
    # 创建按日期分类的logs目录
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(PATH_CONFIG['log_dir']) / current_date
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 清除现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志将保存到: {log_file}")
    return logger

logger = setup_logging()


class ProtocolTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # 早停配置 - 修改为基于F1改进的早停
        self.early_stop_patience = 3       # 连续没有改进的epoch数
        self.early_stop_counter = 0        # 计数器

        
        # 优化器 - 使用更稳定的学习率
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=TRAINING_CONFIG['learning_rate'], 
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        logger.info("使用标准交叉熵损失")

        self.scheduler = None
        logger.info("使用固定学习率训练")

        # 记录最佳模型（仅用于日志显示）
        self.best_val_f1 = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_model_path = PATH_CONFIG['best_model_path']
        
        # 训练历史记录
        self.train_history = {
            'train_loss': [], 'train_f1': [],
            'val_loss': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
            'learning_rates': []
        }
        
    def train_epoch(self):
        """训练一个epoch - 使用改进的分组共识评估"""
        self.model.train()
        total_loss = 0
        protocol_results = {}
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            bit_matrices = batch['bit_matrix'].to(self.device)
            labels = batch['labels'].to(self.device)
            protocols = batch['protocols']
            
            attention_mask = (bit_matrices.sum(dim=-1) > 0)
            
            # 获取结构特征
            structure_features = batch.get('structure_features')
            if structure_features is not None:
                structure_features = structure_features.to(self.device)
            
            # 只调用 ColFormerDynamicSGP 模型
            logits = self.model(
                bit_matrix=bit_matrices,
                structure_features=structure_features,
                attention_mask=attention_mask
            )
                 
            
            # 只计算交叉熵损失
            ce_loss = self.criterion(logits.view(-1, 2), labels.view(-1))
            
            # 反向传播
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累计损失
            total_loss += ce_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': ce_loss.item()
            })
            
            # 收集预测结果用于计算共识指标
            with torch.no_grad():
                preds = torch.argmax(logits, dim=2)
                mask = labels != -100
                
                for i in range(len(protocols)):
                    protocol = protocols[i]
                    sample_mask = mask[i]
                    
                    if sample_mask.any():
                        sample_preds = preds[i][sample_mask].cpu().numpy()
                        sample_labels = labels[i][sample_mask].cpu().numpy()

                        if protocol not in protocol_results:
                            protocol_results[protocol] = {'preds': [], 'labels': []}
                        
                        protocol_results[protocol]['preds'].append(sample_preds.tolist())
                        protocol_results[protocol]['labels'].append(sample_labels.tolist())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        # 使用改进的分组共识评估计算训练F1
        protocol_f1_scores = []
        for protocol, data in protocol_results.items():
            if data['labels'] and len(data['preds']) >= 1:
                metrics = self.evaluate_consensus_prediction(
                    data['preds'], data['labels'], threshold_ratio=0.5
                )
                protocol_f1_scores.append(metrics['f1'])
        
        train_f1 = np.mean(protocol_f1_scores) if protocol_f1_scores else 0.0
        
        # 记录损失
        logger.info(f"训练损失: {avg_loss:.4f}")
        
        return avg_loss, train_f1
    
    def validate(self, data_loader, desc="Validation"):
        """验证模型 -  使用改进的分组共识评估"""
        self.model.eval()
        total_loss = 0
        protocol_results = {}

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=desc):
                bit_matrices = batch['bit_matrix'].to(self.device)
                labels = batch['labels'].to(self.device)
                protocols = batch['protocols']
                
                attention_mask = (bit_matrices.sum(dim=-1) > 0)
            
                # 获取结构特征
                structure_features = batch.get('structure_features')
                if structure_features is not None:
                    structure_features = structure_features.to(self.device)
                
                # 只调用 ColFormerDynamicSGP 模型
                logits = self.model(
                    bit_matrix=bit_matrices,
                    structure_features=structure_features,
                    attention_mask=attention_mask
                )

                loss = self.criterion(logits.view(-1, 2), labels.view(-1))
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=2)
                mask = labels != -100

                # 收集预测结果
                for i in range(len(protocols)):
                    protocol = protocols[i]
                    sample_mask = mask[i]
                    
                    if sample_mask.any():
                        sample_preds = preds[i][sample_mask].cpu().numpy()
                        sample_labels = labels[i][sample_mask].cpu().numpy()

                        if protocol not in protocol_results:
                            protocol_results[protocol] = {'preds': [], 'labels': []}
                        
                        protocol_results[protocol]['preds'].append(sample_preds.tolist())
                        protocol_results[protocol]['labels'].append(sample_labels.tolist())

        avg_loss = total_loss / len(data_loader)

        # 计算每个协议的分组分组共识指标
        protocol_metrics = {}
        
        for protocol, data in protocol_results.items():
            if data['labels'] and len(data['preds']) >= 1:
                # 使用改进的分组改进的分组共识评估
                metrics = self.evaluate_consensus_prediction(
                    data['preds'], data['labels'], threshold_ratio=0.5
                )
                protocol_metrics[protocol] = metrics
                
                
        # 计算总体指标 - 从各协议指标计算平均值
        if protocol_metrics:
            overall_metrics = {
                'precision': np.mean([m['precision'] for m in protocol_metrics.values()]),
                'recall': np.mean([m['recall'] for m in protocol_metrics.values()]),
                'f1': np.mean([m['f1'] for m in protocol_metrics.values()]),
                'tp': sum([m['tp'] for m in protocol_metrics.values()]),
                'fp': sum([m['fp'] for m in protocol_metrics.values()]),
                'fn': sum([m['fn'] for m in protocol_metrics.values()])
            }
        else:
            overall_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        return {
            'loss': avg_loss,
            'overall': overall_metrics,
            'by_protocol': protocol_metrics
        }

    
    def calculate_byte_level_metrics(self, all_preds, all_labels):
        """
        层次一：字节级边界评估 (Byte-level Boundary Evaluation)

        计算字节级边界评估的核心指标：
        - 精确率 (Precision): TP / (TP + FP)
        - 召回率 (Recall): TP / (TP + FN)
        - F1分数 (F1-Score): 2 * (Precision * Recall) / (Precision + Recall)
        """
        if len(all_preds) == 0 or len(all_labels) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        # 转换为numpy数组
        preds = np.array(all_preds)
        labels = np.array(all_labels)

        # 第2步：逐点比较与计数
        # 遍历每一个位置，累加三种情况的计数
        tp = np.sum((preds == 1) & (labels == 1))  # 真阳性：模型正确找到边界
        fp = np.sum((preds == 1) & (labels == 0))  # 假阳性：模型错误创造边界
        fn = np.sum((preds == 0) & (labels == 1))  # 假阴性：模型遗漏边界

        # 第3步：计算核心指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def evaluate_consensus_prediction(self, predictions: list, ground_truths: list, threshold_ratio=0.5):
        """改进的共识评估：按长度分组分别计算共识，然后加权平均"""
        if len(predictions) < 2:
            return self.calculate_byte_level_metrics(
                [item for sublist in predictions for item in sublist],
                [item for sublist in ground_truths for item in sublist]
            )

        # Step 1: 按长度分组
        length_groups = {}
        for pred, gt in zip(predictions, ground_truths):
            length = len(pred)
            if length not in length_groups:
                length_groups[length] = {'preds': [], 'gts': [], 'count': 0}
            length_groups[length]['preds'].append(pred)
            length_groups[length]['gts'].append(gt)
            length_groups[length]['count'] += 1
        
        logger.debug(f"长度分布: {[(length, group['count']) for length, group in length_groups.items()]}")
        
        # Step 2: 对每个长度组分别计算共识
        group_metrics = {}
        total_samples = sum(group['count'] for group in length_groups.values())
        
        for length, group in length_groups.items():
            if group['count'] >= 2:  # 至少需要2个样本才能计算共识
                # 计算该长度组的共识
                consensus_pred = self._calculate_consensus_sequence(group['preds'], threshold_ratio)
                consensus_gt = self._calculate_consensus_sequence(group['gts'], 0.5)
                
                metrics = self.calculate_byte_level_metrics(consensus_pred, consensus_gt)
                metrics['sample_count'] = group['count']
                metrics['weight'] = group['count'] / total_samples
                group_metrics[length] = metrics
                
                logger.debug(f"长度{length}: {group['count']}个样本, F1={metrics['f1']:.4f}, 权重={metrics['weight']:.3f}")
        
        # Step 3: 加权平均计算总体指标
        if not group_metrics:
            logger.warning("没有足够样本的长度组进行共识评估")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}
        
        weighted_precision = sum(m['precision'] * m['weight'] for m in group_metrics.values())
        weighted_recall = sum(m['recall'] * m['weight'] for m in group_metrics.values())
        weighted_f1 = sum(m['f1'] * m['weight'] for m in group_metrics.values())
        
        total_tp = sum(m['tp'] for m in group_metrics.values())
        total_fp = sum(m['fp'] for m in group_metrics.values())
        total_fn = sum(m['fn'] for m in group_metrics.values())
        
        # 记录详细信息
        length_info = [f"{length}字节:{group['count']}个" for length, group in length_groups.items()]
        logger.debug(f"分组共识评估完成 - {', '.join(length_info)}")
        
        return {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'length_groups': group_metrics,
            'total_samples': total_samples
        }
    
    
    def _calculate_consensus_sequence(self, sequences, threshold_ratio):
        """计算序列的共识"""
        if not sequences:
            return []
        
        length = len(sequences[0])
        num_sequences = len(sequences)
        consensus = []
        
        for pos in range(length):
            votes = sum(seq[pos] for seq in sequences)
            consensus.append(1 if votes >= (num_sequences * threshold_ratio) else 0)
        
        return consensus
    
    def save_best_model(self, epoch, val_f1, val_loss, val_precision, val_recall):
        """保存最佳模型：F1最高，F1相同时选择val loss最低的"""
        is_best = False
        
        if val_f1 > self.best_val_f1:
            # F1更高，直接保存
            is_best = True
            logger.info(f"🎉 新的最佳F1: {val_f1:.4f} (上一次: {self.best_val_f1:.4f})")
        elif val_f1 == self.best_val_f1 and val_loss < self.best_val_loss:
            # F1相同但验证损失更低，也保存
            is_best = True
            logger.info(f"🎉 相同F1但更低验证损失: {val_loss:.4f} (上一次: {self.best_val_loss:.4f})")
        
        if is_best:
            self.best_val_f1 = val_f1
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            
            # 保存最佳模型
            model_state_dict = (self.model.module.state_dict() 
                              if isinstance(self.model, nn.DataParallel) 
                              else self.model.state_dict())
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_f1': float(val_f1),
                'val_precision': float(val_precision),
                'val_recall': float(val_recall),
                'val_loss': float(val_loss),
                'train_history': self.train_history,
                'is_best_model': True
            }
            
            # 只有当scheduler存在时才保存其状态
            if self.scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint_data, self.best_model_path)
            
            logger.info(f"💾 最佳模型已保存到: {self.best_model_path}")
        
        return is_best

    def train(self, num_epochs=5):
        """完整训练流程 - 修改早停逻辑为基于F1改进"""
        logger.info(f"开始训练，共 {num_epochs} 个epoch")

        for epoch in range(num_epochs):
            logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # 训练
            train_loss, train_f1 = self.train_epoch()
            
            # 验证
            val_results = self.validate(self.val_loader, "Validation")
            
            # 记录历史
            overall = val_results['overall']
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_f1'].append(train_f1)
            self.train_history['val_loss'].append(val_results['loss'])
            self.train_history['val_f1'].append(overall['f1'])
            self.train_history['val_precision'].append(overall['precision'])
            self.train_history['val_recall'].append(overall['recall'])
            self.train_history['learning_rates'].append(current_lr)
            
            # 记录结果
            logger.info(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            logger.info(f"Val Overall - Loss: {val_results['loss']:.4f}, F1: {overall['f1']:.4f}, "
                       f"Precision: {overall['precision']:.4f}, Recall: {overall['recall']:.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # 显示各协议指标
            logger.info("各协议验证指标:")
            for protocol, metrics in val_results['by_protocol'].items():
                logger.info(f"  {protocol:>8} - F1: {metrics['f1']:.4f}")
            
            # 保存最佳模型并检查是否有改进
            is_best = self.save_best_model(
                epoch, overall['f1'], val_results['loss'], 
                overall['precision'], overall['recall']
            )
            
            # 早停检查 - 基于F1改进
            if is_best:
                # 有改进，重置计数器
                self.early_stop_counter = 0
                logger.info(f"✅ 模型有改进，重置早停计数器")
            else:
                # 无改进，增加计数器
                self.early_stop_counter += 1
                logger.info(f"⚠️  连续{self.early_stop_counter}个epoch无改进 "
                           f"(当前F1: {overall['f1']:.4f}, 最佳F1: {self.best_val_f1:.4f})")
                
                if self.early_stop_counter >= self.early_stop_patience:
                    logger.info(f"🛑 连续{self.early_stop_patience}个epoch无改进，提前停止训练")
                    break
        
        logger.info(f"\n=== 训练完成 ===")
        if self.early_stop_counter >= self.early_stop_patience:
            logger.info(f"提前停止于 Epoch {len(self.train_history['train_loss'])} (无改进)")
        else:
            logger.info(f"正常完成训练 (Epoch {len(self.train_history['train_loss'])})")
        
        logger.info(f"最佳模型来自 Epoch {self.best_epoch + 1}")
        logger.info(f"最佳验证F1: {self.best_val_f1:.4f}")
        logger.info(f"最佳验证Loss: {self.best_val_loss:.4f}")
        logger.info(f"总训练轮数: {len(self.train_history['train_loss'])}")
        
        # 绘制训练曲线
        self.plot_training_curves()

    def plot_training_curves(self):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            epochs = range(1, len(self.train_history['train_loss']) + 1)
            
            # 损失曲线
            axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
            axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # F1分数曲线
            axes[0, 1].plot(epochs, self.train_history['train_f1'], 'b-', label='Train F1')
            axes[0, 1].plot(epochs, self.train_history['val_f1'], 'r-', label='Val F1')
            axes[0, 1].set_title('Training and Validation F1 Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 精确率和召回率
            axes[1, 0].plot(epochs, self.train_history['val_precision'], 'g-', label='Precision')
            axes[1, 0].plot(epochs, self.train_history['val_recall'], 'orange', label='Recall')
            axes[1, 0].plot(epochs, self.train_history['val_f1'], 'r-', label='F1')
            axes[1, 0].set_title('Validation Metrics')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 学习率曲线
            axes[1, 1].plot(epochs, self.train_history['learning_rates'], 'purple', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plots_dir / f"training_curves_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {plot_path}")
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib未安装，跳过训练曲线绘制")
        except Exception as e:
            logger.error(f"绘制训练曲线时出错: {e}")

    def load_checkpoint(self, checkpoint_path=None):
        """加载检查点，支持恢复训练"""
        if checkpoint_path is None:
            checkpoint_path = self.best_model_path
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            
            # 处理DataParallel模型的state_dict键名不匹配问题
            model_state_dict = checkpoint['model_state_dict']
            
            is_model_parallel = isinstance(self.model, nn.DataParallel)
            has_module_prefix = any(key.startswith('module.') for key in model_state_dict.keys())
            
            if is_model_parallel and not has_module_prefix:
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            elif not is_model_parallel and has_module_prefix:
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            
            self.model.load_state_dict(model_state_dict)
              
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
            
            self.best_val_f1 = checkpoint.get('val_f1', 0.0)
            
            logger.info(f"成功加载检查点: {checkpoint_path}")
            logger.info(f"最佳验证F1: {self.best_val_f1:.4f}")
            return checkpoint.get('epoch', 0)
        else:
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return 0

    def test(self):
        """测试模型性能 - 始终加载最佳保存的模型"""
        # 加载最佳模型
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # 处理DataParallel模型的state_dict键名不匹配问题
            is_model_parallel = isinstance(self.model, nn.DataParallel)
            has_module_prefix = any(key.startswith('module.') for key in model_state_dict.keys())
            
            if is_model_parallel and not has_module_prefix:
                model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
            elif not is_model_parallel and has_module_prefix:
                model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            
            self.model.load_state_dict(model_state_dict)
            
            best_epoch = checkpoint.get('epoch', 0) + 1
            best_f1 = checkpoint.get('val_f1', 0.0)
            best_loss = checkpoint.get('val_loss', 0.0)
            
            logger.info(f"✅ 加载最佳模型 (Epoch {best_epoch})")
            logger.info(f"   验证F1: {best_f1:.4f}, 验证Loss: {best_loss:.4f}")
        else:
            logger.warning(f"最佳模型文件不存在: {self.best_model_path}")
            logger.info("使用当前训练状态的模型进行测试")
        
        # 测试集测试
        test_results = None
        if self.test_loader:
            logger.info(f"\n--- 未知协议测试 - 字节级边界评估 (共识评估) ---")
            test_results = self.validate(self.test_loader, "Unknown Protocols Test")
            
            logger.info("\n各协议详细指标:")
            for protocol, metrics in test_results['by_protocol'].items():
                logger.info(f"{protocol:>8} - F1: {metrics['f1']:.4f}, "
                           f"Prec: {metrics['precision']:.4f}, "
                           f"Recall: {metrics['recall']:.4f}")
        
        self.show_protocol_examples()
        
        return test_results

    def show_protocol_examples(self):
        """展示各协议的共识预测格式 - 详细分析版本"""
        logger.info("\n" + "="*80)
        logger.info("协议共识格式分析")
        logger.info("="*80)

        self.model.eval()
        
        from config import PROTOCOLS
        target_protocols = PROTOCOLS
        
        protocol_consensus = {}
        
        if self.test_loader is None:
            logger.info("没有测试集数据")
            return

        with torch.no_grad():
            protocol_results = {}
            
            for batch in self.test_loader:
                # 统一处理输入
                bit_matrices = batch['bit_matrix'].to(self.device)
                labels = batch['labels'].to(self.device)
                protocols = batch['protocols']
                
                attention_mask = (bit_matrices.sum(dim=-1) > 0)
                
                # 获取结构特征
                structure_features = batch.get('structure_features')
                if structure_features is not None:
                    structure_features = structure_features.to(self.device)
                
                # 只调用 ColFormerDynamicSGP 模型
                logits = self.model(
                    bit_matrix=bit_matrices,
                    structure_features=structure_features,
                    attention_mask=attention_mask
                )

       

                preds = torch.argmax(logits, dim=2)
                mask = labels != -100

                for i in range(len(protocols)):
                    protocol = protocols[i]
                    sample_mask = mask[i]
                    
                    if sample_mask.any() and protocol in target_protocols:
                        sample_preds = preds[i][sample_mask].cpu().numpy()
                        sample_labels = labels[i][sample_mask].cpu().numpy()
                        sample_bits = bit_matrices[i][sample_mask].cpu().numpy()

                        if protocol not in protocol_results:
                            protocol_results[protocol] = {
                                'preds': [], 'labels': [], 'bits': []
                            }
                        
                        protocol_results[protocol]['preds'].append(sample_preds.tolist())
                        protocol_results[protocol]['labels'].append(sample_labels.tolist())
                        protocol_results[protocol]['bits'].append(sample_bits)

            # 计算每个协议的共识格式（按长度分组）
            for protocol, data in protocol_results.items():
                if len(data['preds']) >= 2:
                    # 按长度分组分析
                    length_groups = {}
                    for i, pred in enumerate(data['preds']):
                        length = len(pred)
                        if length not in length_groups:
                            length_groups[length] = {'count': 0, 'preds': [], 'labels': [], 'bits': []}
                        length_groups[length]['count'] += 1
                        length_groups[length]['preds'].append(pred)
                        length_groups[length]['labels'].append(data['labels'][i])
                        length_groups[length]['bits'].append(data['bits'][i])
                    
                    protocol_consensus[protocol] = length_groups

        # 显示各协议的详细分析
        for protocol in target_protocols:
            if protocol in protocol_consensus:
                logger.info(f"\n--- {protocol.upper()} 协议 ---")
                length_groups = protocol_consensus[protocol]
                
                # 找到样本数最多的组
                max_count = 0
                best_group = None
                best_length = None
                
                for length, group in length_groups.items():
                    if group['count'] > max_count and group['count'] >= 2:  # 至少2个样本
                        max_count = group['count']
                        best_group = group
                        best_length = length
                
                if best_group:
                    # 显示基本信息
                    logger.info(f"样本数: {max_count}")
                    logger.info(f"有效长度: {best_length}")
                    
                    # 获取共识格式
                    pred_format, true_format, detailed_stats = self._get_detailed_consensus_format(
                        best_group['preds'], best_group['labels'], best_group['bits'][0]
                    )
                    
                    logger.info(f"  预测格式: {pred_format}")
                    logger.info(f"  真实格式: {true_format}")
                    
                    # 显示详细的投票统计
                    self._show_voting_statistics(detailed_stats, max_count)
                    
                    # 显示边界位置摘要
                    self._show_boundary_summary(detailed_stats)
                    
                else:
                    logger.info("没有足够样本的长度组进行分析")
            
  

    def _get_detailed_consensus_format(self, predictions, ground_truths, sample_bits):
        """为特定长度组计算详细的共识格式和统计信息"""
        if len(predictions) < 2:
            return "样本不足", "样本不足", None
        
        length = len(predictions[0])
        num_predictions = len(predictions)
        
        # 收集每个位置的投票统计
        position_stats = []
        consensus_pred_seq = []
        consensus_gt_seq = []
        
        for pos in range(length):
            pred_votes = sum(seq[pos] for seq in predictions)
            gt_votes = sum(seq[pos] for seq in ground_truths)
            
            pred_boundary = pred_votes >= (num_predictions * 0.5)
            gt_boundary = gt_votes >= (num_predictions * 0.5)
            
            consensus_pred_seq.append(1 if pred_boundary else 0)
            consensus_gt_seq.append(1 if gt_boundary else 0)
            
            position_stats.append({
                'position': pos,
                'pred_votes': pred_votes,
                'gt_votes': gt_votes,
                'pred_boundary': pred_boundary,
                'gt_boundary': gt_boundary
            })
        
        # 重构十六进制数据
        hex_data = ""
        for i in range(length):
            byte_bits = sample_bits[i]
            byte_value = 0
            for j in range(8):
                if byte_bits[j] > 0.5:
                    byte_value |= (1 << (7-j))
            hex_data += f"{byte_value:02x}"
        
        pred_format = self._format_fields(hex_data, consensus_pred_seq)
        true_format = self._format_fields(hex_data, consensus_gt_seq)
        
        detailed_stats = {
            'position_stats': position_stats,
            'consensus_pred_seq': consensus_pred_seq,
            'consensus_gt_seq': consensus_gt_seq,
            'total_samples': num_predictions,
            'length': length
        }
        
        return pred_format, true_format, detailed_stats

    def _show_voting_statistics(self, detailed_stats, total_samples):
        """显示详细的投票统计表格"""
        logger.info(f"\n位置投票统计 (总样本数: {total_samples}):")
        logger.info("位置  预测投票  真实投票  预测边界  真实边界")
        logger.info("-" * 50)
        
        position_stats = detailed_stats['position_stats']
        length = detailed_stats['length']
        
        # 显示前20个位置，如果超过则显示省略
        display_limit = 20
        for i, stats in enumerate(position_stats[:display_limit]):
            pred_votes = stats['pred_votes']
            gt_votes = stats['gt_votes']
            pred_boundary = "是" if stats['pred_boundary'] else "否"
            gt_boundary = "是" if stats['gt_boundary'] else "否"
            
            logger.info(f"{stats['position']:>4}   {pred_votes:>6.1f}     {gt_votes:>6.1f}      {pred_boundary:>2}       {gt_boundary:>2}")
        
        if length > display_limit:
            logger.info(f"... (还有 {length - display_limit} 个位置)")

    def _show_boundary_summary(self, detailed_stats):
        """显示边界位置摘要"""
        position_stats = detailed_stats['position_stats']
        
        # 提取边界位置
        pred_boundaries = [stats['position'] for stats in position_stats if stats['pred_boundary']]
        gt_boundaries = [stats['position'] for stats in position_stats if stats['gt_boundary']]
        
        # 计算投票强度统计
        pred_votes = [stats['pred_votes'] for stats in position_stats]
        gt_votes = [stats['gt_votes'] for stats in position_stats]
        
        logger.info("\n边界位置摘要:")
        logger.info(f"  预测边界位置: {pred_boundaries}")
        logger.info(f"  真实边界位置: {gt_boundaries}")
        
        if pred_votes:
            logger.info(f"  预测投票强度: 平均={np.mean(pred_votes):.1f}, "
                       f"最大={np.max(pred_votes):.1f}, 最小={np.min(pred_votes):.1f}")
        
        if gt_votes:
            logger.info(f"  真实投票强度: 平均={np.mean(gt_votes):.1f}, "
                       f"最大={np.max(gt_votes):.1f}, 最小={np.min(gt_votes):.1f}")
        
        # 计算匹配度
        pred_set = set(pred_boundaries)
        gt_set = set(gt_boundaries)
        
        correct_boundaries = len(pred_set & gt_set)
        missed_boundaries = len(gt_set - pred_set)
        false_boundaries = len(pred_set - gt_set)
        
        logger.info(f"  边界匹配情况: 正确={correct_boundaries}, "
                   f"遗漏={missed_boundaries}, 误报={false_boundaries}")
        
        # 计算边界级别的精确率和召回率
        if len(pred_set) > 0:
            boundary_precision = correct_boundaries / len(pred_set)
            logger.info(f"  边界精确率: {boundary_precision:.3f}")
        
        if len(gt_set) > 0:
            boundary_recall = correct_boundaries / len(gt_set)
            logger.info(f"  边界召回率: {boundary_recall:.3f}")

    

    def _get_consensus_format_for_group(self, predictions, ground_truths, sample_bits):
        """为特定长度组计算共识格式"""
        if len(predictions) < 2:
            return "样本不足", "样本不足"
        
        length = len(predictions[0])
        num_predictions = len(predictions)
        consensus_pred_seq = []
        consensus_gt_seq = []
        
        for pos in range(length):
            pred_votes = sum(seq[pos] for seq in predictions)
            consensus_pred_seq.append(1 if pred_votes >= (num_predictions * 0.5) else 0)
            
            gt_votes = sum(seq[pos] for seq in ground_truths)
            consensus_gt_seq.append(1 if gt_votes >= (num_predictions * 0.5) else 0)
        
        # 重构十六进制数据
        hex_data = ""
        for i in range(length):
            byte_bits = sample_bits[i]
            byte_value = 0
            for j in range(8):
                if byte_bits[j] > 0.5:
                    byte_value |= (1 << (7-j))
            hex_data += f"{byte_value:02x}"
        
        pred_format = self._format_fields(hex_data, consensus_pred_seq)
        true_format = self._format_fields(hex_data, consensus_gt_seq)
        
        return pred_format, true_format

    def _format_fields(self, hex_data, boundary_seq):
        """根据边界序列格式化字段"""
        boundary_positions = [i for i, label in enumerate(boundary_seq) if label == 1]
        
        if not boundary_positions:
            return hex_data  # 没有边界，整个数据作为一个字段
        
        fields = []
        start_pos = 0
        
        for boundary_pos in boundary_positions:
            end_pos = boundary_pos + 1
            field_hex = hex_data[start_pos*2:end_pos*2]
            if field_hex:
                fields.append(field_hex)
            start_pos = end_pos
        
        # 处理最后一个字段
        if start_pos < len(boundary_seq):
            remaining_hex = hex_data[start_pos*2:]
            if remaining_hex:
                fields.append(remaining_hex)
        
        return " | ".join(fields) if fields else hex_data

def run_cross_validation():
    """运行留一法交叉验证实验"""
    from config import EXPERIMENT_CONFIG
    
    target_protocols = EXPERIMENT_CONFIG['target_protocols']
    all_results = {}
    
    logger.info("="*80)
    logger.info("开始留一法交叉验证实验")
    logger.info(f"目标协议: {target_protocols}")
    logger.info("="*80)
    logger.info("使用共识评估模式")
    
    # 检查是否需要预处理数据
    preprocessor = NetworkPacketPreprocessor()
    dataset_cache_dir = Path("data/dataset")
    
    if not dataset_cache_dir.exists() or not any(dataset_cache_dir.iterdir()):
        logger.info("未找到预处理的数据集，开始预处理...")
        preprocessor.save_cross_validation_datasets()
    else:
        logger.info("发现已预处理的数据集，直接使用")
    
    model_name = get_model_name()
    logger.info(f"使用模型: {model_name}")
    
    # 检查是否有已训练的模型
    checkpoint_dir = Path(PATH_CONFIG['checkpoint_dir']) / model_name
    existing_models = []
    if checkpoint_dir.exists():
        existing_models = [f for f in checkpoint_dir.glob("*_best.pth") if f.is_file()]
    
    retrain_all = False
    if existing_models:
        logger.info(f"\n发现 {len(existing_models)} 个已训练的模型:")
        for model_file in existing_models:
            protocol = model_file.stem.replace('_best', '')
            logger.info(f"  - {protocol}")
    
        retrain_all = True  # 默认重新训练所有模型，便于测试
    
    for i, test_protocol in enumerate(target_protocols):
        logger.info(f"\n{'='*60}")
        logger.info(f"实验 {i+1}/{len(target_protocols)}: 测试协议 {test_protocol}")
        logger.info(f"{'='*60}")
        
        # 加载预处理的数据集
        train_dataset, val_dataset, test_dataset = preprocessor.load_cross_validation_dataset(
            test_protocol
        )

        if train_dataset is None or val_dataset is None or test_dataset is None:
            logger.error(f"加载协议 {test_protocol} 数据集失败，跳过")
            continue

        # 统一使用协议分组数据加载器 - 适配同协议流量场景
        logger.info("使用协议分组数据加载器 (适配同协议流量场景)")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, 
            batch_size=TRAINING_CONFIG['base_batch_size']
        )
        
        # 更新模型保存路径 - 按模型类型分类
        checkpoint_dir = Path(PATH_CONFIG['checkpoint_dir']) / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f"{test_protocol}_best.pth"
        PATH_CONFIG['best_model_path'] = str(model_path)
        
        logger.info(f"模型保存路径: {model_path}")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型
        model = create_model().to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # 创建训练器
        trainer = ProtocolTrainer(model, train_loader, val_loader, test_loader, device)
        
        # 根据用户选择决定是否训练
        if os.path.exists(model_path) and not retrain_all:
            logger.info(f"使用已训练模型: {model_path}")
            test_results = trainer.test()
        else:
            if os.path.exists(model_path):
                logger.info(f"重新训练模型 (未知协议: {test_protocol})")
            else:
                logger.info(f"开始训练 (未知协议: {test_protocol})")
            trainer.train(num_epochs=TRAINING_CONFIG['num_epochs'])
            test_results = trainer.test()
        
        if test_results:
            all_results[test_protocol] = test_results['overall']
        
    
    # 输出最终结果汇总
    if all_results:
        logger.info("\n" + "="*80)
        logger.info(f"留一法交叉验证最终结果汇总 - {model_name}")
        logger.info("="*80)
        
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for protocol, metrics in all_results.items():
            f1_scores.append(metrics['f1'])
            precision_scores.append(metrics['precision'])
            recall_scores.append(metrics['recall'])
            logger.info(f"{protocol:>8} - F1: {metrics['f1']:.4f}, "
                       f"Precision: {metrics['precision']:.4f}, "
                       f"Recall: {metrics['recall']:.4f}")
        
        # 计算平均指标
        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        std_f1 = np.std(f1_scores)
        
        logger.info("-" * 80)
        logger.info(f"{'平均':>8} - F1: {avg_f1:.4f}±{std_f1:.4f}, "
                   f"Precision: {avg_precision:.4f}, "
                   f"Recall: {avg_recall:.4f}")
        logger.info("="*80)



def main():
    # 首先设置随机种子
    set_seed(DATA_CONFIG['random_seed'])
    
    logger.info("="*80)
    logger.info("协议边界检测训练开始")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"随机种子: {DATA_CONFIG['random_seed']}")
    logger.info("="*80)
    

    
    selected_gpus = TRAINING_CONFIG['gpu_ids'] 
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected_gpus))
    logger.info(f"设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"使用GPU: {selected_gpus}")
    logger.info(f"使用模型: ColFormerDynamicSGP")
    
    # 只显示 ColFormerDynamicSGP 的配置
    logger.info(f"模型配置参数: {COLFORMER_DYNAMIC_SGP_CONFIG}")
    
    # 重新初始化CUDA上下文
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 设置设备和多GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    logger.info(f"使用设备: {device}")
    logger.info(f"PyTorch可见GPU数量: {gpu_count}")
    
    # 直接运行留一法交叉验证
    logger.info("开始留一法交叉验证模式...")
    run_cross_validation()

if __name__ == "__main__":
    main()
