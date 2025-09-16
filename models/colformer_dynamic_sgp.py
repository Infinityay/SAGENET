import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """U-Net的双卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样路径"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样路径"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸不匹配
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetProcessor(nn.Module):
    """U-Net结构处理器 - 专门用于结构特征处理"""
    
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        
        # 输出层：生成边界显著性图
        self.outc = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, n_classes, kernel_size=1),
            nn.Sigmoid()  # 输出 [0, 1] 范围的显著性分数
        )

    def forward(self, x):
        """
        输入: [B, C_struct, L] - 多维结构特征
        输出: [B, 1, L] - 边界显著性图
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 生成边界显著性图
        saliency_map = self.outc(x)
        
        return saliency_map

class ColFormerDynamicSGP(nn.Module):
    
    def __init__(self, 
                 input_dim=8, 
                 feature_dim=512, 
                 nhead=8, 
                 intra_message_layers=2,
                 num_labels=2, 
                 dropout=0.1,
                 enable_structure_guidance=True,  # 是否启用结构引导（调制）
                 use_structure_features=True,     # 是否使用结构特征
                 enable_ipc=True,                 # 是否启用包间协作（IPC）
                 structure_features=['abc', 'entropy'],
                 ):
        super().__init__()
        
        self.enable_structure_guidance = enable_structure_guidance
        self.use_structure_features = use_structure_features
        self.enable_ipc = enable_ipc
        self.structure_features = structure_features
        
        # 计算结构特征的通道数
        num_structure_channels = len(structure_features) if structure_features else 1
        
        cnn_input_dim = input_dim
  
        # 多尺度CNN特征提取器
        self.cnn_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(cnn_input_dim, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(cnn_input_dim, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Conv1d(cnn_input_dim, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        ])

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(192, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # === 结构特征处理模块 ===
        if self.use_structure_features:
            if self.enable_structure_guidance:
                # 使用U-Net + 调制方式
                self.structure_processor = UNetProcessor(
                    n_channels=num_structure_channels,  # 动态设置通道数
                    n_classes=1,   # 输出边界显著性图
                    bilinear=False  # 使用ConvTranspose1d避免deterministic问题
                )
                
                # 升级的通道注意力门控网络：显著性图 -> 精细化引导权重
                self.gate_network = nn.Sequential(
                    # 输入是U-Net处理后的1维显著性图 [B, 1, L]
                    nn.Conv1d(1, feature_dim, kernel_size=1),  # 扩展到与cnn_features相同的维度 [B, feature_dim, L]
                    nn.Sigmoid()  # 输出 [0, 1] 范围的通道级别权重
                )
            else:
                # 不使用调制，直接拼接结构特征
                self.structure_processor = None
                self.gate_network = None
                # 修改CNN输入维度以包含结构特征
                cnn_input_dim = input_dim + num_structure_channels
                
                # 重新定义CNN提取器以适应新的输入维度
                self.cnn_extractor = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(cnn_input_dim, 64, kernel_size=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ),
                    nn.Sequential(
                        nn.Conv1d(cnn_input_dim, 64, kernel_size=3, padding=1),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ),
                    nn.Sequential(
                        nn.Conv1d(cnn_input_dim, 64, kernel_size=5, padding=2),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                ])
        else:
            self.structure_processor = None
            self.gate_network = None
        
        # 动态位置编码
        self.pos_embedding = nn.Embedding(64, feature_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.02)
        self.pos_weight = nn.Parameter(torch.tensor(0.5))
        
        # 包内注意力
        intra_encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=nhead, 
            dim_feedforward=feature_dim * 2,  
            dropout=dropout, 
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        
        self.intra_message_transformer = TransformerEncoder(
            intra_encoder_layer, 
            num_layers=intra_message_layers
        )
        
        
        # 包间协作机制（IPC）
        if self.enable_ipc:
            self.cross_batch_attention = nn.MultiheadAttention(
                feature_dim, nhead, dropout=dropout, batch_first=True
            )
            self.cross_batch_norm = nn.LayerNorm(feature_dim)
            self.cross_batch_ffn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Dropout(dropout)
            )
        
            # 自适应权重机制
            self.adaptive_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
        else:
            self.cross_batch_attention = None
            self.cross_batch_norm = None
            self.cross_batch_ffn = None
            self.adaptive_gate = None
        
        # 更强的分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, num_labels)
        )

    def _prepare_cnn_input(self, bit_matrix):
        """准备CNN输入 - 简化为固定模式"""
        return bit_matrix.permute(0, 2, 1)
    
    def _process_cnn_output(self, cnn_features, original_length):
        """处理CNN输出 - 简化为固定模式"""
        return cnn_features.permute(0, 2, 1)  # 固定为 (0, 2, 1)
    
    
    def forward(self, bit_matrix, structure_features=None, attention_mask=None, return_features=False):
        batch_size, max_length, _ = bit_matrix.shape
        
        # === 结构特征处理 ===
        guidance_weights = None
        
        if self.use_structure_features and structure_features is not None:
            if self.enable_structure_guidance:
                # 使用调制方式（U-Net + Gate）
                # 直接使用预计算的结构特征 [B, L, C] -> [B, C, L]
                structure_input = structure_features.permute(0, 2, 1)
                
                # U-Net处理：生成边界显著性图 [B, 1, L]
                saliency_map = self.structure_processor(structure_input)
                
                # 生成精细化引导权重 [B, feature_dim, L] -> [B, L, feature_dim]
                guidance_weights = self.gate_network(saliency_map).permute(0, 2, 1)
            else:
                # 不使用调制，直接拼接结构特征到输入
                # bit_matrix: [B, L, 8], structure_features: [B, L, C]
                bit_matrix = torch.cat([bit_matrix, structure_features], dim=-1)
        
        # 准备CNN输入
        x_cnn = self._prepare_cnn_input(bit_matrix)
        
        # 多尺度CNN特征提取
        multi_scale_features = []
        for cnn_branch in self.cnn_extractor:
            features = cnn_branch(x_cnn)
            multi_scale_features.append(features)
        
        # 融合多尺度特征
        combined_features = torch.cat(multi_scale_features, dim=1)
        cnn_features = self.feature_fusion(combined_features)
        
        # 处理CNN输出
        cnn_features = self._process_cnn_output(cnn_features, max_length)
        
        # 应用结构引导（调制）
        if self.use_structure_features and self.enable_structure_guidance and guidance_weights is not None:
            # guidance_weights: [B, L, feature_dim] - 每个位置每个通道都有独立权重
            fused_features = cnn_features * guidance_weights 
            cnn_features = fused_features
        
        # 位置编码
        pos_ids = torch.arange(max_length, device=bit_matrix.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(pos_ids) * self.pos_weight
        cnn_features = cnn_features + pos_emb
        
        # 注意力掩码
        if attention_mask is None:
            attention_mask = (bit_matrix.sum(dim=-1) > 0)
        padding_mask = ~attention_mask
            
        intra_features = self.intra_message_transformer(
            cnn_features, 
            src_key_padding_mask=padding_mask
        )
        
        if self.enable_ipc:
            # 包间协作机制
            valid_mask = attention_mask.unsqueeze(-1).float()
            masked_features = intra_features * valid_mask
            
            # ========== 位置感知的批次原型==========
            # 对每个位置计算所有样本的平均
            position_mask = valid_mask.sum(dim=0, keepdim=True) + 1e-8  # [1, L, 1]
            batch_prototype = masked_features.sum(dim=0, keepdim=True) / position_mask  # [1, L, D]
            batch_prototype = batch_prototype.expand(batch_size, -1, -1)  # [B, L, D]
            # =====================================================
    
            cross_features, cross_attn_weights = self.cross_batch_attention(
                intra_features, batch_prototype, batch_prototype,
                key_padding_mask=padding_mask
            )
            cross_features = self.cross_batch_norm(cross_features + intra_features)
            
            ffn_output = self.cross_batch_ffn(cross_features)
            inter_features = cross_features + ffn_output
            
            combined = torch.cat([intra_features, inter_features], dim=-1)
            gate = self.adaptive_gate(combined)
            final_features = gate * inter_features + (1 - gate) * intra_features
        else:
            # 不使用包间协作，直接使用包内特征
            final_features = intra_features
    
        
        # 分类
        logits = self.classifier(final_features)
        
        return logits 