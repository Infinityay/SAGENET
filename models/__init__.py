"""
模型模块
包含协议边界检测模型的实现
"""

from .colformer_dynamic_sgp import ColFormerDynamicSGP


def create_model(model_selection=None):
    """创建 ColFormerDynamicSGP 模型"""
    from config import MODEL_SELECTION
    
    # 直接使用 dynamic_sgp 配置
    config = MODEL_SELECTION.get('dynamic_sgp_config', {})
    return ColFormerDynamicSGP(**config)


def get_model_name():
    """获取模型名称"""
    return 'ColFormerDynamicSGP'