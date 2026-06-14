from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from config.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None

def get_bge_m3_ef():
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 从环境变量加载配置
    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16
    # 如果模型没有被提前下载，会自动下载
    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )

    return _bge_m3_ef