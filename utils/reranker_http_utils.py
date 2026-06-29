# utils/reranker_http_utils.py

import dashscope
from dotenv import load_dotenv
from config.reranker_config import reranker_config

load_dotenv()

def rerank_documents(query: str, documents: list[str]) -> list[float]:
    # 创建rerank模型客户端
    dashscope.api_key = reranker_config.text_rerank_api_key
    response = dashscope.TextReRank.call(
        model=reranker_config.text_rerank_model,
        query=query,
        documents=documents,
        top_n=len(documents),
        return_documents=False,
        instruct=reranker_config.text_rerank_instruct, # 指定模型指令
    )

    status_code = response.get("status_code")
    if status_code != 200:
        message = response.get("message")# 获取错误信息
        raise RuntimeError(f"DashScope rerank 调用失败: {message}")
    # 正常执行出结果后则获取结果
    results = response.output.get("results", [])
    scores = [0.0] * len(documents) # 创建一个长度为文档数量的列表，初始值为0.0
    for item in results:
        index = item.get("index")
        score = item.get("relevance_score")
        scores[int(index)] = float(score)
    return scores # 返回一个分数的列表