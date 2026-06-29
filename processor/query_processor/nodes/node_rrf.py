# processor/query_processor/nodes/node_rrf.py
from typing import List, Dict, Tuple, Any

from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_utils import json_dumps


class NodeRrf(NodeBase):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web）进行加权融合排序。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_rrf"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 1. 获取到各路检索结果
        embedding_search_list =  []
        state_embedding_chunks = state.get("embedding_chunks",[])
        for doc in state_embedding_chunks:
            # 做数据健壮性判断，数据必须是字典类型的
            if isinstance(doc, dict):
                embedding_search_list.append(doc["entity"])

        hyde_embedding_search_list = []
        state_hyde_chunks = state.get("hyde_embedding_chunks",[])
        for doc in state_hyde_chunks:
            if isinstance(doc, dict):
                hyde_embedding_search_list.append(doc["entity"])

        # 2.为不同的检索结果设置不同的权重
        rrf_inputs = [
            (embedding_search_list, 1.0),
            (hyde_embedding_search_list, 1.0)
        ]

        # 3.利用RRF的计算公式去获取到所有路查询到的所有chunk对应的score
        rrf_merge_results = self._rrf_merge(rrf_inputs,max_results=10)

        # 4. 获取rrf_chunks（只取文档，不要分数）
        # 为什么只要文档，不去分数，因为这个数据是要给后面的重排序，重排序是按语义进行排序，不需要分数，所以把doc取出来，传递下去
        rrf_chunks = [doc for doc, _ in rrf_merge_results]

        # 5. 更新state
        state['rrf_chunks'] = rrf_chunks

        # 6. 返回state
        return state

    def _rrf_merge(self, rrf_inputs, k: int = 60, max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        利用 RRF 公式计算每一个文档的总得分
        :param rrf_inputs:  列表，每个元素是(各路的搜索结果列表, 权重)的元组
        :param k:           平滑参数(RFF常数)，通常取 60
        :param max_results: 合并完之后返回的文档数，None 表示全部
        :return:            合并以及排序后的文档列表，[(元素, RRF 得分), ...] 按得分降序
        """
        chunk_scores = {}  # 存放所有 chunk 的 RRF 计算后的分数值
        chunk_data = {}  # 存放所有 chunk 的文档数据

        for rrf_input, weight in rrf_inputs:
            # rank(文档排名)
            # doc 文档
            for rank,doc in enumerate(rrf_input,start=1):
                chunk_id = doc["chunk_id"]
                # RRF公式：score += weight * (1 / (rank + k))
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight * (1 / (rank + k))

                # 使用 setdefault 保留首次遇到的文档版本(只记录第一次)
                chunk_data.setdefault(
                    chunk_id,
                    doc
                )  # 无则加，有则不加

            # 2.将得分和文档组合在一起
            # for chunk_id, score in chunk_scores.items():
            #     # chunk_data 是一个字典
            #     # chunk_id 是键（比如 "chunk_1"）
            #     # ["score"]： 访问这个文档的 "score" 属性，如果不存在就创建，如果存在就覆盖
            #     result_unsorted = chunk_data[chunk_id]["score"] = score

        unsorted_results = []
        for chunk_id, score in chunk_scores.items():
            unsorted_results.append((chunk_data[chunk_id], score))

        # 3.按照得分进行降序
        # 降序：reverse=True
        sorted_result = sorted(unsorted_results, key=lambda x: x[1], reverse=True)

        # 返回max_results个文档数量
        # 如果没有max_results则返回所有数据
        return sorted_result[:max_results] if max_results else sorted_result



if __name__ == '__main__':

    # 模拟两路检索结果
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#3"}},
        ]
    }

    node_rrf = NodeRrf()
    result = node_rrf(mock_state)
    logger.info(json_dumps(result, indent=4))
