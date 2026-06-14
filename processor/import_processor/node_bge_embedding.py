# processor/import_processor/nodes/node_bge_embedding.py
import json
import logging
from typing import List, Dict

from sympy import content

from knowledge_base_teach.utils.embedding_utils import generate_embeddings
from processor.base import BaseNode, setup_logging
from processor.exceptions import StateFieldError
from processor.state import ImportGraphState


class NodeBGEEmbedding(BaseNode):
    """
    混合向量化节点：使用 BGE-M3 模型将文本转换为向量
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        流程总览：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象

        业务背景：
            前序节点已完成文档解析、图片理解、智能切片和元数据初始化，产出了结构化的 chunk JSON 数据。
            由于 Milvus 主要基于向量字段进行相似度检索，不能直接依赖纯文本 JSON 完成语义召回，
            因此本节点需要对每个 chunk 的文本内容及相关检索字段进行混合向量化，生成稠密向量和稀疏向量，
            并将其与原始 chunk 元数据一起持久化到 Milvus 中，为后续知识库检索和 RAG 问答提供索引基础。

            注意：原始文本和元数据可以作为标量字段或 JSON 字段存入 Milvus，但检索召回依赖的语义表示需要先转换为向量字段。
        """
        # 步骤1：输入数据校验
        chunks = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        return state

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:

        """
        步骤 2: 批量生成向量（核心业务逻辑）
        核心逻辑：
            1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
            2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
            3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
        参数：
            chunks: List[Dict] 待向量化的文本切片列表
        返回：
            List[Dict]: 包含向量字段（dense_vector/sparse_vector）的文本切片列表
        """
        # 定义批处理数量
        batch_size = 3
        # 初始化一个空列表
        output_data = []
        # 按批次遍历文本切片：range(起始, 终止, 步长) → 0,3,6... 分批处理
        for i in range(0, len(chunks), batch_size):
            batch_texts = chunks[i:i + batch_size] # 获取当前批次的文本切片
            input_texts = []
            # 对分组的数据进行遍历，得到item_name,content,并且构造输入文本（用于）
            for doc in batch_texts:
                item_name = doc["item_name"]
                content = doc["content"]
                # 把item_name放到content前面，让模型更明确这段内容属于哪个主题
                input_texts.append(f"{item_name}\n{content}" if item_name else content)

            docs_embeddings = generate_embeddings(input_texts)
            for j,doc in enumerate(batch_texts):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            self.logger.info(f"成功获取第 {i + 1}-{min(i + len(batch_texts), len(chunks))} 项的嵌入。")

        return output_data


    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        步骤 1：输入数据有效性校验
        核心作用：
            1. 从全局状态提取待向量化的chunks切片列表
            2. 严格校验chunks类型和非空性，无有效数据则终止向量化
        参数：
            state: ImportGraphState - 流程全局状态对象
        返回：
            List[Dict] - 校验通过的文本切片列表
        异常：
            若chunks非列表/为空，抛出ValueError，终止当前向量化流程
        """
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks


if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))