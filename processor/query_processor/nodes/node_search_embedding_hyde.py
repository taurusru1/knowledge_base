# processor/query_processor/nodes/node_search_embedding_hyde.py
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.promote.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_utils import json_dumps
from utils.llm_utils import get_llm_client
from utils.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE (Hypothetical Document Embedding) 检索节点
        核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

        执行步骤：
        1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
        2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
        3. 混合检索 (Step 2)：
           - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
           - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
        4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。

        :param state: 会话状态字典，包含 rewritten_query, item_names 等
        :return: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档) 的字典
        """

        # 步骤1：参数增强后的原始提示词和商品名称
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        try:
            # 进行校验判断
            if not rewritten_query:
                logger.error("参数 rewritten_query 为空")
            if not item_names:
                logger.error("参数 item_names 为空")
            # 步骤2：生成假设文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)
            # 步骤3：调用BGEM3大模型和，生成假设文档向量（传入增强后的查询语句，假设性文档，商品名称）
            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )
            # 4、结果封装
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc,
            }
        except:
            logger.exception(f"假设性文档向量搜索失败: {e}")
            return {}

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        """
        阶段2：利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档
        :param item_names: 商品名称列表，用于元数据过滤 (item_name in [...])
        :return: 检索结果列表
        """
        # 1.将rewritten_query和hyde_doc拼接起来
        combined_text = rewritten_query + " " + hyde_doc
        # 2。对这个拼接后的字符串combined_text进行向量化，获取到稠密和稀疏向量
        embeddings = generate_embeddings([combined_text])
        dense_ves = embeddings.get("dense")[0]
        sparse_ves = embeddings.get("sparse")[0]
        # 3. 获取Milvus的集合
        collection_name = milvus_config.chunks_collection
        # 4.构造过滤条件
        try:
            expr = None
            if item_names:
                # quoted = ", ".join(f'"{v}"' for v in item_names)
                # expr = f"item_name in [{quoted}]"
                # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
                # 构造混合查询的请求
                expr = f'item_name in {item_names}'
                logger.info(f"过滤条件: {expr}")
            else:
                logger.info("未指定商品名过滤，将全库检索")

            # 5、构造Milvus混合搜索请求对象
            reqs = create_hybrid_search_requests(
                dense_vector=dense_ves,
                sparse_vector=sparse_ves,
                expr=expr,  # 构造的过滤条件
                limit=10  # 底层检索返回数量（后续会再过滤为5，预留更多结果做重排序）
            )

            # 6、执行milvus混合向量检索
            logger.info("开始执行 Milvus 混合检索...")
            client = get_milvus_client()
            res = hybrid_search(
                client=client,
                collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
                reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
                ranker_weights=(0.8, 0.2),  # 稠/稀疏向量评分权重配比，各占50%（可按业务调优）
                output_fields=["chunk_id", "content", "item_name"]  # 输出的字段
            )
            # 返回相似度最高的
            return res[0] if res else []
        except Exception as e:
            logger.error(f"步骤2: 检索过程发生异常: {e}")
            raise  e

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        阶段1：利用大模型根据用户查询生成假设性文档（Hypothetical Document）。
        HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
        从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。

        :param rewritten_query: 用户改写后的查询语句
        :return: LLM生成的假设性文档内容
        """
        logger.info("步骤1: 开始生成假设性文档")

        try:
            llm = get_llm_client()
            # 构造提示词，把增强后的查询语句传入
            hyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(hyde_prompt).content
            return hyde_doc

        except Exception as e:
            logger.exception(f"步骤1: 生成假设文档失败: {e}")
            raise e

if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(json_dumps(result, indent=4))
