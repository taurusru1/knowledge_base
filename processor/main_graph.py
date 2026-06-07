from langgraph.constants import END
from langgraph.graph import StateGraph

from processor.base import setup_logging
from processor.import_processor.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.node_document_split import NodeDocumentSplit
from processor.import_processor.node_import_milvus import NodeImportMilvus
from processor.import_processor.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.node_md_img import NodeMDImg
from processor.import_processor.node_pdf_to_md import NodePDFToMD
from processor.import_processor.node_entry import NodeEntry
from processor.state import ImportGraphState


class KBImportWorkflow:
    """知识库导入工作流"""

    def __init__(self):
        """
        初始化工作流对象
        """
        self._compiled_graph = None

    @property #将这个方法当成属性，进行调用
    def graph(self):
        """
        加载延迟(懒加载)：只在第一次使用工作流对象的时候编译图
        """
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph


    def build_graph(self):
        # 1.初始化工作流
        graph = StateGraph(ImportGraphState)
        # 2.注册节点
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_item_name_recognition", NodeItemNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # 3.设置入口节点
        graph.set_entry_point("node_entry")
        # 4.注册条件边
        graph.add_conditional_edges(
            "node_entry",
            self.route_after_entry,
            {
                "node_pdf_to_md":"node_pdf_to_md",
                "node_md_img":"node_md_img",
                END:END
            }
        )
        # 5.注册顺序表
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_item_name_recognition")
        graph.add_edge("node_item_name_recognition", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)
        # 6.编译图
        return graph.compile()

    @staticmethod
    def route_after_entry(state: ImportGraphState) -> str:
        """
            入口节点后的条件路由函数
            :param state: 当前状态
            :return: 下一个节点名称
        """
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END

    def run(self,state: ImportGraphState,stream : bool=False):
        """
        统一执行入口，支持切换invoke/stream
        :param state: 初始状态
        :param stream: 是否流式输出
        :return: 执行结果
        """

        # 打印图结构（仅调试用）
        self.graph.get_graph().print_ascii()

        if stream:
            return self.graph.stream(state, stream_mode="values")
        else:
            return self.graph.invoke(state)

if __name__ == "__main__":
    #
    setup_logging()
    # 模拟一个初始状态进行测试
    init_state = {
        "import_file_path": "一个文件.pdf",
    }
    workflwo = KBImportWorkflow()

    # 传入初始状态
    final_state = workflwo.run(state=init_state)
    print(final_state)
