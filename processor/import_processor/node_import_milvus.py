# processor/import_processor/nodes/node_import_milvus.py

from processor.base import BaseNode
from processor.state import ImportGraphState


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):


        return state