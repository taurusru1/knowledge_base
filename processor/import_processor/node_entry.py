# processor/import_processor/nodes/node_entry.py

import logging
from pathlib import Path

from processor.base import setup_logging, BaseNode
from processor.exceptions import ImportProcessError, StateFieldError, FileProcessingError, ValidationError
from processor.state import ImportGraphState


class NodeEntry(BaseNode):
    """
    入口节点：任务分发
    """

    name = "node_entry"

    def process(self, state: ImportGraphState):

        file_path = state.get("import_file_path")
        #1. 判断路径是否为空
        if not file_path:
            raise StateFieldError(
                field_name="import_file_path",
                message="路径不能为空",
                expected_type=str
            )
        # 讲文件转化为Path文件对象
        import_file_path_obj = Path(file_path)
        # 2.判断文件是否存在
        if not import_file_path_obj:
            raise FileProcessingError(
                message=f"文件{import_file_path_obj.name}不存在"
            )
        # 3.判断文件的后缀
        if  import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["is_md_read_enabled"] = False
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["is_pdf_read_enabled"] = False
        else:
            raise ValidationError(
                message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持"
            )
        # 4.提取文件名作为标题存入file_title# 5. 返回state
        #         return state
        state["file_title"] = import_file_path_obj.name
        return state

# 单元测试
if __name__ == "__main__":

    # 激活log日志
    setup_logging()

    init_state = {
        "import_file_path" : r"D:\课程\尚硅谷\掌柜智库\2.资料\04-设备手册汇总\doc\hak180产品安全手册.pdf"

    }
    node_entry = NodeEntry()
    result = node_entry(init_state)
    print(result)
    logging.getLogger().info(result)
    # logging.getLogger().info(111111)