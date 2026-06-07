# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
from typing import Tuple, List, Dict

from torch.ao.quantization.fx.utils import return_arg_list

from processor.base import BaseNode, setup_logging
from processor.exceptions import StateFieldError
from processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    节点：文档切分（node_document_split）
    整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
    核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
    后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

    必要参数：md_content、file_title
    更新参数：chunks

    :param state: 工作流状态对象
    :return: 更新后的状态对象
    """

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 步骤1：加载并标准化输入数据，输入值state，返回值：cotent，file_title
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按MD标题进行初次切分
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # 步骤3：无标题场景兜底处理
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：Chunk精细化处理（长切短合
        sections = self._step_4_refine_chunks(sections)

        # 步骤5：输出文档切分统计信息
        self._step_5_print_stats(lines_count, sections)

        # 步骤6：Chunk结果本地JSON备份
        self._step_6_backup(state, sections)

        # 写入状态字典
        state["chunks"] = sections
        return state

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """
        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'^\s*#{1,6}\s+.+'

        # 2、初始化需要的数据
        lines = content.split("\n") # 以空格拆分，得到每一个行列表
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中

        # 3.定义组装sections章节列表的函数
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            sections.append({
                "title":current_title,
                "content" : "\n".join(current_lines), #
                "file_title": file_title
            })

        # 4. 逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            # 去空格处理
            striped_line = line.strip()
            # 4.1判断是不是代码块，识别代码块是看```、~~~、````、~~~~ 等（至少 3 个连续字符）
            if striped_line.startswith("```") or striped_line.startswith("~~~"):
                # 如果是代码块就把in_code_block改为true，并且把这个line直接放入current_lines中
                in_code_block = not in_code_block
                current_lines.append(line)
                continue
            # 4.2 判断是不是普通标题，判断标题是看title_pattern是不是匹配标题的正则表达式
            is_valid_title = in_code_block and re.match(title_pattern, striped_line)
            if is_valid_title:
                # 如果是标题 就要把上面的文本进行调用_flush_section进行组装sections，并且将striped_line到current_title中，并且放到current_lines
                _flush_section()
                title_count+=1
                current_title = line
                current_lines = [current_title]
            else:
                # 4.3此时是普通行，直接放到current_lines中
                current_lines.append(line)

        _flush_section()

        return sections, title_count, len(lines)

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题
        """
        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        # 不同系统中的换行是不一样的->win版 换行\r\n，老式mac版本是\r，所以我们要统一为\n，便于切分
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title



if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "hak180产品安全手册"
    }
    # 执行文档切分节点
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))