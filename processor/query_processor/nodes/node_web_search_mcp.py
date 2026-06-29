# processor/query_processor/nodes/node_web_search_mcp.py
import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_utils import json_dumps


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 1.获取到用户增强后的请求
        query = state.get("rewritten_query","")

        web_docs = []
        if query :
            # 2.调用MCP,得到一个字符串的结果
            result = asyncio.run(self._mcp_call(query))

            if result:
                json_str = result.content[0].text
            # 3.将结果转为字典
            pages = json.loads(json_str).get("pages")

            for item in pages:
                snippet = item.get("snippet")
                url = item.get("url")
                title = item.get("title")

                web_docs.append({"title": title, "url": url, "snippet": snippet})
            logger.info(f"MCP 搜索结果:%s",web_docs)
        if web_docs:
            return {"web_search_docs": web_docs}
        return {}

        # 2。调用外部搜索引擎，获取结果

    async def _mcp_call(self, query):
        # 构建没mcp客户端
        search_mcp= MCPServerStreamableHttp(
            name="search_mcp", # 请求的节点的名称
            params={
                "url": mcp_config.mcp_base_url, # 请求的url
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"}, # 请求头
                "timeout": 10, # 请求超时时间
            },
            cache_tools_list=True, # 是否缓存结果
            max_retry_attempts=3,# 最大重试次数
        )
        try:
            # 连接mcp
            await search_mcp.connect()
            # 调用mcp的工具
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            # 执行完毕后要及时清理残留
            await search_mcp.cleanup()


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(json_dumps(result, indent=4))
