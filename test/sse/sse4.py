import asyncio

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from test.sse.sse3 import QueryRequest, task_queues

# 1.构建 应用
app = FastAPI()

# 2。设置跨域
app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3.构建pydantic模型，用于接收前端的参数
class QueryRequest(BaseModel):
    query: str
    session_id: str

# 创建一个用于接收后台任务的队列列表
task_queues = {}

# 4.构建后台任务
async def long_task(query: QueryRequest, session_id: str):
    # 为当前会话创建专属异步队列
    queue =  asyncio.Queue()
    task_queues[session_id] = queue

    # 真正执行的业务

    # 按查询词生成5条结果，每秒1条丢进队列
    for i in range(5):
        msg = f"【{query}】的第{i + 1}段回答：xxx{i + 1}"
        await queue.put(msg)
        await asyncio.sleep(1)

    # 关键：放入结束标记，告诉SSE停止推送
    await queue.put("[END]")


# 4.设置后端接口
@app.post("/submit_query")
async def submit_query(query_request: QueryRequest,background_tasks: BackgroundTasks):
    """
    提交查询请求
    :param query_request: 查询请求
    :return: 查询结果
    """
    # 构建后台任务
    background_tasks.add_task(long_task, query_request, query_request.session_id)
    return {"message": "任务已启动", "session_id": query_request.session_id}


# 创建接收流式推送的接口
@app.get("/stream/{session_id}")
async def stream_response(session_id: str):
    """
    接收推送的接口
    :param session_id: 会话ID
    :return: 推送结果
    """
    async def event_generator():
        while session_id not in task_queues:
            await asyncio.sleep(0.1)

        # 有这个session_id所对应的数据
        queue = task_queues[session_id]
        # 不断的从队列中获取数据
        while True:
            msg = await queue.get()
            if msg == "[END]":
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream", # 指定媒体类型为SSE
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

