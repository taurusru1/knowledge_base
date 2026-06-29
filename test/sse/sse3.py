import asyncio

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pymilvus.grpc_gen.milvus_pb2 import QueryRequest
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

# 1.创建引用
app = FastAPI()

# 2.跨域
app.add_middleware(
    CORSMiddleware,  # 启用跨域中间件
    allow_origins=["*"],  # 允许所有来源（任何网页都能调用）
    allow_credentials=True,  # 允许携带 Cookie
    allow_methods=["*"],  # 允许所有请求方式（GET/POST等）
    allow_headers=["*"],  # 允许所有请求头
)
# 3.定义请求体
class QueryRequest(BaseModel):
    query: str
    session_id: str

# 4. 定义字典:
# key:会话id session_id
# value:异步队列
task_queues = {}

# 5.定义一个做包子的函数
async def long_task(session_id: str, query: str):
    # 为当前会话创建专属异步队列,用于存放做的包子
    queue = asyncio.Queue()
    task_queues[session_id] = queue # 将这个队列和会话id关联起来

    # 根据问题生成结果 TODO

    # 做包子：按查询词生成5条结果，每秒1条丢进队列
    for i in range(5):
        progress_msg = {
            "event": "progress",
            "data": f"【{query}】的第{i + 1}段回答:xxx{i + 1}"
        }
        await queue.put(progress_msg)  # 进度消息入队
        await asyncio.sleep(1)

    # 任务完成，往队列丢完成消息
    complete_msg = {
        "event": "complete",
        "data": f"【{session_id}】查询完成！所有结果已返回"
    }
    await queue.put(complete_msg)



# 6.定义接受前端请求的接口
@app.post("/submit_query")
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    # 把查询词和会话ID传给后台任务
    background_tasks.add_task(long_task, req.session_id, req.query) # 这个后台任务开始做包子
    return {"message": "任务已启动", "session_id": req.session_id}



# 7.从队列获取到数据，并且采用SSE流式输出向前端推流
@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    # 定义生成器函数
    async def  event_generator():
        # 等待当前会话的队列创建（防止SSE比任务先启动）
        while session_id not in task_queues:
            await asyncio.sleep(0.1)

        # 有相应的 session_id 对应的队列，开始从队列中获取数据
        queue = task_queues[session_id]

        # 开始获取消息
        # 循环从队列中获取消息，有消息就推，收到结束标记就停
        while True:
            msg = await queue.get()  # 异步阻塞等待消息
            if msg is None: # 收到结束标记，退出循环
                break

            # 拼接自定义Event的SSE格式
            yield f"event: {msg['event']}\n"
            yield f"data: {msg['data']}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8081)