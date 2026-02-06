# APIserver在框架中的作用
## 调用链路
vLLM的online serving采用分层架构，从HTTP接口到核心推理引擎的完整链路如下：

客户端请求 → HTTP服务器 → API路由 → 服务层 → 引擎客户端 → 核心引擎 → 推理执行

具体文件链路：
examples/online_serving/openai_chat_completion_client.py (客户端)
    ↓
vllm/entrypoints/openai/api_server.py (HTTP服务器入口)          压测其服务的文件地址
    ↓  
vllm/entrypoints/openai/chat_completion/api_router.py (API路由)
    ↓
vllm/entrypoints/openai/chat_completion/serving.py (服务层)
    ↓
vllm/v1/engine/async_llm.py (异步引擎客户端)
    ↓
vllm/v1/engine/core_client.py (引擎核心客户端)
    ↓
vllm/v1/engine/core.py (核心引擎)
    ↓
vllm/v1/executor/ (推理执行器)


客户端 (examples/online_serving/openai_chat_completion_client.py)
    ↓ HTTP请求到 http://localhost:8000/v1/chat/completions
服务器 (vllm/entrypoints/openai/api_server.py)
    ↓ 路由到 vllm/entrypoints/openai/chat_completion/api_router.py
    ↓ 调用 vllm/entrypoints/openai/chat_completion/serving.py
    ↓ 委托给 vllm/v1/engine/async_llm.py

## 服务模式和程序模式的区别：
场景 1：前端网页要用模型👉 那只能走 HTTP  
场景 2：很多人同时用模型👉 模型必须只加载一次  
场景 3：模型要一直开着（7×24）👉 不适合做服务  
API Server 就是为了解决上面这些问题  

## 什么是 FastAPI？
如果没有 FastAPI，你要手写很多麻烦的东西：
解析 HTTP
解析 JSON
校验参数
返回结果
FastAPI 帮你全做了。

```python 
from fastapi import FastAPI

app = FastAPI()

@app.post("/hello")
def hello(req: dict):
    return {"reply": "Hello " + req["name"]}```

uvicorn         # 负责监听端口
  └─ FastAPI app
      └─ /v1/completions
          └─ 调用 vLLM Engine`



