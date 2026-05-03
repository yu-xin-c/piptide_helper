"""
Peptide Helper FastAPI 后端服务。

启动方式：
    python -m peptide_helper.server

或：
    uvicorn peptide_helper.server:app --reload --port 8000
"""

import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .graph import app as langgraph_app
from .state import create_initial_state

app = FastAPI(title="Peptide Helper", version="1.0.0")


# ── 请求模型 ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    user_input: str = Field(
        default="ACDEFGHIKLMNPQRSTVWY 帮我评估这条序列的抗菌活性和毒性风险",
        description="用户输入（可包含序列和分析需求，模型自动区分）",
    )


class AnalyzeStep(BaseModel):
    """流式输出的单步状态"""

    step: str
    data: dict = Field(default_factory=dict)


# ── API 路由 ───────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """执行多肽分析，返回完整结果。"""
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="请输入多肽序列和分析需求")

    state = create_initial_state(user_request=req.user_input.strip())

    try:
        result = await asyncio.to_thread(langgraph_app.invoke, state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

    return {
        "final_report": result.get("final_report", ""),
        "sequences": result.get("sequences", []),
        "multi_results": [
            r.model_dump() for r in result.get("multi_results", [])
        ],
    }


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """执行多肽分析，SSE 流式返回中间状态。"""
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="请输入多肽序列和分析需求")

    state = create_initial_state(user_request=req.user_input.strip())

    async def event_stream():
        try:
            yield "event: start\ndata: {}\n\n"

            for event in langgraph_app.stream(state):
                step_name = list(event.keys())[0]
                step_data = event[step_name]

                # 将 Pydantic 模型序列化为 dict
                serializable = {}
                for k, v in step_data.items():
                    if hasattr(v, "model_dump"):
                        serializable[k] = v.model_dump()
                    else:
                        serializable[k] = v

                import json

                yield f"event: step\ndata: {json.dumps({'step': step_name, 'data': serializable}, ensure_ascii=False)}\n\n"

            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            import json

            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── 静态文件（前端） ────────────────────────────────────────

import os

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ── 直接运行入口 ────────────────────────────────────────────

def main():
    import uvicorn

    uvicorn.run("peptide_helper.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()