import os
import json
import asyncio
import threading
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from rlm.models import get_model
from rlm.engine import RLMEngine
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="Recursive Language Model (RLM) Visualizer API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/run")
async def run_rlm(
    query: str = Form(...),
    provider: str = Form("mock"),
    model_name: str = Form("gpt-4o-mini"),
    max_depth: int = Form(3),
    max_steps: int = Form(30),
    context_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    environment: str = Form("local")
):
    # Retrieve or extract context content
    context = ""
    if file:
        file_content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            try:
                import io
                pdf_file = io.BytesIO(file_content)
                reader = PdfReader(pdf_file)
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                context = "\n".join(pages_text)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {str(e)}")
        else:
            # Assume plain text
            try:
                context = file_content.decode("utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode text file: {str(e)}")
    elif context_text:
        context = context_text
    else:
        raise HTTPException(status_code=400, detail="Either file upload or context_text is required.")

    if not context.strip():
        raise HTTPException(status_code=400, detail="Context is empty.")

    # Initialize model
    try:
        model = get_model(provider, model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

    # Setup async queue for SSE events
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(event: dict):
        # Push callback event into the asyncio queue safely from engine executor thread
        loop.call_soon_threadsafe(queue.put_nowait, event)

    engine = RLMEngine(
        model=model,
        leaf_model=model,
        max_depth=max_depth,
        max_steps=max_steps,
        verbose=True,
        callback=callback,
        environment=environment
    )

    # Run the engine execution loop in a background thread
    def run_engine():
        try:
            result = engine.run(query, context)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "complete", "final_answer": result})
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, "DONE")

    threading.Thread(target=run_engine, daemon=True).start()

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event == "DONE":
                    break
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
