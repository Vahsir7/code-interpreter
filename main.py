import sys
import os
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google import genai
from google.genai import types

app = FastAPI()

# Maximum CORS leniency
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Handle OPTIONS preflight manually too
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Add headers to every single response
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "3600"
    response.headers["ngrok-skip-browser-warning"] = "true"
    response.headers["bypass-tunnel-reminder"] = "true"
    return response

# Request/Response models
class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]

# Tool: execute Python code
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

# AI: analyze error and return line numbers
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{tb}

Return the line number(s) where the error is located."""

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    result = ErrorAnalysis.model_validate_json(response.text)
    return result.error_lines

# Endpoint
@app.post("/code-interpreter")
async def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return JSONResponse(
            content={"error": [], "result": execution["output"]},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    else:
        error_lines = analyze_error_with_ai(request.code, execution["output"])
        return JSONResponse(
            content={"error": error_lines, "result": execution["output"]},
            headers={"Access-Control-Allow-Origin": "*"}
        )