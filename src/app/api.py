from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from src.predict import analyze

app = FastAPI(title="Fake News Detection + Summarizer API")

class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Raw article text")
    url: Optional[str] = Field(default=None, description="Article URL")

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    return analyze(text=req.text, url=req.url, do_summary=True)