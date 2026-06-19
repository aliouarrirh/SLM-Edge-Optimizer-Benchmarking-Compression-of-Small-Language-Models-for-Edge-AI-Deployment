from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 0 if torch.cuda.is_available() else -1
print(f"Loading distilgpt2 on {'GPU' if device == 0 else 'CPU'}...")
generator = pipeline("text-generation", model="distilgpt2", device=device)
print("Model ready.")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200


@app.post("/api/generate")
def generate(req: GenerateRequest):
    results = generator(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    story = results[0]["generated_text"]
    return {"story": story}


@app.get("/health")
def health():
    return {"status": "ok"}
