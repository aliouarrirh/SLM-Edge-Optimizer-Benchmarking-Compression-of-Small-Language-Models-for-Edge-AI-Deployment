from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemin vers le modèle fine-tuné local (après modal volume get)
# Si le dossier n'existe pas, repli sur distilgpt2 de base depuis HuggingFace
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_compressed")
model_source = MODEL_PATH if os.path.isdir(MODEL_PATH) else "distilgpt2"

print(f"Chargement du modèle depuis : {model_source}")
tokenizer = AutoTokenizer.from_pretrained(model_source)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_source)

# Quantification dynamique int8 (CPU, ~2s, réduit la taille de ~50%)
print("Application de la quantification dynamique int8...")
model = torch.quantization.quantize_dynamic(
    base_model, {torch.nn.Linear}, dtype=torch.qint8
)
model.eval()
print("Modèle prêt (quantifié int8, CPU).")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU — le modèle quantifié int8 ne tourne que sur CPU
)


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
