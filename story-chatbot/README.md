# Story Chatbot — distilgpt2

## Lancement

### 1. Backend (Python)

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Le modèle distilgpt2 sera téléchargé automatiquement depuis HuggingFace au premier lancement (~350 MB).

### 2. Frontend (React)

```bash
cd frontend
npm install
npm run dev
```

Ouvre `http://localhost:5173` dans ton navigateur.

### Test sur iPhone

Une fois le frontend lancé, le terminal affiche une URL du type :
```
➜  Network: http://192.168.x.x:5173/
```
Ouvre cette URL dans Safari sur ton iPhone (même WiFi requis).
