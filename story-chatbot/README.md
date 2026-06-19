# Story Chatbot — distilgpt2

## Lancement

### 1. Backend (Python)

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Le modèle distilgpt2 sera téléchargé automatiquement depuis HuggingFace au premier lancement (~350 MB).

### 2. App mobile (Expo Go)

```bash
cd mobile
npm install
npx expo start
```

Scanne le QR code avec **Expo Go** sur ton iPhone.

#### Important — IP locale

Pour tester sur iPhone physique, remplace `localhost` dans `App.tsx` par l'IP de ton PC :

```ts
// App.tsx ligne 10
const BACKEND = "http://192.168.x.x:8000"; // ton IP locale
```

Trouve ton IP avec : `ipconfig` (cherche "Adresse IPv4")
