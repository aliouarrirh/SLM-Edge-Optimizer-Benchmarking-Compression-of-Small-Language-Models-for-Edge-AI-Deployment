import { useState, useRef, useEffect } from "react";

const BACKEND = "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  text: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text: "Bonjour ! Donne-moi un début d'histoire et je la continuerai pour toi. ✨",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function send() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    const updated: Message[] = [...messages, { role: "user", text: prompt }];
    setMessages(updated);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 200 }),
      });
      const data = await res.json();
      setMessages([...updated, { role: "assistant", text: data.story }]);
    } catch {
      setMessages([
        ...updated,
        { role: "assistant", text: "Erreur : impossible de joindre le backend." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="flex flex-col h-dvh bg-[#0f0f0f] text-white">
      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-4 border-b border-white/10 bg-[#1a1a1a]">
        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">
          ✦
        </div>
        <div>
          <p className="font-semibold text-sm leading-tight">Story Generator</p>
          <p className="text-xs text-white/40">distilgpt2 · local</p>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-purple-600 text-white rounded-br-sm"
                  : "bg-[#1e1e1e] text-white/90 border border-white/10 rounded-bl-sm"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-[#1e1e1e] border border-white/10 rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1 items-center h-4">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-bounce [animation-delay:300ms]" />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 pb-6 pt-2 bg-[#0f0f0f] border-t border-white/10">
        <div className="flex gap-2 items-end bg-[#1a1a1a] border border-white/10 rounded-2xl px-4 py-2">
          <textarea
            className="flex-1 bg-transparent resize-none outline-none text-sm text-white placeholder-white/30 max-h-32 py-1"
            rows={1}
            placeholder="Commence une histoire..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={loading}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="mb-1 w-8 h-8 rounded-full bg-purple-600 hover:bg-purple-500 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors flex-shrink-0"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
        <p className="text-center text-xs text-white/20 mt-2">
          Entrée pour envoyer · Shift+Entrée pour sauter une ligne
        </p>
      </div>
    </div>
  );
}
