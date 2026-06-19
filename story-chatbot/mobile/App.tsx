import { useState, useRef, useEffect } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  SafeAreaView,
} from "react-native";
import { StatusBar } from "expo-status-bar";

// Remplace par l'IP de ton PC si tu testes sur iPhone physique
// ex: const BACKEND = "http://192.168.1.42:8000";
const BACKEND = "http://localhost:8000";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "0",
      role: "assistant",
      text: "Bonjour ! Donne-moi un début d'histoire et je la continuerai pour toi ✨",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<FlatList>(null);

  useEffect(() => {
    if (messages.length > 0) {
      setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }, [messages, loading]);

  async function send() {
    const prompt = input.trim();
    if (!prompt || loading) return;

    const userMsg: Message = { id: Date.now().toString(), role: "user", text: prompt };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 200 }),
      });
      const data = await res.json();
      setMessages([
        ...next,
        { id: (Date.now() + 1).toString(), role: "assistant", text: data.story },
      ]);
    } catch {
      setMessages([
        ...next,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: "Erreur : impossible de joindre le backend. Vérifie que le serveur Python tourne.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function renderItem({ item }: { item: Message }) {
    const isUser = item.role === "user";
    return (
      <View style={[s.bubbleRow, isUser ? s.rowRight : s.rowLeft]}>
        <View style={[s.bubble, isUser ? s.bubbleUser : s.bubbleBot]}>
          <Text style={[s.bubbleText, isUser ? s.textUser : s.textBot]}>
            {item.text}
          </Text>
        </View>
      </View>
    );
  }

  return (
    <SafeAreaView style={s.root}>
      <StatusBar style="light" />

      {/* Header */}
      <View style={s.header}>
        <View style={s.avatar}>
          <Text style={s.avatarText}>✦</Text>
        </View>
        <View>
          <Text style={s.headerTitle}>Story Generator</Text>
          <Text style={s.headerSub}>distilgpt2 · local</Text>
        </View>
      </View>

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        keyboardVerticalOffset={0}
      >
        {/* Messages */}
        <FlatList
          ref={listRef}
          data={messages}
          keyExtractor={(m) => m.id}
          renderItem={renderItem}
          contentContainerStyle={s.list}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
        />

        {loading && (
          <View style={s.loadingRow}>
            <View style={s.loadingBubble}>
              <ActivityIndicator color="#a855f7" size="small" />
            </View>
          </View>
        )}

        {/* Input */}
        <View style={s.inputBar}>
          <TextInput
            style={s.input}
            placeholder="Commence une histoire..."
            placeholderTextColor="#555"
            value={input}
            onChangeText={setInput}
            multiline
            editable={!loading}
            onSubmitEditing={send}
          />
          <TouchableOpacity
            style={[s.sendBtn, (!input.trim() || loading) && s.sendBtnDisabled]}
            onPress={send}
            disabled={!input.trim() || loading}
          >
            <Text style={s.sendIcon}>➤</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0f0f0f" },

  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: "#1a1a1a",
    borderBottomWidth: 1,
    borderBottomColor: "#2a2a2a",
  },
  avatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "#7c3aed",
    alignItems: "center",
    justifyContent: "center",
  },
  avatarText: { color: "#fff", fontSize: 16 },
  headerTitle: { color: "#f0f0f0", fontWeight: "600", fontSize: 15 },
  headerSub: { color: "#555", fontSize: 12 },

  list: { padding: 16, gap: 10 },

  bubbleRow: { flexDirection: "row", marginVertical: 4 },
  rowRight: { justifyContent: "flex-end" },
  rowLeft: { justifyContent: "flex-start" },

  bubble: {
    maxWidth: "80%",
    borderRadius: 18,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  bubbleUser: { backgroundColor: "#7c3aed", borderBottomRightRadius: 4 },
  bubbleBot: {
    backgroundColor: "#1e1e1e",
    borderWidth: 1,
    borderColor: "#2a2a2a",
    borderBottomLeftRadius: 4,
  },
  bubbleText: { fontSize: 15, lineHeight: 22 },
  textUser: { color: "#fff" },
  textBot: { color: "#e0e0e0" },

  loadingRow: { flexDirection: "row", paddingHorizontal: 16, paddingBottom: 4 },
  loadingBubble: {
    backgroundColor: "#1e1e1e",
    borderWidth: 1,
    borderColor: "#2a2a2a",
    borderRadius: 18,
    borderBottomLeftRadius: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
  },

  inputBar: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    padding: 12,
    backgroundColor: "#0f0f0f",
    borderTopWidth: 1,
    borderTopColor: "#2a2a2a",
  },
  input: {
    flex: 1,
    backgroundColor: "#1a1a1a",
    borderWidth: 1,
    borderColor: "#2a2a2a",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    color: "#f0f0f0",
    fontSize: 15,
    maxHeight: 100,
  },
  sendBtn: {
    width: 42,
    height: 42,
    borderRadius: 21,
    backgroundColor: "#7c3aed",
    alignItems: "center",
    justifyContent: "center",
  },
  sendBtnDisabled: { opacity: 0.3 },
  sendIcon: { color: "#fff", fontSize: 16 },
});
