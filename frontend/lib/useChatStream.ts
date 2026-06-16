"use client";
import { useCallback, useState } from "react";
import { API_BASE } from "./api";
import type { ChatEvent } from "./types";

export function useChatStream() {
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [answer, setAnswer] = useState<string>("");
  const [highlight, setHighlight] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const ask = useCallback(async (question: string) => {
    setEvents([]);
    setAnswer("");
    setHighlight([]);
    setBusy(true);
    try {
      const resp = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // SSE frames are separated by a blank line; servers may use \n or \r\n.
        const frames = buf.split(/\r?\n\r?\n/);
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.split(/\r?\n/).find((l) => l.startsWith("data:"));
          if (!line) continue;
          const evt = JSON.parse(line.slice(5).trim()) as ChatEvent;
          setEvents((prev) => [...prev, evt]);
          if (evt.type === "answer") {
            setAnswer(evt.content);
            setHighlight(evt.highlight);
          }
        }
      }
    } finally {
      setBusy(false);
    }
  }, []);

  return { events, answer, highlight, busy, ask };
}
